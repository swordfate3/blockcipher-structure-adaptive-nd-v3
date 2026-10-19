# Reviewed extraction: src/blockcipher_nd_v3/ciphers/expanded.py
# Source commit: d7a8698f4c474e1899f848b5de7a5abb570a13ea
# Source SHA-256: 5cdf3a57b48c5862f6a33339099265b27e73ca5300764afedf3206e92589e4f7

"""Bounded vectorized encryption for the four supported 64-bit tasks.

Adjacent equal-key blocks share schedules; chunking bounds working memory.
All rounds operate on fixed-width NumPy arrays, not scalar encryptors.
"""

from __future__ import annotations

from collections.abc import Sequence
from operator import index

import numpy as np
from numpy.typing import NDArray

from .present80 import PRESENT_SBOX
from .simeck64 import SIMECK_Z1
from .simon64 import SIMON_Z3
from .skinny64 import SKINNY64_SBOX, SKINNY64_SHIFT_ROWS, SKINNY64_TK_PERMUTATION

type Words = NDArray[np.uint64]
type Rows = NDArray[np.intp] | None

_CHUNK_SIZE = 65536
_MASK32 = np.uint64((1 << 32) - 1)
# block bits, key bits, minimum rounds, maximum rounds
_LIMITS = {
    "simon64": (64, 128, 1, 44),
    "present80": (64, 80, 1, 31),
    "skinny64": (64, 64, 0, 32),
    "simeck64": (64, 128, 1, 44),
}
_SKINNY_SBOX = np.asarray(SKINNY64_SBOX, dtype=np.uint8)
_PRESENT_SBOX = np.asarray(PRESENT_SBOX, dtype=np.uint64)


def _present_sp_table() -> Words:
    """Fuse the PRESENT S-box and bit permutation into eight byte lookup tables."""
    table = np.empty((8, 256), dtype=np.uint64)
    for position in range(8):
        for value in range(256):
            substituted = PRESENT_SBOX[value & 15] | (PRESENT_SBOX[value >> 4] << 4)
            output = 0
            for bit in range(8):
                source = 8 * position + bit
                target = 63 if source == 63 else (16 * source) % 63
                output |= ((substituted >> bit) & 1) << target
            table[position, value] = output
    return table


_PRESENT_SP = _present_sp_table()


def _integer(value: int, label: str) -> int:
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be an integer") from exc


def _pack(values: Sequence[int], bits: int, label: str) -> Words:
    """Validate before conversion; split 80/128-bit integers into little-endian limbs."""
    byte_count = 8 if bits <= 64 else 16
    limit = 1 << bits

    def encoded() -> bytes:
        parts: list[bytes] = []
        for value in values:
            integer = _integer(value, label)
            if not 0 <= integer < limit:
                raise ValueError(f"{label} must fit in {bits} bits")
            parts.append(integer.to_bytes(byte_count, "little"))
        return b"".join(parts)

    # Explicit byte order also preserves semantics on a big-endian NumPy host.
    return (
        np.frombuffer(encoded(), dtype="<u8")
        .reshape(-1, byte_count // 8)
        .T.astype(np.uint64, copy=False)
    )


def _key_rows(keys: Words) -> tuple[Words, Rows]:
    """Coalesce adjacent equal keys without sorting or losing the 128-bit high limb."""
    changed = np.empty(keys.shape[1], dtype=np.bool_)
    changed[0] = True
    changed[1:] = np.any(keys[:, 1:] != keys[:, :-1], axis=0)
    count = int(np.count_nonzero(changed))
    if count == keys.shape[1]:
        return keys, None
    if count == 1:
        return keys[:, :1], None  # NumPy broadcasts the single shared round key.
    rows = np.cumsum(changed, dtype=np.intp) - 1
    return keys[:, changed], rows


def _for_blocks(word: Words, rows: Rows) -> Words:
    return word if rows is None else word[rows]


def _rol(word: Words, amount: int, width: int = 32) -> Words:
    amount %= width
    return ((word << np.uint64(amount)) | (word >> np.uint64(width - amount))) & np.uint64(
        (1 << width) - 1
    )


def _ror(word: Words, amount: int, width: int = 32) -> Words:
    return _rol(word, width - amount, width)


def _words32(limbs: Words) -> list[Words]:
    return [limbs[0] & _MASK32, limbs[0] >> 32, limbs[1] & _MASK32, limbs[1] >> 32]


def _simon(rounds: int, keys: Words, plaintexts: Words, rows: Rows) -> Words:
    words = _words32(keys)
    x, y = plaintexts[0] >> 32, plaintexts[0] & _MASK32
    for counter in range(rounds):
        if counter < 4:
            key = words[counter]
        else:
            temporary = _ror(words[3], 3) ^ words[1]
            temporary ^= _ror(temporary, 1)
            key = words[0] ^ temporary ^ np.uint64(0xFFFFFFFC ^ int(SIMON_Z3[(counter - 4) % 62]))
            words = [words[1], words[2], words[3], key]
        x, y = y ^ (_rol(x, 1) & _rol(x, 8)) ^ _rol(x, 2) ^ _for_blocks(key, rows), x
    return np.asarray([(x << 32) | y], dtype=np.uint64)


def _present(rounds: int, keys: Words, plaintexts: Words, rows: Rows) -> Words:
    low, high = keys[0], keys[1]
    state = plaintexts[0]
    for counter in range(1, rounds + 1):
        state = state ^ _for_blocks((high << 48) | (low >> 16), rows)
        permuted = np.zeros_like(state)
        for position in range(8):
            permuted |= _PRESENT_SP[position, (state >> (8 * position)) & 0xFF]
        state = permuted
        # Rotate the full 80-bit register left by 61 (right by 19).
        low, high = (low << 61) | (high << 45) | (low >> 19), (low >> 3) & 0xFFFF
        high = (high & 0xFFF) | (_PRESENT_SBOX[high >> 12] << 12)
        low = low ^ (counter << 15)
    state = state ^ _for_blocks((high << 48) | (low >> 16), rows)
    return np.asarray([state], dtype=np.uint64)


def _skinny(rounds: int, keys: Words, plaintexts: Words, rows: Rows) -> Words:
    state = np.asarray([(plaintexts[0] >> s) & 15 for s in range(60, -1, -4)], dtype=np.uint8)
    tweakey = np.asarray([(keys[0] >> s) & 15 for s in range(60, -1, -4)], dtype=np.uint8)
    constant = 1
    for _ in range(rounds):
        state = _SKINNY_SBOX[state]
        state[0] ^= constant & 15
        state[4] ^= (constant >> 4) & 3
        state[8] ^= 2
        state[:8] ^= tweakey[:8] if rows is None else tweakey[:8, rows]
        shifted = state[list(SKINNY64_SHIFT_ROWS)]
        state[:4] = shifted[:4] ^ shifted[8:12] ^ shifted[12:]
        state[4:8] = shifted[:4]
        state[8:12] = shifted[4:8] ^ shifted[8:12]
        state[12:] = shifted[:4] ^ shifted[8:12]
        tweakey = tweakey[list(SKINNY64_TK_PERMUTATION)]
        constant = ((constant << 1) & 0x3F) | (((constant >> 5) ^ (constant >> 4) ^ 1) & 1)
    output = np.zeros(plaintexts.shape[1], dtype=np.uint64)
    for cell in state:
        output = (output << 4) | cell.astype(np.uint64)
    return np.asarray([output], dtype=np.uint64)


def _simeck(rounds: int, keys: Words, plaintexts: Words, rows: Rows) -> Words:
    words = _words32(keys)
    x, y = plaintexts[0] >> 32, plaintexts[0] & _MASK32
    for counter in range(rounds):
        feedback = (_rol(x, 5) & x) ^ _rol(x, 1)
        x, y = y ^ feedback ^ _for_blocks(words[0], rows), x
        if counter + 1 < rounds:
            feedback = (_rol(words[1], 5) & words[1]) ^ _rol(words[1], 1)
            next_word = words[0] ^ feedback ^ np.uint64(0xFFFFFFFC ^ SIMECK_Z1[counter % 63])
            words = [words[1], words[2], words[3], next_word]
    return np.asarray([(x << 32) | y], dtype=np.uint64)


_KERNELS = {
    "simon64": _simon,
    "present80": _present,
    "skinny64": _skinny,
    "simeck64": _simeck,
}


def encrypt_expanded(
    cipher: str, rounds: int, keys: Sequence[int], plaintexts: Sequence[int]
) -> list[int]:
    """Encrypt one 64-bit block per key, returning Python ints.

    All inputs are range-checked before narrowing. Empty input still validates the
    cipher and round count. Zero rounds are accepted only for SKINNY, as in v2.
    Inputs may be lists/tuples or one-dimensional integer NumPy arrays; they are
    never mutated. Internal chunks bound memory, not the length of this interface.
    """
    if cipher not in _LIMITS:
        raise ValueError(f"unsupported cipher: {cipher}")
    block_bits, key_bits, minimum, maximum = _LIMITS[cipher]
    rounds = _integer(rounds, "rounds")
    if not minimum <= rounds <= maximum:
        raise ValueError(f"{cipher} rounds must be between {minimum} and {maximum}")
    if len(keys) != len(plaintexts):
        raise ValueError("expected one key per plaintext")
    output: list[int] = []
    kernel = _KERNELS[cipher]
    for start in range(0, len(plaintexts), _CHUNK_SIZE):
        packed_keys = _pack(keys[start : start + _CHUNK_SIZE], key_bits, "key")
        packed_plaintexts = _pack(plaintexts[start : start + _CHUNK_SIZE], block_bits, "plaintext")
        distinct_keys, rows = _key_rows(packed_keys)
        encrypted = kernel(rounds, distinct_keys, packed_plaintexts, rows)
        output.extend(encrypted[0].tolist())
    return output

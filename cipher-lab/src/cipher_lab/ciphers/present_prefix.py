"""Bounded PRESENT-80 batches, with selected independently whitened prefixes."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_SBOX = (0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
         0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2)
_MASK80 = (1 << 80) - 1


def _round_keys(key_words: tuple[int, int, int], last: int) -> tuple[np.uint64, ...]:
    low, middle, high = key_words
    register = low | (middle << 32) | (high << 64)
    keys = [np.uint64(register >> 16)]
    for r in range(1, last + 1):
        register = ((register << 61) | (register >> 19)) & _MASK80
        top = register >> 76
        register = (register & ((1 << 76) - 1)) | (_SBOX[top] << 76)
        register ^= r << 15
        keys.append(np.uint64(register >> 16))
    return tuple(keys)


def _p_layer_table() -> NDArray[np.uint64]:
    # Bit b in nibble i moves to 16*b + i; this also fixes bit 63.
    # Spread the S-box outputs of nibbles 0 and 1. Byte j's contribution
    # is the same layout shifted left by 2*j, disjoint from all other bytes.
    spread = tuple(sum(((value >> b) & 1) << (16 * b) for b in range(4)) for value in _SBOX)
    return np.array([spread[v & 15] | (spread[v >> 4] << 1) for v in range(256)],
                    dtype=np.uint64)


def _encrypt_into(state: NDArray[np.uint64], key_words: tuple[int, int, int],
                  rounds: tuple[int, ...], out: NDArray[np.uint64]) -> None:
    keys = _round_keys(key_words, rounds[-1])
    table = _p_layer_table()
    byte = np.empty_like(state)
    permuted = np.empty_like(state)
    col = 0
    for r in range(1, rounds[-1] + 1):
        state ^= keys[r - 1]
        permuted.fill(0)
        for j in range(8):
            np.right_shift(state, np.uint64(8 * j), out=byte)
            byte &= np.uint64(255)
            permuted |= table[byte] << np.uint64(2 * j)
        state, permuted = permuted, state
        if r == rounds[col]:
            # Emit Kr+1-whitened output, but preserve naked state for the next round.
            np.bitwise_xor(state, keys[r], out=out[:, col])
            col += 1


def encrypt_prefixes(points: NDArray[np.uint64], key_words: tuple[int, int, int],
                     rounds: tuple[int, ...]) -> NDArray[np.uint64]:
    """Return a fresh native C-order matrix with one column per requested round.

    The points must be a native contiguous uint64 vector of 1..65536 blocks.
    Key words are low-first uint32, uint32, uint16; rounds strictly increase
    within 1..31. Each column includes that reduced cipher's final whitening.
    """
    if (type(points) is not np.ndarray or points.dtype != np.dtype(np.uint64)
            or points.ndim != 1 or not points.flags.c_contiguous or not 1 <= len(points) <= 65536):
        raise ValueError('points must be 1..65536 native contiguous uint64 blocks')
    if (type(key_words) is not tuple or len(key_words) != 3
            or any(type(w) is not int or not 0 <= w < 1 << 32 for w in key_words)
            or key_words[2] >= 1 << 16):
        raise ValueError('key_words must be low-first uint32/uint32/uint16')
    if (type(rounds) is not tuple or not 1 <= len(rounds) <= 31
            or any(type(r) is not int or not 1 <= r <= 31 for r in rounds)
            or any(a >= b for a, b in zip(rounds, rounds[1:]))):
        raise ValueError('rounds must be strictly increasing in 1..31')
    out = np.empty((len(points), len(rounds)), dtype=np.uint64)
    _encrypt_into(points.copy(), key_words, rounds, out)
    return out

"""Pure SKINNY-64/64 adapter; static extraction only, NOT runtime/KAT tested.

Two low-first uint32 key words. State is a row-major MS-nibble-first 4x4
matrix. Only complete blocks returned; this is NOT the public key-free backstep.
"""
from __future__ import annotations

_SBOX = (0xC, 0x6, 0x9, 0x0, 0x1, 0xA, 0x2, 0xB,
         0x3, 0x8, 0x5, 0xD, 0x4, 0xE, 0x7, 0xF)
_TK_PERMUTATION = (9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7)
_SHIFT_ROWS = (0, 1, 2, 3, 7, 4, 5, 6, 10, 11, 8, 9, 13, 14, 15, 12)


def encrypt_many(points: tuple[int, ...], key_words: tuple[int, ...],
                 rounds: int) -> tuple[int, ...]:
    """1..65536 blocks, 1..32 rounds. Production execution requires authorization."""
    if type(rounds) is not int or not 1 <= rounds <= 32:
        raise ValueError("rounds must be in 1..32")
    if (type(key_words) is not tuple or len(key_words) != 2
            or any(type(word) is not int or not 0 <= word < (1 << 32) for word in key_words)):
        raise ValueError("key_words must be two uint32 words")
    if (type(points) is not tuple or not 1 <= len(points) <= 65536
            or any(type(point) is not int or not 0 <= point < (1 << 64) for point in points)):
        raise ValueError("points must contain 1..65536 uint64 blocks")
    key = key_words[0] | (key_words[1] << 32)
    tweakey = [(key >> shift) & 15 for shift in range(60, -1, -4)]
    constant = 1
    schedule: list[tuple[tuple[int, ...], int]] = []
    for _ in range(rounds):
        schedule.append((tuple(tweakey[:8]), constant))
        tweakey = [tweakey[index] for index in _TK_PERMUTATION]
        constant = ((constant << 1) & 63) | (((constant >> 5) ^ (constant >> 4) ^ 1) & 1)
    result: list[int] = []
    for point in points:
        state = [(point >> shift) & 15 for shift in range(60, -1, -4)]
        for round_key, constant in schedule:
            state = [_SBOX[cell] for cell in state]
            state[0] ^= constant & 15
            state[4] ^= (constant >> 4) & 3
            state[8] ^= 2
            for index in range(8):
                state[index] ^= round_key[index]
            shifted = [state[index] for index in _SHIFT_ROWS]
            for column in range(4):
                a, b, c, d = (shifted[column + 4 * row] for row in range(4))
                state[column] = a ^ c ^ d
                state[4 + column] = a
                state[8 + column] = b ^ c
                state[12 + column] = a ^ c
        packed = 0
        for cell in state:
            packed = (packed << 4) | cell
        result.append(packed)
    return tuple(result)

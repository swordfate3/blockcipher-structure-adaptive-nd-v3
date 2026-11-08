"""Pure PRESENT-80 batch adapter; static extraction only, NOT runtime/KAT tested.

Key words are three low-first uint32s; the highest word fits uint16 (80 bits).
State is one uint64, with final whitening after the requested reduced rounds.
No secret state is retained; no dependencies on collection, training or models.
"""
from __future__ import annotations

_SBOX = (0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
         0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2)
_MASK80 = (1 << 80) - 1


def encrypt_many(points: tuple[int, ...], key_words: tuple[int, ...],
                 rounds: int) -> tuple[int, ...]:
    """1..65536 blocks, 1..31 rounds. Production execution requires authorization."""
    if type(rounds) is not int or not 1 <= rounds <= 31:
        raise ValueError("rounds must be in 1..31")
    if (type(key_words) is not tuple or len(key_words) != 3
            or any(type(word) is not int or not 0 <= word < (1 << 32) for word in key_words)
            or key_words[2] >= (1 << 16)):
        raise ValueError("key_words must be low-first uint32/uint32/uint16")
    if (type(points) is not tuple or not 1 <= len(points) <= 65536
            or any(type(point) is not int or not 0 <= point < (1 << 64) for point in points)):
        raise ValueError("points must contain 1..65536 uint64 blocks")
    register = sum(word << (32 * index) for index, word in enumerate(key_words))
    schedule = [register >> 16]
    for counter in range(1, rounds + 1):
        register = ((register << 61) | (register >> 19)) & _MASK80
        top = (register >> 76) & 15
        register = (register & ~(15 << 76)) | (_SBOX[top] << 76)
        register ^= (counter & 31) << 15
        schedule.append(register >> 16)
    result: list[int] = []
    for point in points:
        state = point
        for key in schedule[:-1]:
            state ^= key
            substituted = sum(_SBOX[(state >> (4 * index)) & 15] << (4 * index)
                              for index in range(16))
            state = ((substituted >> 63) & 1) << 63
            for bit in range(63):
                state |= ((substituted >> bit) & 1) << ((16 * bit) % 63)
        result.append(state ^ schedule[-1])
    return tuple(result)

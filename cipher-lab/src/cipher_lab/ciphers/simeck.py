"""Pure SIMECK64/128 batch adapter, statically extracted; NOT runtime/KAT tested.

Complete uint64 blocks: high left/low right uint32. Four key words are uint32,
least-significant first, identical to the SIMON adapter convention. No sampling,
resource accounting, data or model dependency. Source SHA in docs/provenance.json.
"""
from __future__ import annotations

_MASK = 0xFFFFFFFF
_Z1 = "111111000001000011000101001111010001110010010110111011001101010"


def _rol(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (32 - shift))) & _MASK


def encrypt_many(points: tuple[int, ...], key_words: tuple[int, ...],
                 rounds: int) -> tuple[int, ...]:
    """1..65536 blocks, 1..44 rounds. Production execution requires authorization."""
    if type(rounds) is not int or not 1 <= rounds <= 44:
        raise ValueError("rounds must be in 1..44")
    if (type(key_words) is not tuple or len(key_words) != 4
            or any(type(word) is not int or not 0 <= word <= _MASK for word in key_words)):
        raise ValueError("key_words must be four uint32 words")
    if (type(points) is not tuple or not 1 <= len(points) <= 65536
            or any(type(point) is not int or not 0 <= point < (1 << 64) for point in points)):
        raise ValueError("points must contain 1..65536 uint64 blocks")
    words = list(key_words)
    schedule = [words[0]]
    for index in range(rounds - 1):
        feedback = (_rol(words[1], 5) & words[1]) ^ _rol(words[1], 1)
        next_word = words[0] ^ feedback ^ (_MASK ^ 3) ^ int(_Z1[index % 63])
        words = [words[1], words[2], words[3], next_word & _MASK]
        schedule.append(words[0])
    result: list[int] = []
    for point in points:
        left, right = point >> 32, point & _MASK
        for key in schedule:
            feedback = (_rol(left, 5) & left) ^ _rol(left, 1)
            left, right = (right ^ feedback ^ key) & _MASK, left
        result.append((left << 32) | right)
    return tuple(result)

"""Standard-library SIMON64/128 batch adapter, not yet KAT/runtime validated.

Statically adapted from the source recorded in docs/provenance.json. State
packs x in the high 32 bits and y in the low 32 bits; key_words are uint32s
in least-significant-first order. No private state is exposed or retained.
"""
from __future__ import annotations

_MASK = 0xFFFFFFFF
_Z3 = "11011011101011000110010111100000010010001010011100110100001111"


def _ror(word: int, distance: int) -> int:
    return ((word >> distance) | (word << (32 - distance))) & _MASK


def encrypt_many(points: tuple[int, ...], key_words: tuple[int, ...],
                 rounds: int) -> tuple[int, ...]:
    """Complete-block encryption only; 1..65536 blocks, 1..44 rounds.

    This implementation has only been statically reviewed in this phase;
    engineering tests must inject a fake oracle instead of calling it.
    """
    if type(rounds) is not int or not 1 <= rounds <= 44:
        raise ValueError("rounds must be in 1..44")
    if (type(key_words) is not tuple or len(key_words) != 4
            or any(type(word) is not int or not 0 <= word <= _MASK for word in key_words)):
        raise ValueError("key_words must be four uint32 words")
    if (type(points) is not tuple or not 1 <= len(points) <= 65536
            or any(type(point) is not int or not 0 <= point < (1 << 64) for point in points)):
        raise ValueError("points must contain 1..65536 uint64 blocks")
    schedule = list(key_words)
    for index in range(4, rounds):
        temporary = _ror(schedule[index - 1], 3) ^ schedule[index - 3]
        temporary ^= _ror(temporary, 1)
        schedule.append(schedule[index - 4] ^ temporary ^ 0xFFFFFFFC
                        ^ int(_Z3[(index - 4) % 62]))
    result: list[int] = []
    for point in points:
        x, y = point >> 32, point & _MASK
        for round_key in schedule[:rounds]:
            rotated1 = ((x << 1) | (x >> 31)) & _MASK
            rotated8 = ((x << 8) | (x >> 24)) & _MASK
            rotated2 = ((x << 2) | (x >> 30)) & _MASK
            x, y = (y ^ (rotated1 & rotated8) ^ rotated2 ^ round_key) & _MASK, x
        result.append((x << 32) | y)
    return tuple(result)

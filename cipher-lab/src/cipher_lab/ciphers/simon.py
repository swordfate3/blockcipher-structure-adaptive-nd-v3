"""Scalar and NumPy SIMON64/128 adapters, not yet KAT/runtime validated.

Statically adapted from the source recorded in docs/provenance.json. State
packs x in the high 32 bits and y in the low 32 bits; key_words are uint32s
in least-significant-first order. No private state is exposed or retained.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

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


def _round_words(x: NDArray[np.uint32], y: NDArray[np.uint32],
                 round_key: int) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    """Public one-round uint32 arithmetic; fresh outputs, no input mutation."""
    import numpy as np

    for words in (x, y):
        if not isinstance(words, np.ndarray) or words.dtype != np.dtype(np.uint32):
            raise TypeError("round words must be native uint32 arrays")
        if words.ndim != 1 or not words.flags.c_contiguous or words.size > 65536:
            raise ValueError("round words must be contiguous vectors of at most 65536")
    if x.shape != y.shape:
        raise ValueError("round word shapes must agree")
    if type(round_key) is not int or not 0 <= round_key <= _MASK:
        raise ValueError("round_key must be uint32")
    rotated1 = (x << np.uint32(1)) | (x >> np.uint32(31))
    rotated8 = (x << np.uint32(8)) | (x >> np.uint32(24))
    rotated2 = (x << np.uint32(2)) | (x >> np.uint32(30))
    return y ^ (rotated1 & rotated8) ^ rotated2 ^ np.uint32(round_key), x.copy()


def encrypt_batch(points: NDArray[np.uint64], key_words: tuple[int, ...],
                  rounds: int) -> NDArray[np.uint64]:
    """Complete NumPy SIMON64/128; STATIC REVIEW ONLY, no KAT/runtime validation.

    Strict native C-contiguous one-dimensional uint64 blocks (0..65536).
    Low-first four-word key and high-x/low-y packing match encrypt_many.
    No caller mutation, state traces, resource policy or sampling dependencies.
    """
    import numpy as np

    if type(rounds) is not int or not 1 <= rounds <= 44:
        raise ValueError("rounds must be in 1..44")
    if (type(key_words) is not tuple or len(key_words) != 4
            or any(type(word) is not int or not 0 <= word <= _MASK for word in key_words)):
        raise ValueError("key_words must be four uint32 words")
    if not isinstance(points, np.ndarray) or points.dtype != np.dtype(np.uint64):
        raise TypeError("points must be a native uint64 array")
    if points.ndim != 1 or not points.flags.c_contiguous or points.size > 65536:
        raise ValueError("points must be a contiguous vector of at most 65536 blocks")
    schedule = np.empty(max(4, rounds), dtype=np.uint32)
    schedule[:4] = key_words
    for index in range(4, rounds):
        previous = schedule[index - 1]
        temporary = ((previous >> np.uint32(3)) | (previous << np.uint32(29)))
        temporary ^= schedule[index - 3]
        temporary ^= (temporary >> np.uint32(1)) | (temporary << np.uint32(31))
        schedule[index] = (schedule[index - 4] ^ temporary ^ np.uint32(0xFFFFFFFC)
                           ^ np.uint32(int(_Z3[(index - 4) % 62])))
    x = (points >> np.uint64(32)).astype(np.uint32)
    y = (points & np.uint64(_MASK)).astype(np.uint32)
    for round_key in schedule[:rounds]:
        x, y = _round_words(x, y, int(round_key))
    return (x.astype(np.uint64) << np.uint64(32)) | y.astype(np.uint64)

"""Pure SIMON32/64: low-word-first key, high x / low y block packing.

Independent scalar and uint16-vector arithmetic. Public vector execution is
reserved for the approved run; see docs/provenance.json for static sources.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

Key = tuple[int, int, int, int]
_MASK = 0xffff
_Z0 = '11111010001001010110000111001101111101000100101011000011100110'


def _parameters(key_words: object, rounds: object) -> None:
    if type(rounds) is not int or not 1 <= rounds <= 32:
        raise ValueError('rounds must be in 1..32')
    if (type(key_words) is not tuple or len(key_words) != 4
            or any(type(w) is not int or not 0 <= w <= _MASK for w in key_words)):
        raise ValueError('key_words must be four uint16 words')


def _scalar_input(point: object) -> None:
    if type(point) is not int or not 0 <= point < 2**32:
        raise ValueError('point must be a uint32 block')


def _batch_input(points: object) -> None:
    import numpy as np

    if type(points) is not np.ndarray or points.dtype != np.dtype(np.uint32):
        raise TypeError('points must be a native uint32 array')
    if points.ndim != 1 or not points.flags.c_contiguous or points.size > 65536:
        raise ValueError('points must be a contiguous vector of at most 65536 blocks')


def encrypt_scalar(point: int, key_words: Key, rounds: int) -> int:
    _parameters(key_words, rounds)
    _scalar_input(point)
    keys = list(key_words)
    for i in range(4, rounds):
        k = keys[i - 1]
        t = (((k >> 3) | (k << 13)) & _MASK) ^ keys[i - 3]
        t ^= ((t >> 1) | (t << 15)) & _MASK
        keys.append(keys[i - 4] ^ t ^ 0xfffc ^ int(_Z0[(i - 4) % 62]))
    x, y = point >> 16, point & _MASK
    for k in keys[:rounds]:
        a = ((x << 1) | (x >> 15)) & _MASK
        b = ((x << 8) | (x >> 8)) & _MASK
        c = ((x << 2) | (x >> 14)) & _MASK
        x, y = y ^ (a & b) ^ c ^ k, x
    return (x << 16) | y


def encrypt_batch(points: NDArray[np.uint32], key_words: Key, rounds: int) -> NDArray[np.uint32]:
    """Encrypt 0..65536 blocks without mutating or aliasing the input."""
    import numpy as np

    _parameters(key_words, rounds)
    _batch_input(points)
    keys = np.empty(max(4, rounds), dtype=np.uint16)
    keys[:4] = key_words
    for i in range(4, rounds):
        k = keys[i - 1]
        t = ((k >> np.uint16(3)) | (k << np.uint16(13))) ^ keys[i - 3]
        keys[i] = (keys[i - 4] ^ t ^ ((t >> np.uint16(1)) | (t << np.uint16(15)))
                   ^ np.uint16(0xfffc) ^ np.uint16(int(_Z0[(i - 4) % 62])))
    x = (points >> np.uint32(16)).astype(np.uint16)
    y = points.astype(np.uint16)
    for k in keys[:rounds]:
        a = (x << np.uint16(1)) | (x >> np.uint16(15))
        b = (x << np.uint16(8)) | (x >> np.uint16(8))
        c = (x << np.uint16(2)) | (x >> np.uint16(14))
        x, y = y ^ (a & b) ^ c ^ k, x
    return cast('NDArray[np.uint32]', (x.astype(np.uint32) << np.uint32(16)) | y.astype(np.uint32))

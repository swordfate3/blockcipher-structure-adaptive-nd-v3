"""Bounded inversion of the complete (K3-whitened) PRESENT-80 round-2 output."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import present_prefix

_INV_SBOX = tuple(present_prefix._SBOX.index(i) for i in range(16))
# For each pre-pLayer nibble i, read its four bits from the four planes.
_NIBBLE_SOURCES = tuple(tuple(i + 16 * bit for bit in range(4)) for i in range(16))


def _invert_into(state: NDArray[np.uint64], key_words: tuple[int, int, int]) -> None:
    k1, k2, k3 = present_prefix._round_keys(key_words, 2)
    state ^= k3
    table = np.array(_INV_SBOX, dtype=np.uint64)
    nibble = np.empty_like(state)
    work = np.empty_like(state)
    previous = np.empty_like(state)
    for key in (k2, k1):
        previous.fill(0)
        for i, sources in enumerate(_NIBBLE_SOURCES):
            nibble.fill(0)
            for bit, source in enumerate(sources):
                np.right_shift(state, np.uint64(source), out=work)
                work &= np.uint64(1)
                nibble |= work << np.uint64(bit)
            previous |= table[nibble] << np.uint64(4 * i)
        # Invert S after undoing pLayer, then undo this round's key XOR.
        np.bitwise_xor(previous, key, out=state)


def invert_second_output(points: NDArray[np.uint64],
                         key_words: tuple[int, int, int]) -> NDArray[np.uint64]:
    """Return fresh plaintexts for a native contiguous uint64 vector of 1..65536 S2s.

    Key words are low-first uint32, uint32, uint16. Neither input is changed.
    """
    if (type(points) is not np.ndarray or points.dtype != np.dtype(np.uint64)
            or points.ndim != 1 or not points.flags.c_contiguous or not 1 <= len(points) <= 65536):
        raise ValueError('points must be 1..65536 native contiguous uint64 blocks')
    if (type(key_words) is not tuple or len(key_words) != 3
            or any(type(w) is not int or not 0 <= w < 1 << 32 for w in key_words)
            or key_words[2] >= 1 << 16):
        raise ValueError('key_words must be low-first uint32/uint32/uint16')
    out = points.copy()
    _invert_into(out, key_words)
    return out

"""Bounded forward PRESENT-80 with one independent key per pair."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cipher_lab.ciphers.present_prefix import _SBOX, _p_layer_table

MAX_PAIRS = 32_768


def encrypt_pairs(points: NDArray[np.uint64], low: NDArray[np.uint64],
                  high: NDArray[np.uint16]) -> NDArray[np.uint64]:
    """Encrypt 1..32768 native contiguous pairs to whitened round-seven outputs."""
    if (type(points) is not np.ndarray or points.dtype != np.uint64
            or points.ndim != 2 or points.shape[1] != 2
            or not points.flags.c_contiguous or not 1 <= len(points) <= MAX_PAIRS):
        raise ValueError('INVALID_PAIR_POINTS')
    for words, dtype in ((low, np.uint64), (high, np.uint16)):
        if (type(words) is not np.ndarray or words.dtype != dtype
                or words.shape != (len(points),) or not words.flags.c_contiguous):
            raise ValueError('INVALID_PAIR_KEYS')

    table = _p_layer_table()
    sbox = np.asarray(_SBOX, dtype=np.uint64)
    state = points.copy(order='C')
    scratch = np.empty_like(state)
    byte = np.empty_like(state)
    lo, hi = low, high.astype(np.uint64)
    for counter in range(1, 8):
        state ^= ((hi << 48) | (lo >> 16))[:, None]
        scratch.fill(0)
        for j in range(8):
            np.right_shift(state, np.uint64(8 * j), out=byte)
            byte &= np.uint64(255)
            scratch |= table[byte] << np.uint64(2 * j)
        state, scratch = scratch, state
        # Rotate the 80-bit register right by 19, then update its top nibble.
        lo, hi = ((lo >> np.uint64(19)) | (hi << np.uint64(45)) | (lo << np.uint64(61)),
                  (lo >> np.uint64(3)) & np.uint64(0xFFFF))
        hi = (hi & np.uint64(0xFFF)) | (sbox[hi >> np.uint64(12)] << np.uint64(12))
        lo ^= np.uint64(counter << 15)
    state ^= ((hi << 48) | (lo >> 16))[:, None]
    return state

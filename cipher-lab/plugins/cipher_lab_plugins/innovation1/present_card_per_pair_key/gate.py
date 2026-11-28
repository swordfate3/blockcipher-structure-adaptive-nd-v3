"""Meter the existing public KAT and two distinct public pair keys."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from cipher_lab.execution import Meter

from ..prefix_bridge.kat import Key, Words
from .config import ROUNDS

PairEncrypt = Callable[[Words, Words, NDArray[np.uint16]], Words]
_POINTS = ((0, 1), (0, 1))
_KEYS: tuple[Key, Key] = ((0, 0, 0), (0x06070809, 0x02030405, 1))


def check(meter: Meter, encrypt: PairEncrypt, scalar: Callable[[int, Key, int], int]) -> None:
    expected = np.empty((2, 2), dtype=np.uint64)
    for row, (pair, key) in enumerate(zip(_POINTS, _KEYS, strict=True)):
        for col, point in enumerate(pair):
            meter.reserve('real_blocks', 1)
            word = scalar(point, key, ROUNDS[0])
            if type(word) is not int or not 0 <= word < 1 << 64:
                raise ValueError('PUBLIC_PAIR_REFERENCE')
            expected[row, col] = word
            meter.complete('real_blocks', 1)
    points = np.array(_POINTS, dtype=np.uint64)
    low = np.array([a | (b << 32) for a, b, _ in _KEYS], dtype=np.uint64)
    high = np.array([c for _, _, c in _KEYS], dtype=np.uint16)
    original_points, original_low, original_high = points.copy(), low.copy(), high.copy()
    meter.reserve('real_blocks', 4)
    result = encrypt(points, low, high)
    if (type(result) is not np.ndarray or result.dtype != expected.dtype
            or not result.flags.c_contiguous or not np.array_equal(result, expected)
            or any(np.shares_memory(value, result) for value in (points, low, high))
            or points.dtype != original_points.dtype or not np.array_equal(points, original_points)
            or low.dtype != original_low.dtype or not np.array_equal(low, original_low)
            or high.dtype != original_high.dtype or not np.array_equal(high, original_high)):
        raise ValueError('PUBLIC_PAIR_KEY_GATE')
    meter.complete('real_blocks', 4)
    meter.check()


def production(meter: Meter) -> None:
    from cipher_lab.ciphers import present

    from ..mask_aux import kat
    from .cipher import encrypt_pairs

    def scalar(point: int, key: Key, rounds: int) -> int:
        result, = present.encrypt_many((point,), key, rounds)
        return result

    kat.production(meter)
    check(meter, encrypt_pairs, scalar)

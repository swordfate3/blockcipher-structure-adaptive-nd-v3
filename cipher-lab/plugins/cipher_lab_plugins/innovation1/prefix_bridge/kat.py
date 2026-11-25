"""Public two-path gate: one official KAT plus six reduced-round comparisons."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from cipher_lab.execution import Meter

from .config import ROUNDS

Words = NDArray[np.uint64]
Key = tuple[int, int, int]
_ZERO_KAT = 0x5579C1387B228445
_PUBLIC_KEY: Key = (0x06070809, 0x02030405, 0x0001)
_PUBLIC_POINT = 0x0123456789ABCDEF


def check(meter: Meter, scalar: Callable[[int, Key, int], int],
          prefixes: Callable[[Words, Key, tuple[int, ...]], Words]) -> None:
    """Charge one request per arm and vector; stop at the first failure."""
    for r in (31, *ROUNDS):
        point, key = (0, (0, 0, 0)) if r == 31 else (_PUBLIC_POINT, _PUBLIC_KEY)
        meter.reserve('real_blocks', 1)
        expected = scalar(point, key, r)
        if (type(expected) is not int or not 0 <= expected < 1 << 64
                or (r == 31 and expected != _ZERO_KAT)):
            raise ValueError('public scalar PRESENT gate')
        meter.complete('real_blocks', 1)

        supplied = np.array([point], dtype=np.uint64)
        meter.reserve('real_blocks', 1)
        result = prefixes(supplied, key, (r,))
        if (type(result) is not np.ndarray or result.dtype != np.dtype(np.uint64)
                or result.shape != (1, 1) or not result.flags.c_contiguous
                or np.shares_memory(supplied, result) or int(result[0, 0]) != expected
                or supplied.shape != (1,) or supplied.dtype != np.dtype(np.uint64)
                or int(supplied[0]) != point):
            raise ValueError('public batch PRESENT gate')
        meter.complete('real_blocks', 1)
    meter.check()


def production(meter: Meter) -> None:
    from cipher_lab.ciphers import present, present_prefix

    def scalar(point: int, key: Key, rounds: int) -> int:
        result, = present.encrypt_many((point,), key, rounds)
        return result

    check(meter, scalar, present_prefix.encrypt_prefixes)

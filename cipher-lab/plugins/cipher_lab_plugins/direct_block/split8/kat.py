"""Single, metered public 32-round SIMON32/64 vector (one block per interface)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from cipher_lab.execution import Meter

Words = NDArray[np.uint32]
Key = tuple[int, int, int, int]
_PUBLIC_KEY: Key = (0x0100, 0x0908, 0x1110, 0x1918)
_POINT = 0x65656877
_EXPECTED = 0xc69be9bb


def check(meter: Meter, scalar: Callable[[int, Key, int], int],
          batch: Callable[[Words, Key, int], Words]) -> None:
    meter.reserve('real_blocks', 1)
    a = scalar(_POINT, _PUBLIC_KEY, 32)
    if type(a) is not int or a != _EXPECTED:
        raise ValueError('public scalar KAT')
    meter.complete('real_blocks', 1)
    supplied = np.array([_POINT], dtype=np.uint32)
    meter.reserve('real_blocks', 1)
    b = batch(supplied, _PUBLIC_KEY, 32)
    if (type(b) is not np.ndarray or b.dtype != np.uint32 or b.shape != (1,)
            or not b.flags.c_contiguous or int(b[0]) != _EXPECTED
            or supplied.shape != (1,) or supplied.dtype != np.uint32 or int(supplied[0]) != _POINT
            or np.shares_memory(supplied, b)):
        raise ValueError('public batch KAT')
    meter.complete('real_blocks', 1)
    meter.check()


def production(meter: Meter) -> None:
    from cipher_lab.ciphers import simon32

    check(meter, simon32.encrypt_scalar, simon32.encrypt_batch)

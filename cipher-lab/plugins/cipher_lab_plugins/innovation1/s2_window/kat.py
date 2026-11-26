"""Named, independently referenced public gate (not a new official KAT)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cipher_lab.execution import Meter

from ..prefix_bridge.data import Encrypt
from ..prefix_bridge.kat import Key
from . import config
from .data import Inverse, _full_forward, _inverse_points

_PUBLIC_KEY: Key = (0x06070809, 0x02030405, 0x0001)
_PUBLIC_POINT = 0x0123456789ABCDEF


def check(meter: Meter, scalar: Callable[[int, Key, int], int], inverse: Inverse,
          prefixes: Encrypt) -> None:
    expected: list[int] = []
    for rounds in config.ROUNDS:
        meter.reserve('real_blocks', 1)
        value = scalar(_PUBLIC_POINT, _PUBLIC_KEY, rounds)
        if type(value) is not int or not 0 <= value < 1 << 64:
            raise ValueError('PUBLIC_SCALAR_RESPONSE')
        meter.complete('real_blocks', 1)
        expected.append(value)
    s2 = np.array([expected[0]], dtype=np.uint64)
    plain = _inverse_points(meter, s2, _PUBLIC_KEY, inverse, expected_plain=_PUBLIC_POINT)
    _full_forward(meter, plain, s2, _PUBLIC_KEY, prefixes, expected_c7=expected[1])
    meter.check()


def production(meter: Meter) -> None:
    from cipher_lab.ciphers import present, present_prefix, present_s2

    def scalar(point: int, key: Key, rounds: int) -> int:
        value, = present.encrypt_many((point,), key, rounds)
        return value

    check(meter, scalar, present_s2.invert_second_output, present_prefix.encrypt_prefixes)

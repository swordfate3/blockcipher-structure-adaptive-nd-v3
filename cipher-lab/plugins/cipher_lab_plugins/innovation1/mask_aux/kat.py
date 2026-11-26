"""The public 31-round KAT and reduced r7 reference only."""
from __future__ import annotations

from collections.abc import Callable

from cipher_lab.execution import Meter

from ..prefix_bridge import kat as bridge
from .config import ROUNDS


def check(meter: Meter, scalar: Callable[[int, bridge.Key, int], int],
          prefixes: Callable[[bridge.Words, bridge.Key, tuple[int, ...]], bridge.Words]) -> None:
    bridge.check(meter, scalar, prefixes, rounds=ROUNDS)


def production(meter: Meter) -> None:
    bridge.production(meter, rounds=ROUNDS)

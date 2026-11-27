"""Select just the pure-card branch of the shared single-fit chronology."""
from __future__ import annotations

from collections.abc import Callable

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter

from ..s2_window import training as parent
from . import config


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        gate: Callable[[Meter], None],
        make: Callable[[str, dict[str, object], bool], parent._CardRuntime],
        make_panels: Callable[[Meter], parent._Panels],
        peaks: Callable[[], dict[str, int]]) -> None:
    parent.run(meter, store, summary, gate=gate, make=make, make_panels=make_panels, peaks=peaks,
               _card_recipe=config)

"""Bind the fixed pair recipe to the existing serial A/B flow."""
from __future__ import annotations

from collections.abc import Callable

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter

from ..mask_aux import training as group
from . import config
from .data import Panels


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        gate: Callable[[Meter], None], make: Callable[[str, dict[str, object], bool], group.Engine],
        make_panels: Callable[[Meter], Panels], peaks: Callable[[], dict[str, int]]) -> None:
    group.run(meter, store, summary, gate=gate, make=make, make_panels=make_panels, peaks=peaks,
              _pair_recipe=config)

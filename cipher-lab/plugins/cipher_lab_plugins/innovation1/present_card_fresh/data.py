"""C7-only compact pair provider with the fresh card's split and shape domain."""
from __future__ import annotations

from cipher_lab.execution import Meter

from ..mask_aux_single import data as parent
from ..mask_aux_single.data import BatchPanel, Panels, Sample, _sample
from ..prefix_bridge.data import Draw, Encrypt, Split
from . import config


def make_panels(meter: Meter, draw: Draw, encrypt: Encrypt, *, sample: Sample = _sample) -> Panels:
    return parent.make_panels(meter, draw, encrypt, sample=sample, _card_recipe=config)


def production_panels(meter: Meter) -> FitPanels:
    return FitPanels(parent.production_panels(meter, _card_recipe=config))


class FitPanels:
    def __init__(self, panels: Panels) -> None:
        self._panels = panels

    @property
    def hashes(self) -> dict[str, str]:
        return self._panels.hashes

    def view(self, split: Split) -> BatchPanel:
        return self._panels.view(split, config.SLOT)

    def test(self) -> None:
        self._panels.test()

"""Bind the pure card, matching augmentation, and slot-specific checkpoint identity."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from cipher_lab.execution import Meter

from ..checkpoint import canonical
from ..config import Task
from ..prefix_bridge.data import Floats, Labels
from ..present_card_fresh import config as pure
from ..present_card_fresh import weights
from ..present_card_swap import runtime as swap
from ..present_card_swap import views
from . import config


def _identity(slot: str, source: Mapping[str, str], run_id: str, epoch: int, bce: float,
              initial: str, geometry: Mapping[str, str]) -> dict[str, Any]:
    if slot not in config.SLOTS:
        raise ValueError('INVALID_SLOT')
    identity = weights.identity(pure.SLOT, source, run_id, epoch, bce, initial, geometry)
    identity.update(method=config.METHOD, slot=slot, recipe=config.recipe())
    return identity


def _require_identity(slot: str, metadata: dict[str, Any]) -> None:
    try:
        expected = _identity(slot, metadata['source'], metadata['run_id'], metadata['epoch'],
                             metadata['bce'], metadata['initial_sha256'], metadata['geometry'])
        if canonical(expected) != canonical(metadata):
            raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        raise ValueError('CHECKPOINT_IDENTITY') from None


class Runtime(swap.Runtime):
    _require_identity = staticmethod(_require_identity)

    def _input_domain(self, slot: str) -> tuple[Task, None]:
        if slot not in config.SLOTS:
            raise ValueError('INVALID_SLOT')
        return Task('PRESENT', pure.ROUNDS[0], pure.DIFFERENCE), None

    def batch_update(self, inputs: Floats, labels: Labels, meter: Meter) -> float:
        if len(inputs) != pure.BATCH_SIZE:
            raise ValueError('TRAIN_BATCH_SIZE')
        self._validate(inputs)
        inputs = views.swap_half(inputs, seed=config.SWAP_SEED, step=self._local_step(meter))
        # The inherited swap applies only to the legacy 'swap-50' slot.
        return super().batch_update(inputs, labels, meter)


class FitRuntime(swap.FitRuntime):
    def __init__(self, slot: str, arm: dict[str, object], train: bool, *,
                 source: dict[str, str], run_id: str, origin: int) -> None:
        self._runtime = Runtime(slot, origin=origin)
        self._slot, self._arm, self._source, self._run_id = slot, arm, dict(source), run_id
        if train:
            self._runtime.start()

    @property
    def classifier_initial_sha(self) -> str:
        return self.initial_sha

    def _metadata(self, point: object) -> dict[str, Any]:
        if not isinstance(point, dict) or type(point.get('epoch')) is not int:
            raise ValueError('MISSING_DEV_SELECTION')
        if type(point.get('step')) is not int or point['step'] != pure._steps(point['epoch']):
            raise ValueError('INVALID_DEV_STEP')
        return _identity(self._slot, self._source, self._run_id, point['epoch'], point['bce'],
                         cast(str, self._arm['initial_sha256']),
                         cast('dict[str, str]', self._arm['geometry']))

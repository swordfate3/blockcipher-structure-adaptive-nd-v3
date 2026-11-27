"""Pure-card fitting with arm-local scheduling and endpoint augmentation."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from cipher_lab.execution import Meter

from ..checkpoint import canonical
from ..config import Task
from ..prefix_bridge.data import Floats, Labels
from ..present_card_fresh import config as pure_config
from ..present_card_fresh import runtime as pure_runtime
from ..present_card_fresh import weights as pure_weights
from . import views

_SLOTS = ('original', 'swap-50')
_SWAP_SEED = 20261128
_COMPLETED = 'completed_optimizer_updates'
_ATTEMPTED = 'attempted_optimizer_updates'


def _identity(slot: str, source: Mapping[str, str], run_id: str, epoch: int, bce: float,
              initial: str, geometry: Mapping[str, str]) -> dict[str, Any]:
    from . import config

    if slot not in _SLOTS:
        raise ValueError('INVALID_SLOT')
    identity = pure_weights.identity(pure_config.SLOT, source, run_id, epoch, bce,
                                     initial, geometry)
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


class Runtime(pure_runtime.Runtime):
    _require_identity = staticmethod(_require_identity)

    def __init__(self, slot: str, *, origin: int) -> None:
        super().__init__(slot)
        if type(origin) is not int or origin < 0:
            raise ValueError('UPDATE_SEQUENCE')
        self._origin = origin

    def _input_domain(self, slot: str) -> tuple[Task, None]:
        if slot not in _SLOTS:
            raise ValueError('INVALID_SLOT')
        return super()._input_domain(pure_config.SLOT)

    def _local_step(self, meter: Meter) -> int:
        ledger = meter.snapshot()
        completed, attempted = ledger[_COMPLETED], ledger[_ATTEMPTED]
        if type(completed) is not int or type(attempted) is not int:
            raise ValueError('UPDATE_SEQUENCE')
        step = completed - self._origin
        if step < 0 or attempted - self._origin != step + 1:
            raise ValueError('UPDATE_SEQUENCE')
        return step

    def _learning_rate(self, meter: Meter) -> float:
        return pure_config.learning_rate(self._local_step(meter))

    def batch_update(self, inputs: Floats, labels: Labels, meter: Meter) -> float:
        if self.slot == 'swap-50':
            if len(inputs) != pure_config.BATCH_SIZE:
                raise ValueError('TRAIN_BATCH_SIZE')
            self._validate(inputs)
            step = self._local_step(meter)
            inputs = views.swap_half(inputs, seed=_SWAP_SEED, step=step)
        return super().batch_update(inputs, labels, meter)


class FitRuntime(pure_runtime.FitRuntime):
    """Bind a slot and its cumulative update count before the first reservation."""

    def __init__(self, slot: str, arm: dict[str, object], train: bool, *,
                 source: dict[str, str], run_id: str, origin: int) -> None:
        self._runtime = Runtime(slot, origin=origin)
        self._slot, self._arm, self._source, self._run_id = slot, arm, dict(source), run_id
        if train:
            self._runtime.start()

    def _metadata(self, point: object) -> dict[str, Any]:
        if not isinstance(point, dict) or type(point.get('epoch')) is not int:
            raise ValueError('MISSING_DEV_SELECTION')
        step = point.get('step')
        if type(step) is not int or step != pure_config._steps(point['epoch']):
            raise ValueError('INVALID_DEV_STEP')
        return _identity(self._slot, self._source, self._run_id, point['epoch'], point['bce'],
                         cast(str, self._arm['initial_sha256']),
                         cast('dict[str, str]', self._arm['geometry']))

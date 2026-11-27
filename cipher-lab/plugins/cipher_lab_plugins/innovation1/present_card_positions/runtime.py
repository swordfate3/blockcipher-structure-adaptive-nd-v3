"""Position-arm identity and arm-relative LR and masking step."""
from __future__ import annotations

from typing import Any, cast

from cipher_lab.execution import Meter

from ..checkpoint import canonical
from ..config import Task
from ..prefix_bridge import runtime as bridge
from ..prefix_bridge.data import Floats, Labels
from ..present_card_fresh import config as pure
from ..present_card_mask import config as prior_config
from ..present_card_mask import runtime as prior
from ..present_card_mask import weights as prior_weights
from . import config, masking


def _identity(slot: str, data: dict[str, Any]) -> dict[str, Any]:
    if slot not in config.SLOTS:
        raise ValueError('INVALID_SLOT')
    base = prior_weights.identity(prior_config.SLOT, data['source'], data['run_id'],
                                  data['epoch'], data['bce'], data['initial_sha256'],
                                  data['geometry'])
    return {**base, 'method': config.METHOD, 'slot': slot, 'recipe': config.recipe()}


def _require_identity(slot: str, metadata: dict[str, Any]) -> None:
    try:
        if canonical(_identity(slot, metadata)) != canonical(metadata):
            raise ValueError
    except (AttributeError, TypeError, KeyError, ValueError, OverflowError):
        raise ValueError('CHECKPOINT_IDENTITY') from None


class Runtime(prior.Runtime):
    _require_identity = staticmethod(_require_identity)
    origin = 0

    def _input_domain(self, slot: str) -> tuple[Task, None]:
        if slot not in config.SLOTS:
            raise ValueError('INVALID_SLOT')
        return Task('PRESENT', pure.ROUNDS[0], pure.DIFFERENCE), None

    def _learning_rate(self, meter: Meter) -> float:
        return pure.learning_rate(self._step(meter) - self.origin)

    def batch_update(self, inputs: Floats, labels: Labels, meter: Meter) -> float:
        if len(inputs) != pure.BATCH_SIZE or self._state != 'TRAINING':
            raise ValueError('INVALID_TRAIN_BATCH')
        self._learning_rate(meter)
        self._masked = masking.prepare(inputs, self._step(meter) - self.origin + 1, self.slot)
        try:
            bce = bridge.Runtime.batch_update(self, inputs, labels, meter)
            if self._batch_aux is None:
                raise ValueError('MISSING_AUX_BCE')
            self.aux_sum += self._batch_aux * len(inputs)
            self.aux_rows += len(inputs)
            return bce
        finally:
            self._masked = None
            self._batch_aux = None


class FitRuntime(prior.FitRuntime):
    _runtime_type = Runtime

    def __init__(self, slot: str, arm: dict[str, object], train: bool, *,
                 source: dict[str, str], run_id: str, origin: int) -> None:
        super().__init__(slot, arm, train, source=source, run_id=run_id)
        cast(Runtime, self._runtime).origin = origin

    def _metadata(self, point: object) -> dict[str, Any]:
        if not isinstance(point, dict) or type(point.get('epoch')) is not int:
            raise ValueError('MISSING_DEV_SELECTION')
        if type(point.get('step')) is not int or point['step'] != pure._steps(point['epoch']):
            raise ValueError('INVALID_DEV_STEP')
        return _identity(self._slot, dict(source=self._source, run_id=self._run_id,
                         epoch=point['epoch'], bce=point['bce'],
                         initial_sha256=self._arm['initial_sha256'],
                         geometry=self._arm['geometry']))

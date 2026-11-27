"""Arm-local swap augmentation and bound checkpoints for the two card models."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from cipher_lab.execution import Meter

from ..checkpoint import canonical
from ..config import Task
from ..prefix_bridge import runtime as bridge
from ..prefix_bridge.data import Floats, Labels
from ..prefix_bridge.model import initial_sha256
from ..present_card_fresh import config as pure_config
from ..present_card_fresh import runtime as pure_runtime
from ..present_card_fresh import weights as pure_weights
from ..present_card_mask import config as masked_config
from ..present_card_mask import runtime as masked_runtime
from ..present_card_mask.model import MaskedCard
from ..present_card_swap import views

_PURE = 'swap-50'
_MASKED = 'swap-50-mask'
_SWAP_SEED = 20261128
_METHOD = 'innovation1-present-card-swap-mask-v1'


def _identity(slot: str, source: Mapping[str, str], run_id: str, epoch: int, bce: float,
              initial: str, geometry: Mapping[str, str]) -> dict[str, Any]:
    from . import config

    if slot not in (_PURE, _MASKED):
        raise ValueError('INVALID_SLOT')
    identity = pure_weights.identity(pure_config.SLOT, source, run_id, epoch, bce,
                                     initial, geometry)
    identity.update(method=_METHOD, slot=slot, recipe=config.recipe())
    return identity


def _require_identity(slot: str, metadata: dict[str, Any]) -> None:
    try:
        expected = _identity(slot, metadata['source'], metadata['run_id'], metadata['epoch'],
                             metadata['bce'], metadata['initial_sha256'], metadata['geometry'])
        if canonical(expected) != canonical(metadata):
            raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        raise ValueError('CHECKPOINT_IDENTITY') from None


class _Swapped(bridge.Runtime):
    _require_identity = staticmethod(_require_identity)

    def __init__(self, slot: str, *, origin: int) -> None:
        if type(origin) is not int or origin < 0:
            raise ValueError('UPDATE_SEQUENCE')
        super().__init__(slot)
        self._origin = origin

    def _input_domain(self, slot: str) -> tuple[Task, None]:
        return Task('PRESENT', pure_config.ROUNDS[0], pure_config.DIFFERENCE), None

    def _local_step(self, meter: Meter) -> int:
        ledger = meter.snapshot()
        completed = ledger['completed_optimizer_updates']
        attempted = ledger['attempted_optimizer_updates']
        if type(completed) is not int or type(attempted) is not int:
            raise ValueError('UPDATE_SEQUENCE')
        step = completed - self._origin
        if step < 0 or attempted - self._origin != step + 1:
            raise ValueError('UPDATE_SEQUENCE')
        return step

    def _learning_rate(self, meter: Meter) -> float:
        return pure_config.learning_rate(self._local_step(meter))

    def batch_update(self, inputs: Floats, labels: Labels, meter: Meter) -> float:
        if len(inputs) != pure_config.BATCH_SIZE:
            raise ValueError('TRAIN_BATCH_SIZE')
        self._validate(inputs)
        step = self._local_step(meter)
        swapped = views.swap_half(inputs, seed=_SWAP_SEED, step=step)
        return super().batch_update(swapped, labels, meter)


class _PureRuntime(_Swapped, pure_runtime.Runtime):
    @property
    def classifier_initial_sha(self) -> str:
        return self._initial


class _MaskedRuntime(_Swapped, masked_runtime.Runtime):
    classifier_initial_sha = ''

    def _step(self, meter: Meter) -> int:  # type: ignore[override]
        return self._local_step(meter)

    def _fresh_cpu(self) -> MaskedCard:
        net = super()._fresh_cpu()
        self.classifier_initial_sha = initial_sha256(net.classifier)
        return net


class FitRuntime(pure_runtime.FitRuntime):
    """Keep the physical models and complete initialization identities arm-specific."""

    def __init__(self, slot: str, arm: dict[str, object], train: bool, *,
                 source: dict[str, str], run_id: str, origin: int) -> None:
        if slot == _PURE:
            runtime: _PureRuntime | _MaskedRuntime = _PureRuntime(slot, origin=origin)
        elif slot == _MASKED:
            runtime = _MaskedRuntime(slot, origin=origin)
        else:
            raise ValueError('INVALID_SLOT')
        self._runtime = runtime
        self._slot, self._arm, self._source, self._run_id = slot, arm, dict(source), run_id
        if train:
            runtime.start()

    @property
    def classifier_initial_sha(self) -> str:
        return cast('_PureRuntime | _MaskedRuntime', self._runtime).classifier_initial_sha

    @property
    def geometry(self) -> dict[str, str]:
        return dict(cast('_PureRuntime | _MaskedRuntime', self._runtime).geometry)

    def _metadata(self, point: object) -> dict[str, Any]:
        if not isinstance(point, dict) or type(point.get('epoch')) is not int:
            raise ValueError('MISSING_DEV_SELECTION')
        step = point.get('step')
        if type(step) is not int or step != pure_config._steps(point['epoch']):
            raise ValueError('INVALID_DEV_STEP')
        return _identity(self._slot, self._source, self._run_id, point['epoch'], point['bce'],
                         cast(str, self._arm['initial_sha256']),
                         cast('dict[str, str]', self._arm['geometry']))

    def dump(self) -> bytes:
        if (self.initial_sha != self._arm.get('initial_sha256')
                or self.parameters != self._arm.get('parameters')
                or self.geometry != self._arm.get('geometry')):
            raise ValueError('MODEL_INITIAL_IDENTITY')
        return super().dump()

    def batch_update(self, inputs: Floats, labels: Labels, meter: Meter) -> float:
        loss = super().batch_update(inputs, labels, meter)
        runtime = self._runtime
        if isinstance(runtime, _MaskedRuntime):
            self._arm['auxiliary_train'] = {
                'rows': runtime.aux_rows,
                'targets': runtime.aux_rows * masked_config.MASK_SITES,
                'bce': runtime.aux_sum / runtime.aux_rows,
            }
        return loss

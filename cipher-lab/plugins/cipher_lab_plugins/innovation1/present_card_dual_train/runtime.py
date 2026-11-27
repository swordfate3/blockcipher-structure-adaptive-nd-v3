"""Pure-card fitting and metered dual-order evaluation."""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from cipher_lab.execution import Meter

from ...direct_block.fit import Engine, predict_batch
from ..checkpoint import canonical
from ..config import Task
from ..prefix_bridge.data import Floats, Labels
from ..present_card_fresh import config as pure_config
from ..present_card_fresh import runtime as pure_runtime
from ..present_card_fresh import weights as pure_weights
from ..present_card_swap import runtime as swap_runtime
from ..present_card_swap.views import swap_all
from . import config
from .probability import average_logits, average_logits_torch


def _identity(slot: str, source: dict[str, str], run_id: str, epoch: int, bce: float,
              initial: str, geometry: dict[str, str]) -> dict[str, Any]:
    if slot not in config.SLOTS:
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


class Runtime(swap_runtime.Runtime):
    _require_identity = staticmethod(_require_identity)

    def _input_domain(self, slot: str) -> tuple[Task, None]:
        if slot not in config.SLOTS:
            raise ValueError('INVALID_SLOT')
        return pure_runtime.Runtime._input_domain(self, pure_config.SLOT)

    def _training_loss(self, slot: str, x: Tensor, logits: Tensor, y: Tensor,
                       resources: Meter) -> tuple[Tensor, float]:
        if slot == 'dual-train':
            rows = int(y.shape[0])
            resources.reserve('forward_rows', rows)
            swapped = self._forward(slot, self._task, torch.flip(x, [2]))
            if (swapped.dtype != logits.dtype or swapped.shape != y.shape
                    or not bool(torch.isfinite(swapped).all())):
                raise ValueError('FORWARD_INVALID')
            resources.complete('forward_rows', rows)
            logits = average_logits_torch(logits, swapped)
        return super()._training_loss(slot, x, logits, y, resources)


class FitRuntime(pure_runtime.FitRuntime):
    """Bind a slot, the local update origin and the meter for legacy DEV fit."""

    def __init__(self, slot: str, arm: dict[str, object], train: bool, *,
                 source: dict[str, str], run_id: str, origin: int, meter: Meter) -> None:
        self._runtime = Runtime(slot, origin=origin)
        self._slot, self._arm, self._source, self._run_id = slot, arm, dict(source), run_id
        self._dual_dev, self._meter = train, meter
        if train:
            self._runtime.start()

    @property
    def classifier_initial_sha(self) -> str:
        return self.initial_sha

    def _metadata(self, point: object) -> dict[str, Any]:
        if not isinstance(point, dict) or type(point.get('epoch')) is not int:
            raise ValueError('MISSING_DEV_SELECTION')
        step = point.get('step')
        if type(step) is not int or step != pure_config._steps(point['epoch']):
            raise ValueError('INVALID_DEV_STEP')
        return _identity(self._slot, self._source, self._run_id, point['epoch'], point['bce'],
                         cast(str, self._arm['initial_sha256']),
                         cast('dict[str, str]', self._arm['geometry']))

    def predict(self, inputs: Floats) -> Floats:
        with torch.inference_mode():
            first = super().predict(inputs)
            if not self._dual_dev:
                return first
            expected = (len(inputs), 1)
            if (first.dtype != np.float32 or first.shape != expected
                    or not np.isfinite(first).all()):
                raise ValueError('invalid logits')
            # The legacy outer predict_batch owns the first reservation.
            rows = len(inputs)
            self._meter.reserve('forward_rows', rows)
            second = super().predict(swap_all(inputs))
            if (second.dtype != np.float32 or second.shape != expected
                    or not np.isfinite(second).all()):
                raise ValueError('invalid logits')
            self._meter.complete('forward_rows', rows)
            return average_logits(first, second)


def predict_orders(engine: Engine, inputs: Floats, labels: Labels,
                   meter: Meter) -> tuple[Floats, Floats]:
    """Return original and dual logits from a single-order scorer in two passes."""
    if isinstance(engine, FitRuntime) and engine._dual_dev:
        raise ValueError('REQUIRES_SINGLE_ORDER_SCORER')
    first = predict_batch(engine, inputs, labels, meter)
    second = predict_batch(engine, swap_all(inputs), labels, meter)
    return first, average_logits(first, second)

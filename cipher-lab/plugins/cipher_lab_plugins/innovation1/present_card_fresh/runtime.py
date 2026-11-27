"""Card construction and LR selection on the bridge's one-slot lifecycle."""
from __future__ import annotations

import math
from typing import Any, cast

from torch import Tensor, nn

from cipher_lab.execution import Meter

from ..config import Optimizer, Recipe, Task
from ..prefix_bridge import runtime as parent
from ..prefix_bridge.data import Floats, Labels
from ..prefix_bridge.model import initial_sha256
from . import config, model, weights

_OPT_CONFIG = Recipe(optimizer=Optimizer(lr_max=config.LR_MAX, lr_min=config.LR_MIN,
                                       betas=config.BETAS, eps=config.EPS,
                                       weight_decay=config.WEIGHT_DECAY, clip_norm=config.CLIP))


class Runtime(parent.Runtime):
    _input_shape = (1, 2, 64)
    _require_identity = staticmethod(weights.require_identity)

    def __init__(self, slot: str) -> None:
        super().__init__(slot)
        self.recipe = _OPT_CONFIG   # Parent Adam reads this; no optimizer is made by __init__.
        self.geometry: dict[str, str] = {}

    def _input_domain(self, slot: str) -> tuple[Task, None]:
        if slot != config.SLOT:
            raise ValueError('INVALID_SLOT')
        return Task('PRESENT', config.ROUNDS[0], config.DIFFERENCE), None

    def _fresh_cpu(self) -> model.Classifier:
        net = model.build()
        self._initial = initial_sha256(net)
        self._parameters = sum(p.numel() for p in net.parameters())
        self.geometry = model.geometry(net)
        return net

    def _fixed_state(self, model: nn.Module) -> tuple[str, ...]:
        # Read back every fixed public buffer and frozen router tensor exactly.
        return (*super()._fixed_state(model),
                *(name for name, p in model.named_parameters() if not p.requires_grad))

    def _learning_rate(self, meter: Meter) -> float:
        ledger = meter.snapshot()
        step = cast(int, ledger['completed_optimizer_updates'])
        # Shared fit reserves exactly one update before calling this method.
        if ledger['attempted_optimizer_updates'] != step + 1:
            raise ValueError('UPDATE_SEQUENCE')
        return config.learning_rate(step)

    def batch_update(self, inputs: Floats, labels: Labels, meter: Meter) -> float:
        if len(inputs) != config.BATCH_SIZE:
            raise ValueError('TRAIN_BATCH_SIZE')
        return super().batch_update(inputs, labels, meter)

    def _training_loss(self, slot: str, x: Any, logits: Tensor, y: Tensor,
                       resources: Any) -> tuple[Tensor, float]:
        loss = self._loss(logits, y)
        # Finish the fallible host copy before the step; the inherited post-sync
        # zero_grad may still fail, in which case fit does not record completion.
        reported = float(loss.detach().cpu())
        if not math.isfinite(reported) or reported < 0:
            raise ValueError('INVALID_CLASS_BCE')
        return loss, reported


class FitRuntime(parent.FitRuntime):
    _runtime_type = Runtime

    def _metadata(self, point: object) -> dict[str, Any]:
        if not isinstance(point, dict) or type(point.get('epoch')) is not int:
            raise ValueError('MISSING_DEV_SELECTION')
        if type(point.get('step')) is not int or point['step'] != config._steps(point['epoch']):
            raise ValueError('INVALID_DEV_STEP')
        return weights.identity(self._slot, self._source, self._run_id, point['epoch'],
                                point['bce'], cast(str, self._arm['initial_sha256']),
                                cast('dict[str, str]', self._arm['geometry']))

    @property
    def geometry(self) -> dict[str, str]:
        return dict(cast(Runtime, self._runtime).geometry)

    def restore(self, raw: bytes) -> None:
        super().restore(raw)
        if self.geometry != self._arm['geometry']:
            raise ValueError('MODEL_GEOMETRY_IDENTITY')

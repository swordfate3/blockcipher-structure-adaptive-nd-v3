"""One synchronized update with separate classification and reconstruction losses."""
from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import Tensor, nn

from cipher_lab.execution import Meter

from ..config import Task
from ..prefix_bridge import runtime as parent
from ..prefix_bridge.data import Floats, Indices, Labels
from ..prefix_bridge.model import initial_sha256
from ..present_card_fresh import config as pure_config
from ..present_card_fresh.model import geometry
from ..present_card_fresh.runtime import _OPT_CONFIG
from . import config, masking, model, weights


class Runtime(parent.Runtime):
    _input_shape = (1, 2, 64)
    _require_identity = staticmethod(weights.require_identity)

    def __init__(self, slot: str) -> None:
        super().__init__(slot)
        self.recipe = _OPT_CONFIG
        self.geometry: dict[str, str] = {}
        self._masked: tuple[Floats, Indices, Floats] | None = None
        self._batch_aux: float | None = None
        self.aux_rows = 0
        self.aux_sum = 0.

    def _input_domain(self, slot: str) -> tuple[Task, None]:
        if slot != config.SLOT:
            raise ValueError('INVALID_SLOT')
        return Task('PRESENT', config.ROUNDS[0], config.DIFFERENCE), None

    def _fresh_cpu(self) -> model.MaskedCard:
        net = model.build()
        self._initial = initial_sha256(net)
        self._parameters = sum(p.numel() for p in net.parameters())
        self.geometry = geometry(net.classifier)
        return net

    def _fixed_state(self, net: nn.Module) -> tuple[str, ...]:
        return (*super()._fixed_state(net),
                *(name for name, value in net.named_parameters() if not value.requires_grad))

    @staticmethod
    def _step(meter: Meter) -> int:
        ledger = meter.snapshot()
        step = cast(int, ledger['completed_optimizer_updates'])
        if ledger['attempted_optimizer_updates'] != step + 1:
            raise ValueError('UPDATE_SEQUENCE')
        return step

    def _learning_rate(self, meter: Meter) -> float:
        return pure_config.learning_rate(self._step(meter))

    def batch_update(self, inputs: Floats, labels: Labels, meter: Meter) -> float:
        if len(inputs) != config.BATCH_SIZE or self._state != 'TRAINING':
            raise ValueError('INVALID_TRAIN_BATCH')
        self._learning_rate(meter)
        self._masked = masking.prepare(inputs, self._step(meter) + 1)
        try:
            bce = super().batch_update(inputs, labels, meter)
            if self._batch_aux is None:
                raise ValueError('MISSING_AUX_BCE')
            self.aux_sum += self._batch_aux * len(inputs)
            self.aux_rows += len(inputs)
            return bce
        finally:
            self._masked = None
            self._batch_aux = None

    def _training_loss(self, slot: str, x: Any, logits: Tensor, y: Tensor,
                       resources: Any) -> tuple[Tensor, float]:
        net = self.models[slot]
        if self._masked is None or not isinstance(net, model.MaskedCard):
            raise ValueError('INVALID_AUX_STATE')
        masked, indices, targets = self._masked
        rows = len(masked)
        if rows != len(logits) or targets.shape != (rows * config.MASK_SITES,):
            raise ValueError('INVALID_AUX_TARGETS')
        classification = self._loss(logits, y)
        on_device = self._on_device(masked)
        resources.reserve('forward_rows', rows)
        predictions = net.masked_logits(on_device)
        if (predictions.shape != (rows, 1, 2, 64) or predictions.dtype != torch.float32
                or not bool(torch.isfinite(predictions).all())):
            raise ValueError('INVALID_AUX_LOGITS')
        resources.complete('forward_rows', rows)
        locations = torch.from_numpy(indices).to(device=self.device, dtype=torch.int64)
        reconstruction = self._loss(predictions.flatten().index_select(0, locations),
                                    self._on_device(targets))
        cls_bce = float(classification.detach().cpu())
        aux_bce = float(reconstruction.detach().cpu())
        if not all(math.isfinite(value) and value >= 0 for value in (cls_bce, aux_bce)):
            raise ValueError('INVALID_BATCH_BCE')
        self._batch_aux = aux_bce
        return classification + config.AUX_WEIGHT * reconstruction, cls_bce


class FitRuntime(parent.FitRuntime):
    _runtime_type = Runtime

    def _metadata(self, point: object) -> dict[str, Any]:
        if not isinstance(point, dict) or type(point.get('epoch')) is not int:
            raise ValueError('MISSING_DEV_SELECTION')
        step = pure_config._steps(point['epoch'])
        if type(point.get('step')) is not int or point['step'] != step:
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

    def batch_update(self, inputs: Floats, labels: Labels, meter: Meter) -> float:
        loss = super().batch_update(inputs, labels, meter)
        runtime = cast(Runtime, self._runtime)
        self._arm['auxiliary_train'] = {
            'rows': runtime.aux_rows, 'targets': runtime.aux_rows * config.MASK_SITES,
            'bce': runtime.aux_sum / runtime.aux_rows,
        }
        return loss

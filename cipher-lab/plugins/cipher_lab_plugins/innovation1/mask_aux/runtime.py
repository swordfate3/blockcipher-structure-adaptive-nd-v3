"""B adds a masked loss to the parent's one synchronized update."""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from cipher_lab.execution import Meter

from ..prefix_bridge import runtime as bridge
from ..prefix_bridge.data import Floats, Indices, Labels
from ..prefix_bridge.model import initial_sha256
from . import config, model, weights
from .masking import mask_for_step, masked_view


class Runtime(bridge.Runtime):
    _slot_input = staticmethod(config.slot_input)
    _baseline = config.BASELINE
    _auxiliary = config.AUXILIARY
    _targets_per_row = config.TARGETS_PER_ROW
    _aux_input_shape = (4, 2, 64)
    _require_identity = staticmethod(weights.require_identity)

    def __init__(self, slot: str) -> None:
        super().__init__(slot)
        self._step = 0
        self._masked: tuple[Floats, Indices, Floats] | None = None
        self._aux: float | None = None
        self._aux_sum = 0.
        self._aux_rows = 0

    def _fresh_cpu(self) -> nn.Module:
        if self.slot == self._baseline:
            return super()._fresh_cpu()
        net = model.build()
        self._initial = initial_sha256(net.classifier)
        self._parameters = sum(t.numel() for t in net.parameters())
        return net

    def batch_update(self, inputs: Floats, labels: Labels, meter: Meter) -> float:
        if self.slot == self._baseline:
            return super().batch_update(inputs, labels, meter)
        self._step += 1
        self._masked = self._prepare_mask(inputs, self._step)
        try:
            class_bce = super().batch_update(inputs, labels, meter)
            # Both scalars were validated before backward. Only credit this
            # auxiliary batch once the parent has returned from its update.
            if self._aux is not None:
                self._aux_sum += self._aux * len(inputs)
                self._aux_rows += len(inputs)
            return class_bce
        finally:
            self._masked = None
            self._aux = None

    def _prepare_mask(self, inputs: Floats, step: int) -> tuple[Floats, Indices, Floats]:
        return masked_view(inputs, mask_for_step(len(inputs), step))

    def _masked_forward(self, net: nn.Module) -> Callable[[Tensor], Tensor]:
        if not isinstance(net, model.MaskAux):
            raise ValueError('INVALID_AUX_STATE')
        return net.masked_logits

    def take_aux_epoch(self) -> tuple[float, int]:
        if self.slot != self._auxiliary or not self._aux_rows:
            raise ValueError('MISSING_AUX_EPOCH')
        mean, rows = self._aux_sum / self._aux_rows, self._aux_rows
        self._aux_sum, self._aux_rows = 0., 0
        return mean, rows

    def _training_loss(self, slot: str, x: Any, logits: Tensor, y: Tensor,
                       resources: Any) -> tuple[Tensor, Tensor | float]:
        if slot == self._baseline:
            return super()._training_loss(slot, x, logits, y, resources)
        net = self.models[slot]
        if self._masked is None:
            raise ValueError('INVALID_AUX_STATE')
        masked_forward = self._masked_forward(net)
        masked, indices, targets = self._masked
        if len(masked) != logits.shape[0] or targets.shape != (len(masked)*self._targets_per_row,):
            raise ValueError('INVALID_AUX_TARGETS')
        class_loss = self._loss(logits, y)
        masked_tensor = self._on_device(masked)
        resources.reserve('forward_rows', len(masked))
        aux_logits = masked_forward(masked_tensor)
        if (aux_logits.shape != (len(masked), *self._aux_input_shape)
                or aux_logits.dtype != torch.float32
                or not bool(torch.isfinite(aux_logits).all())):
            raise ValueError('INVALID_AUX_LOGITS')
        resources.complete('forward_rows', len(masked))
        selected = aux_logits.reshape(-1).index_select(
            0, torch.from_numpy(indices).to(device=self.device, dtype=torch.int64))
        aux_loss = self._loss(selected, self._on_device(targets))
        aux_bce = float(aux_loss.detach().cpu())
        if not math.isfinite(aux_bce) or aux_bce < 0:
            raise ValueError('INVALID_AUX_BCE')
        class_bce = float(class_loss.detach().cpu())
        if not math.isfinite(class_bce) or class_bce < 0:
            raise ValueError('INVALID_CLASS_BCE')
        self._aux = aux_bce
        return class_loss + config.AUX_WEIGHT * aux_loss, class_bce


class FitRuntime(bridge.FitRuntime):
    _runtime_type: type[Runtime] = Runtime
    _identity = staticmethod(weights.identity)

    def take_aux_epoch(self) -> tuple[float, int]:
        assert isinstance(self._runtime, Runtime)
        return self._runtime.take_aux_epoch()

    def _metadata(self, point: object) -> dict[str, Any]:
        metadata = super()._metadata(point)
        assert isinstance(point, dict)
        if type(point.get('step')) is not int or point['step'] != metadata['step']:
            raise ValueError('CHECKPOINT_IDENTITY')
        return metadata

"""Pair layout hooks on the existing masked/full-view synchronized update."""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from torch import Tensor, nn

from ..mask_aux import runtime as group
from ..mask_aux.masking import mask_for_step, masked_view
from ..prefix_bridge.data import Floats, Indices
from ..prefix_bridge.model import initial_sha256
from . import config, model, weights


class Runtime(group.Runtime):
    _slot_input = staticmethod(config.slot_input)
    _require_identity = staticmethod(weights.require_identity)
    _baseline = config.BASELINE
    _auxiliary = config.AUXILIARY
    _targets_per_row = config.TARGETS_PER_ROW
    _input_shape = (1, 6, 64)
    _aux_input_shape = (1, 2, 64)

    def _validate(self, x: Floats) -> None:
        super()._validate(x)
        if len(x) % 4:
            raise ValueError('PAIR_LAYOUT')

    def _fresh_cpu(self) -> model.PairClassifier:
        net = model.build(self.slot == self._auxiliary)
        self._initial = initial_sha256(net.classifier)
        self._parameters = sum(t.numel() for t in net.parameters())
        return net

    def _prepare_mask(self, inputs: Floats, step: int) -> tuple[Floats, Indices, Floats]:
        self._validate(inputs)
        n = len(inputs)
        masked, indices, targets = masked_view(inputs.reshape(n // 4, 4, 6, 64),
                                                mask_for_step(n // 4, step))
        # The same flat order indexes [N,1,2,64] and [N/4,4,2,64].
        return masked.reshape(n, *self._input_shape), indices, targets

    def _masked_forward(self, net: nn.Module) -> Callable[[Tensor], Tensor]:
        if not isinstance(net, model.PairAux):
            raise ValueError('INVALID_AUX_STATE')
        return net.masked_logits

    def _training_loss(self, slot: str, x: Any, logits: Tensor, y: Tensor,
                       resources: Any) -> tuple[Tensor, float]:
        loss, reported = super()._training_loss(slot, x, logits, y, resources)
        # B already captured classification BCE; for A, any failed host copy
        # must happen before backward and before a synchronized Adam update.
        class_bce = reported if isinstance(reported, float) else float(reported.detach().cpu())
        if not math.isfinite(class_bce) or class_bce < 0:
            raise ValueError('INVALID_CLASS_BCE')
        return loss, class_bce


class FitRuntime(group.FitRuntime):
    _runtime_type: type[Runtime] = Runtime
    _identity = staticmethod(weights.identity)

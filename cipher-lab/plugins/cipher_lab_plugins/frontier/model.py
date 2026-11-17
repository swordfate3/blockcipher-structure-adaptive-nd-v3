"""Raw64 public-plan adapter and two-hidden-layer bit MLP; no I/O or labels."""
from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from cipher_lab.models.frontier_mlp import FrontierBitMLP as Network
from cipher_lab_plugins.frontier.config import FrontierMLPSpec
from cipher_lab_plugins.frontier.frontier_inputs import frontier_features


class FrontierBitMLP(Network):
    def __init__(self, spec: FrontierMLPSpec) -> None:
        if type(spec) is not FrontierMLPSpec:
            raise ValueError('expected FrontierMLPSpec')
        spec.__post_init__()
        super().__init__(spec.identity.dimension, spec.hidden_width)
        self.spec = spec

    def forward(self, inputs: Tensor) -> Tensor:
        features = frontier_features(inputs, self.spec.plan)
        logits = self.layers(features)
        if logits.shape != (len(inputs), 1) or not bool(torch.isfinite(logits).all()):
            raise ValueError('invalid frontier logits')
        return cast(Tensor, logits)

"""Raw-public-input model with an explicit, fixed 13-coordinate adapter."""
from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from cipher_lab.models.feature_mlp import FeatureBitMLP as Network
from cipher_lab_plugins.feature_probe.config import FeatureMLPSpec
from cipher_lab_plugins.feature_probe.representation import simon_r3_residual13


class FeatureBitMLP(Network):
    def __init__(self, spec: FeatureMLPSpec) -> None:
        if type(spec) is not FeatureMLPSpec:
            raise ValueError('expected FeatureMLPSpec')
        spec.__post_init__()
        super().__init__(13, spec.hidden_width)
        self.spec = spec

    def forward(self, inputs: Tensor) -> Tensor:
        features = simon_r3_residual13(inputs)
        logits = self.layers(features)
        if logits.shape != (len(inputs), 1) or not bool(torch.isfinite(logits).all()):
            raise ValueError('invalid feature logits')
        return cast(Tensor, logits)

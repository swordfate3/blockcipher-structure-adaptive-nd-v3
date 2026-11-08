"""Small binary-output MLP, independent of cipher and dataset modules."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from cipher_lab.training_config import MLPSpec


class BitMLP(nn.Module):
    """Map binary 64-bit observations to one logit per row."""

    def __init__(self, spec: MLPSpec) -> None:
        super().__init__()
        if not isinstance(spec, MLPSpec):
            raise ValueError("spec must be a Task1 MLPSpec")
        self.spec = spec
        hidden_width = spec.hidden_width
        self.layers = nn.Sequential(
            nn.Linear(64, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if (inputs.layout != torch.strided or inputs.ndim != 2
                or inputs.shape[0] < 1 or inputs.shape[1] != 64):
            raise ValueError("inputs must have nonempty shape [N,64]")
        if inputs.dtype != torch.float32:
            raise TypeError("inputs must have dtype torch.float32")
        if inputs.device != next(self.parameters()).device:
            raise ValueError("inputs and model must be on the same device")
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("inputs must be finite")
        if not bool(((inputs == 0) | (inputs == 1)).all()):
            raise ValueError("inputs must contain binary values")
        logits = self.layers(inputs)
        if logits.shape != (inputs.shape[0], 1) or not bool(torch.isfinite(logits).all()):
            raise ValueError("model produced nonfinite or invalid logits")
        return cast(Tensor, logits)

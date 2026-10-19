"""Small binary-output MLP, independent of cipher and dataset modules."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class BitMLP(nn.Module):
    """Map binary 64-bit observations to one logit per row."""

    def __init__(self, hidden_width: int = 64) -> None:
        super().__init__()
        if type(hidden_width) is not int:
            raise TypeError("hidden_width must be a built-in int")
        if not 1 <= hidden_width <= 4096:
            raise ValueError("hidden_width must be between 1 and 4096")
        self.hidden_width = hidden_width
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
        if inputs.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[1] != 64:
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
            raise RuntimeError("model produced invalid logits")
        return cast(Tensor, logits)

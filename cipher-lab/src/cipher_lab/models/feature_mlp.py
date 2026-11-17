"""Raw-public-input model with an explicit, fixed 13-coordinate adapter."""
from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class FeatureBitMLP(nn.Module):
    """Two-hidden-layer pure network; caller owns public input representation."""

    def __init__(self, input_width: int, hidden_width: int) -> None:
        super().__init__()
        if (type(input_width) is not int or not 1 <= input_width <= 8192
                or type(hidden_width) is not int or not 1 <= hidden_width <= 4096):
            raise ValueError('invalid network dimensions')
        width = hidden_width
        self.layers = nn.Sequential(nn.Linear(input_width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, inputs: Tensor) -> Tensor:
        logits = self.layers(inputs)
        if logits.shape != (len(inputs), 1) or not bool(torch.isfinite(logits).all()):
            raise ValueError('invalid feature logits')
        return cast(Tensor, logits)

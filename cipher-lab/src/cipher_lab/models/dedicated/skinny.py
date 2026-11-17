"""SKINNY backstep-XOR ResNet four-pair adaptation, not the single-pair protocol.

Extracted from expansion/skinny.py: only the backstep_xor_resnet variant remains.
Keyless backstep lives in the public representation layer and is applied once.
Default Torch parameter initialization and layer construction order are retained.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from cipher_lab.representations.dedicated import author_pair_layout


class _Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.01),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + cast(torch.Tensor, self.layers(x))


class SkinnyExpert(nn.Module):
    def __init__(self, *, rounds: int = 8) -> None:
        super().__init__()
        if type(rounds) is not int or not 1 <= rounds <= 32:
            raise ValueError("invalid SKINNY rounds")
        self.rounds = rounds
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 1),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.01),
            nn.ReLU(),
            *[_Residual() for _ in range(10)],
            nn.Flatten(),
            nn.Linear(32 * 3 * 16, 64),
            nn.BatchNorm1d(64, eps=0.001, momentum=0.01),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64, eps=0.001, momentum=0.01),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        grid = author_pair_layout(features, rounds=self.rounds)
        hidden = self.encoder(grid).reshape(features.shape[0], 4, 64).mean(dim=1)
        return cast(torch.Tensor, self.head(hidden))

"""Fixed PRESENT XOR/undilated expert, with frozen-statistics trainable-affine BN.

Extracted from kev_selection/components.py; variant controls and forward-time
auxiliary kernel-L2 side effect intentionally omitted for the common BCE recipe.
Default Torch parameter initialization and layer construction order are retained.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from cipher_lab.representations.dedicated import component_input


class SameConv(nn.Conv1d):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        effective = self.dilation[0] * (self.kernel_size[0] - 1)
        return super().forward(F.pad(features, (effective // 2, effective - effective // 2)))


class Residual(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = SameConv(96, 96, width, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(96, eps=0.001, momentum=0.01)
        self.conv2 = SameConv(96, 96, width, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(96, eps=0.001, momentum=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.norm1(self.conv1(features)))
        return features + F.relu(self.norm2(self.conv2(hidden)))


class PresentExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branches = nn.ModuleList([SameConv(12, 32, k) for k in (1, 2, 4)])
        self.initial_norm = nn.BatchNorm1d(96, eps=0.001, momentum=0.01)
        self.blocks = nn.ModuleList(
            [Residual(k, 1) for k in (3, 5, 7, 9, 11)]
        )
        self.classifier = nn.Linear(96, 1)
        self.train()

    def train(self, mode: bool = True) -> PresentExpert:
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
        return self

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        hidden = component_input(bits)
        hidden = F.relu(self.initial_norm(torch.cat([m(hidden) for m in self.branches], 1)))
        for block in self.blocks:
            hidden = block(hidden)
        pooled = hidden.reshape(-1, 4, 96, 16).mean(dim=(1, 3))
        return cast(torch.Tensor, self.classifier(pooled))

"""Four-pair adaptation inspired by arXiv:2201.03767, not an exact reproduction.

Eight public words per pair, convolutions along the pair axis (length FOUR),
multiscale stem, squeeze/excitation and five residual blocks. No secret inputs.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from cipher_lab.representations.dedicated import relation_words


class _SE(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(channels, channels // 16, bias=False),
            nn.ReLU(),
            nn.Linear(channels // 16, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, x * self.gate(x.mean(2)).unsqueeze(2))


def _conv(in_channels: int, out_channels: int, kernel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2),
        nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01),
        nn.ReLU(),
    )


class _Residual(nn.Module):
    def __init__(self, kernel: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(_conv(64, 64, kernel), _conv(64, 64, kernel))
        self.se = _SE(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, x + self.se(self.layers(x)))


class FeistelSEExpert(nn.Module):
    def __init__(self, cipher: str) -> None:
        super().__init__()
        if cipher not in ("simon64", "simeck64"):
            raise ValueError("unsupported cipher")
        self.cipher = cipher
        self.branches = nn.ModuleList([_conv(256, 64, k) for k in (1, 3, 5, 7)])
        self.se = _SE(256)
        self.project = _conv(256, 64, 1)
        self.residual = nn.Sequential(*[_Residual(k) for k in (3, 5, 7, 9, 11)])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 64, 64),
            nn.BatchNorm1d(64, eps=0.001, momentum=0.01),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64, eps=0.001, momentum=0.01),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = relation_words(features, self.cipher)
        x = x.reshape(-1, 4, 256).transpose(1, 2)
        hidden = self.se(torch.cat([branch(x) for branch in self.branches], dim=1))
        hidden = self.residual(self.project(hidden)).transpose(1, 2).contiguous()
        return cast(torch.Tensor, self.head(hidden))


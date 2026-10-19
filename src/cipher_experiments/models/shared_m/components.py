"""Composable stems, shared residual suffix and shared readout."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from cipher_experiments.models.shared_m.config import ModelConfig
from cipher_experiments.representations.neighborhoods import (
    gather_neighborhood,
    gather_relation,
    neighborhood_indices,
)


class GeometryStem(nn.Module):
    indices: torch.Tensor

    def __init__(self, kind: str, config: ModelConfig) -> None:
        super().__init__()
        self.channels = config.channels
        self.register_buffer("indices", neighborhood_indices(kind))
        self.network = nn.Sequential(
            nn.Conv1d(27, config.stem_width, 1),
            nn.GroupNorm(config.norm_groups, config.stem_width),
            nn.ReLU(),
            nn.Conv1d(config.stem_width, config.channels, 1),
            nn.GroupNorm(config.norm_groups, config.channels),
            nn.ReLU(),
        )
        self.relation_projection = nn.Conv1d(9, config.stem_width, 1, bias=False)
        nn.init.zeros_(self.relation_projection.weight)

    def forward(self, raw: torch.Tensor, channel: torch.Tensor) -> torch.Tensor:
        base = self.network[0](gather_neighborhood(raw, self.indices))
        added = self.relation_projection(gather_relation(channel, self.indices))
        # Do not move addition past GroupNorm/ReLU or mix before each stem's norm.
        hidden = self.network[1:](base + added)
        return cast(torch.Tensor, hidden).reshape(-1, 4, self.channels, 64)


class Residual(nn.Module):
    def __init__(self, kernel: int, channels: int, groups: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2),
            nn.GroupNorm(groups, channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2),
            nn.GroupNorm(groups, channels),
            nn.ReLU(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + cast(torch.Tensor, self.network(hidden))


class SharedReadout(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.channels = config.channels
        self.mode = config.head
        self.base = nn.Linear(config.channels * 64, 1)
        self.residual = nn.Sequential(
            nn.Linear(config.channels * 64, config.head_hidden),
            nn.ReLU(),
            nn.Linear(config.head_hidden, 1),
        )
        final = cast(nn.Linear, self.residual[2])
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4 or hidden.shape[1:] != (4, self.channels, 64):
            raise ValueError("exactly four aligned pair features required")
        pooled = hidden.mean(1).flatten(1)
        base = cast(torch.Tensor, self.base(pooled)).flatten()
        if self.mode == "group":
            residual = self.residual(pooled).flatten()
        else:
            residual = self.residual(hidden.flatten(2)).squeeze(-1).mean(1)
        return cast(torch.Tensor, base + residual)

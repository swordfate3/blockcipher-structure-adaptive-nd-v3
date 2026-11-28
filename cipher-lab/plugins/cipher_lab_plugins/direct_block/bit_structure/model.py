"""CPU-seeded models for the public bit-structure comparison."""
from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from .geometry import LEVELS, SHUFFLE, SUPPORT

_WIDTH = 37


def _check_shape(bits: Tensor) -> None:
    if bits.ndim != 2 or bits.shape[1] != 32:
        raise ValueError('expected input of shape [N, 32]')


class _MLP(nn.Sequential):
    def forward(self, bits: Tensor) -> Tensor:
        _check_shape(bits)
        for layer in self:
            bits = layer(bits)
        return bits


class _Level(nn.Module):
    indices: Tensor

    def __init__(self, indices: list[list[int]]) -> None:
        super().__init__()
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.long), persistent=False)
        self.linear = nn.Linear(5 * _WIDTH, _WIDTH, bias=False)
        self.bias = nn.Parameter(torch.zeros(len(indices), _WIDTH))
        self.norm = nn.LayerNorm(_WIDTH)

    def forward(self, history: Tensor) -> Tensor:
        parents = history[:, self.indices]  # [N, nodes, 4, width]
        product = parents[:, :, 0] * parents[:, :, 1]
        features = torch.cat((parents.flatten(2), product), dim=-1)
        return torch.relu(self.norm(self.linear(features) + self.bias))


class _Structured(nn.Module):
    leaves: Tensor

    def __init__(self, *, shuffled: bool) -> None:
        super().__init__()
        columns = [SHUFFLE[bit] if shuffled else bit for bit in SUPPORT]
        self.register_buffer('leaves', torch.tensor(columns, dtype=torch.long), persistent=False)
        self.embed = nn.Linear(1, _WIDTH)
        self.rounds = nn.ModuleList()
        slots = {(0, bit): index for index, bit in enumerate(SUPPORT)}
        for level in LEVELS:
            indices = [[slots[parent] for parent in parents] for _, parents in level]
            self.rounds.append(_Level(indices))
            for node, _ in level:
                slots[node] = len(slots)
        self.head = nn.Linear(_WIDTH, 1)

    def forward(self, bits: Tensor) -> Tensor:
        _check_shape(bits)
        history = self.embed(bits.index_select(1, self.leaves).unsqueeze(-1))
        for level in self.rounds:
            history = torch.cat((history, level(history)), dim=1)
        return cast(Tensor, self.head(history[:, -1]))


def build(arm: str, seed: int) -> nn.Module:
    """Build an independent arm with reproducible initial parameters."""
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.default_generator.manual_seed(seed)
        if arm == 'mlp':
            return _MLP(
                nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 131), nn.ReLU(), nn.Linear(131, 1)
            )
        if arm in ('correct', 'shuffled'):
            return _Structured(shuffled=arm == 'shuffled')
    raise ValueError(f'unknown model arm: {arm!r}')

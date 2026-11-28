"""Full-word SIMON32 prediction models with public or shuffled input wiring."""
from __future__ import annotations

from hashlib import sha256
from typing import cast

import torch
from torch import Tensor, nn

from ..bit_structure.model import _WIDTH, _check_shape, _Level


def _shuffled_columns() -> list[int]:
    def digest(column: int) -> bytes:
        return sha256(f'simon32-full32-frontier-shuffle-v1/{column}'.encode()).digest()

    columns = list(range(32))
    for start in (0, 16):
        ordered = sorted(range(start, start + 16), key=digest)
        for source, target in zip(ordered, ordered[1:] + ordered[:1], strict=True):
            columns[source] = target
    return columns


class _Full32(nn.Module):
    columns: Tensor

    def __init__(self, *, rounds: int, shuffled: bool) -> None:
        super().__init__()
        columns = _shuffled_columns() if shuffled else list(range(32))
        self.register_buffer('columns', torch.tensor(columns, dtype=torch.long), persistent=False)
        self.embed = nn.Linear(1, _WIDTH)
        # A left rotation reads a positive offset in the MSB0 high word.
        indices = [[(bit + 1) % 16, (bit + 8) % 16, (bit + 2) % 16, 16 + bit]
                   for bit in range(16)]
        self.levels = nn.ModuleList(_Level(indices) for _ in range(rounds))
        self.head = nn.Linear(_WIDTH, 1)

    def forward(self, bits: Tensor) -> Tensor:
        _check_shape(bits)
        words = self.embed(bits.index_select(1, self.columns).unsqueeze(-1))
        for level in self.levels:
            # The new low word copies the preceding high representation.
            words = torch.cat((level(words), words[:, :16]), dim=1)
        return cast(Tensor, self.head(words).squeeze(-1))


def build(arm: str, seed: int, *, rounds: int) -> nn.Module:
    """Build a CPU-initialized arm with matched parameters at each seed and depth."""
    if type(rounds) is not int or not 3 <= rounds <= 6:
        raise ValueError('rounds must be an integer from 3 through 6')
    if arm not in ('correct', 'shuffled'):
        raise ValueError(f'unknown model arm: {arm!r}')
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.default_generator.manual_seed(seed)
        return _Full32(rounds=rounds, shuffled=arm == 'shuffled')

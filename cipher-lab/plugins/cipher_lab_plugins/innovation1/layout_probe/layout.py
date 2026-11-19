"""Public nibble indices in original bit order, independent of data or model state."""
from __future__ import annotations

import torch
from torch import Tensor

from cipher_lab.representations.neighborhoods import neighborhood_indices


def nibble_indices(scan: str) -> Tensor:
    """Return a fresh CPU int64 [64,9] table with the original padding sentinel.

    Predecessor, center and successor follow a continuous scan, including
    connections across row/column boundaries, not strict grid adjacency.
    """
    if scan == 'row':
        with torch.device('cpu'):
            return neighborhood_indices('nibble')
    if scan != 'column':
        raise ValueError('expected row or column scan')

    order = [4 * row + col for col in range(4) for row in range(4)]
    padded = [None, *order, None]
    neighbors = {n: padded[i:i + 3] for i, n in enumerate(order)}
    rows = [[4 * n + b if n is not None and 0 <= b < 4 else 64
             for n in neighbors[bit // 4] for b in range(bit % 4 - 1, bit % 4 + 2)]
            for bit in range(64)]
    return torch.tensor(rows, dtype=torch.long, device='cpu')

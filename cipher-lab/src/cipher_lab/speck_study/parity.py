"""Public MSB-first character arithmetic; no oracle, sampling or model imports."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from torch import Tensor

import numpy as np
from numpy.typing import NDArray

MAX_ROWS = 50_000
TILE_ROWS = 4096


def integer(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name}: expected bounded integer')
    return value


def array_rows(value: object, dtype: np.dtype[np.generic], *, empty: bool = False) -> int:
    # In particular reject MaskedArray: its overridden reductions hide samples.
    if type(value) is not np.ndarray or value.ndim != 1 or value.dtype != dtype:
        raise ValueError('expected native one-dimensional ndarray of exact dtype')
    if not (0 if empty else 1) <= value.size <= MAX_ROWS:
        raise ValueError('array outside bounded batch')
    return int(value.size)


def tiles(n: int, check: Callable[[], None]) -> Iterator[slice]:
    for start in range(0, n, TILE_ROWS):
        check()
        yield slice(start, min(start + TILE_ROWS, n))


def parity_tile(points: NDArray[np.uint64], mask: int) -> NDArray[np.int8]:
    folded = points & np.uint64(mask)
    for shift in (32, 16, 8, 4, 2, 1):
        folded ^= folded >> np.uint64(shift)
    return cast(NDArray[np.int8], (1 - 2 * (folded & np.uint64(1)).astype(np.int8)).astype(np.int8))


def parity_sign(points: NDArray[np.uint64], mask: int) -> NDArray[np.int8]:
    """chi(mask,points); bit63 is column0. At most 50k rows; no sampling."""
    integer(mask, 'mask', 0, 2**64 - 1)
    n = array_rows(points, np.dtype(np.uint64), empty=True)
    result = np.empty(n, dtype=np.int8)
    for start in range(0, n, TILE_ROWS):
        part = slice(start, min(start + TILE_ROWS, n))
        result[part] = parity_tile(points[part], mask)
    return result


def parity_sign_tensor(points: Tensor, mask: int) -> Tensor:
    """Torch int64 bit patterns, including negative values representing uint64.

    Lazy Torch import keeps NumPy-only consumers free of Torch initialization.
    The caller chooses CPU/GPU; this function never initializes CUDA itself.
    """
    import torch

    integer(mask, 'mask', 0, 2**64 - 1)
    if type(points) is not torch.Tensor or points.ndim != 1 or points.dtype != torch.int64:
        raise ValueError('expected native int64 tensor vector')
    if points.numel() > MAX_ROWS:
        raise ValueError('tensor outside bounded batch')
    signed_mask = mask if mask < 2**63 else mask - 2**64
    result = torch.empty(points.shape, dtype=torch.int8, device=points.device)
    for start in range(0, points.numel(), TILE_ROWS):
        folded = points[start:start + TILE_ROWS] & signed_mask
        # Arithmetic shifts have the same final low parity bit as logical shifts.
        for shift in (32, 16, 8, 4, 2, 1):
            folded = folded ^ (folded >> shift)
        result[start:start + TILE_ROWS] = (1 - 2 * (folded & 1)).to(torch.int8)
    return result

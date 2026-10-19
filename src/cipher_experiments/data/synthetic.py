"""Small deterministic synthetic binary classification datasets."""

from __future__ import annotations

import random

import torch
from torch import Tensor

_MAX_SPLIT = 1_000_000
_MAX_TOTAL = 2_000_000
_DRAWS_PER_SAMPLE = 16


def _count(value: int, label: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{label} must be a built-in int")
    if not 1 <= value <= _MAX_SPLIT:
        raise ValueError(f"{label} must be between 1 and {_MAX_SPLIT}")


def _seed(seed: int, label: str = "seed") -> None:
    if type(seed) is not int:
        raise TypeError(f"{label} must be a built-in int")
    if not 0 <= seed < (1 << 32):
        raise ValueError(f"{label} must be between 0 and 2^32-1")


def _counts(train: int, dev: int, test: int) -> int:
    for name, value in (("train", train), ("dev", dev), ("test", test)):
        _count(value, name)
    total = train + dev + test
    if total > _MAX_TOTAL:
        raise ValueError(f"combined split counts must not exceed {_MAX_TOTAL}")
    return total


def _unique_blocks(total: int, seed: int) -> list[int]:
    """Draw globally unique uint64 blocks with a finite local-RNG budget."""
    rng = random.Random(seed)
    blocks: list[int] = []
    seen: set[int] = set()
    for _ in range(total * _DRAWS_PER_SAMPLE):
        value = rng.getrandbits(64)
        if value in seen:
            continue
        seen.add(value)
        blocks.append(value)
        if len(blocks) == total:
            return blocks
    raise RuntimeError("unique uint64 sampling exhausted its bounded draw budget")


def _binary_rows(blocks: list[int]) -> Tensor:
    # Work in bounded chunks so an allowed large dataset does not also require
    # a second full-size Python list-of-lists of bit values.
    output = torch.empty((len(blocks), 64), dtype=torch.float32)
    shifts = torch.arange(63, -1, -1, dtype=torch.int64)
    for start in range(0, len(blocks), 8192):
        part = blocks[start : start + 8192]
        values = torch.tensor(
            [value if value < (1 << 63) else value - (1 << 64) for value in part],
            dtype=torch.int64,
        )
        output[start : start + len(part)] = ((values[:, None] >> shifts) & 1).to(torch.float32)
    return output


def _split(
    x: Tensor, y: Tensor, train: int, dev: int
) -> dict[str, tuple[Tensor, Tensor]]:
    return {
        "train": (x[:train].contiguous(), y[:train].contiguous()),
        "dev": (x[train : train + dev].contiguous(), y[train : train + dev].contiguous()),
        "test": (x[train + dev :].contiguous(), y[train + dev :].contiguous()),
    }


def make_synthetic(
    train: int, dev: int, test: int, *, seed: int
) -> dict[str, tuple[Tensor, Tensor]]:
    """Create unique uint64 feature rows with the learnable target ``label=X[0]``.

    Returns train/dev/test pairs of CPU FP32 tensors; X has shape [N,64], and
    y has shape [N,1]. Split identities are sampled once and are globally
    unique. This is an engineering fixture, not cipher-derived data.
    """
    total = _counts(train, dev, test)
    _seed(seed)
    x = _binary_rows(_unique_blocks(total, seed))
    y = x[:, :1].clone()
    return _split(x, y, train, dev)

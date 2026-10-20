"""Finite artificial Boolean task, not cipher data or a SIMON approximation."""
from __future__ import annotations

import torch
from torch import Tensor

from .synthetic import _seed


def make_circuit_inputs(train: int, dev: int, test: int, *, seed: int) -> dict[str, Tensor]:
    """Shuffle distinct 8-bit assignments, then embed into 64 binary inputs.

    The other 56 bits are zero, so uniqueness holds on the actual relevant
    domain, not merely on irrelevant padding. No relevant input repeats across
    splits. Splits are independent of target values (no label stratification).
    """
    if any(type(n) is not int or n < 1 for n in (train, dev, test)):
        raise ValueError("circuit split counts must be positive built-in integers")
    if train + dev + test > 256:
        raise ValueError("circuit task has only 256 relevant input assignments")
    _seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randperm(256, generator=generator)[:train + dev + test]
    inputs = torch.zeros((len(values), 64), dtype=torch.float32)
    inputs[:, :8] = ((values[:, None] >> torch.arange(7, -1, -1)) & 1).float()
    return {"train": inputs[:train], "dev": inputs[train:train + dev],
            "test": inputs[train + dev:]}


def circuit_labels(inputs: Tensor) -> Tensor:
    """XOR four disjoint adjacent ANDs; target depends on all first eight bits."""
    if (not isinstance(inputs, Tensor) or inputs.device.type != "cpu"
            or inputs.layout != torch.strided or inputs.dtype != torch.float32
            or inputs.ndim != 2 or inputs.shape[1] != 64 or len(inputs) < 1):
        raise ValueError("circuit labels require nonempty dense CPU FP32 [N,64]")
    if inputs.requires_grad or not bool(((inputs == 0) | (inputs == 1)).all()):
        raise ValueError("circuit inputs must be detached binary values")
    bits = inputs[:, :8].bool()
    result = ((bits[:, 0] & bits[:, 1]) ^ (bits[:, 2] & bits[:, 3])
              ^ (bits[:, 4] & bits[:, 5]) ^ (bits[:, 6] & bits[:, 7]))
    return result.float().unsqueeze(1)

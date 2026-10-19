"""Bounded balanced differential-pair material and ciphertext batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from cipher_experiments.ciphers.expanded import encrypt_expanded
from cipher_experiments.ciphers.tasks import TaskSpec

_MAX_GROUPS = 65_536
_MAX_SEED = 1 << 32


@dataclass(frozen=True)
class Material:
    """Plaintext/key material; keys stay inside data generation, not model inputs."""

    keys: tuple[int, ...]
    first: tuple[tuple[int, ...], ...]
    second: tuple[tuple[int, ...], ...]
    labels: tuple[int, ...]


def _integers(rng: np.random.Generator, count: int, bits: int) -> list[int]:
    raw = rng.integers(0, 256, (count, bits // 8), dtype=np.uint8)
    return [int.from_bytes(row.tobytes(), "big") for row in raw]


def _validate(groups: int, seed: int) -> None:
    if type(groups) is not int or groups < 2 or groups > _MAX_GROUPS or groups % 2:
        raise ValueError(f"groups must be even and between 2 and {_MAX_GROUPS}")
    if type(seed) is not int or not 0 <= seed < _MAX_SEED:
        raise ValueError("seed must be a built-in int between 0 and 2^32-1")


def generate_material(spec: TaskSpec, *, groups: int, seed: int) -> Material:
    """Create balanced groups: XOR-related positives and independent negatives."""
    if not isinstance(spec, TaskSpec):
        raise TypeError("spec must be a TaskSpec")
    _validate(groups, seed)
    rng = np.random.default_rng(seed)
    keys = _integers(rng, groups, spec.key_bits)
    first_flat = _integers(rng, groups * 4, spec.block_bits)
    negative = _integers(rng, groups * 2, spec.block_bits)
    first = [tuple(first_flat[i * 4 : (i + 1) * 4]) for i in range(groups)]
    second = [tuple(x ^ spec.input_difference for x in row) for row in first[: groups // 2]]
    second.extend(tuple(negative[i * 4 : (i + 1) * 4]) for i in range(groups // 2))
    order = rng.permutation(groups).tolist()
    return Material(
        tuple(keys[i] for i in order),
        tuple(first[i] for i in order),
        tuple(second[i] for i in order),
        tuple(int(i < groups // 2) for i in order),
    )


@dataclass(frozen=True)
class Batch:
    """Encrypted four-pair batch, with MSB-first bits and binary labels."""

    bits: Tensor
    label: Tensor

    def __len__(self) -> int:
        return int(self.label.shape[0])

    def to(self, device: torch.device | str) -> Batch:
        return Batch(self.bits.to(device), self.label.to(device))


def _pack(rows: list[list[int]]) -> Tensor:
    if not rows or any(len(row) != 8 for row in rows):
        raise ValueError("expected nonempty four-pair rows")
    values = [value for row in rows for value in row]
    if any(value < 0 or value >= 1 << 64 for value in values):
        raise ValueError("ciphertext outside block width")
    raw = b"".join(value.to_bytes(8, "big") for value in values)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(len(rows), 4, 2, 64)
    return torch.from_numpy(bits.copy()).to(torch.float32)


def generate_batch(spec: TaskSpec, *, groups: int, seed: int) -> Batch:
    """Encrypt both classes under one fresh shared key per eight-block group."""
    material = generate_material(spec, groups=groups, seed=seed)
    plaintexts = [
        plaintext
        for first, second in zip(material.first, material.second, strict=True)
        for pair in zip(first, second, strict=True)
        for plaintext in pair
    ]
    keys = [key for key in material.keys for _ in range(8)]
    encrypted = encrypt_expanded(spec.cipher, spec.rounds, keys, plaintexts)
    return Batch(
        _pack([encrypted[i * 8 : (i + 1) * 8] for i in range(groups)]),
        torch.tensor(material.labels, dtype=torch.float32),
    )

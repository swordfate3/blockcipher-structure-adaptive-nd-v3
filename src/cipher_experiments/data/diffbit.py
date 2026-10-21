"""Fixed-difference output-bit data: plaintext P -> bit of E_K(P) xor E_K(P^delta).

Key material stays in this module. TRAIN and the evaluation splits draw from
disjoint key ranges, so a model that scores above the constant baseline on DEV or
TEST has generalised to keys it was never trained on.
"""

from __future__ import annotations

import hashlib
import random

import torch
from torch import Tensor

from cipher_experiments.ciphers.simon import encrypt as _encrypt
from cipher_experiments.data.synthetic import (
    _binary_rows,
    _counts,
    _seed,
    _split,
    _unique_blocks,
)

_SPLITS = ("train", "dev", "test")
_MAX_ROUNDS = 6
_MAX_ROWS = 2_000_000
_MAX_KEYS = 4096


def _validate(delta: int, rounds: int, target_bit: int, key_count: int,
              train_key_count: int) -> None:
    if type(delta) is not int:
        raise TypeError("input_difference must be a built-in int")
    if not 0 < delta < (1 << 64):
        raise ValueError("input_difference must be a nonzero unsigned 64-bit integer")
    if type(rounds) is not int or not 1 <= rounds <= _MAX_ROUNDS:
        raise ValueError(f"rounds must be between 1 and {_MAX_ROUNDS}")
    if type(target_bit) is not int or not 0 <= target_bit < 64:
        raise ValueError("target_bit must be between 0 and 63")
    if type(key_count) is not int or not 1 <= key_count <= _MAX_KEYS:
        raise ValueError(f"key_count must be between 1 and {_MAX_KEYS}")
    if type(train_key_count) is not int or not 1 <= train_key_count <= key_count:
        raise ValueError("train_key_count must be between 1 and key_count")


def key_for(key_seed: int, index: int) -> int:
    """Deterministic per-index 128-bit key; the value never leaves this module."""
    material = b"cipher-experiments/diff-key/v1\0" + key_seed.to_bytes(4, "big") + b"\0"
    material += index.to_bytes(4, "big")
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def key_indices(split: str, key_count: int, train_key_count: int) -> tuple[int, ...]:
    """Disjoint key index ranges: TRAIN holds the first block, DEV/TEST the rest."""
    if split not in _SPLITS:
        raise ValueError("split must be train, dev or test")
    if split == "train":
        return tuple(range(train_key_count))
    held_out = tuple(range(train_key_count, key_count))
    if not held_out:
        raise ValueError("no held-out keys: train_key_count must be below key_count")
    # DEV and TEST share the held-out pool but are separated by row parity below.
    return held_out


def make_diff_inputs(train: int, dev: int, test: int, *, seed: int) -> dict[str, Tensor]:
    """Unique plaintext rows without labels, one panel per split."""
    total = _counts(train, dev, test)
    _seed(seed)
    inputs = _binary_rows(_unique_blocks(total, seed))
    return {name: pair[0] for name, pair in _split(inputs, inputs[:, :0], train, dev).items()}


def _pack(inputs: Tensor) -> list[int]:
    result: list[int] = []
    for start in range(0, inputs.shape[0], 8192):
        bits = inputs[start:start + 8192].to(torch.int64)
        packed = torch.zeros(bits.shape[0], dtype=torch.int64)
        for column in range(64):
            packed = (packed << 1) | bits[:, column]
        result.extend(int(v) if v >= 0 else int(v) + (1 << 64) for v in packed.tolist())
    return result


def make_diff_labels(inputs: Tensor, *, delta: int, rounds: int, target_bit: int,
                     key_seed: int, key_count: int, train_key_count: int,
                     split: str) -> Tensor:
    """Bit of ``E_K(P) xor E_K(P^delta)``; keys rotate inside this split's range."""
    rows = int(inputs.shape[0])
    if not 1 <= rows <= _MAX_ROWS:
        raise ValueError(f"row count must be between 1 and {_MAX_ROWS}")
    indices = key_indices(split, key_count, train_key_count)
    keys = tuple(key_for(key_seed, index) for index in indices)
    plaintexts = _pack(inputs)
    mask = 1 << (63 - target_bit)
    labels = torch.empty((rows, 1), dtype=torch.float32)
    for offset, plaintext in enumerate(plaintexts):
        key = keys[offset % len(keys)]
        difference = _encrypt(plaintext, key, rounds) ^ _encrypt(plaintext ^ delta, key, rounds)
        labels[offset, 0] = float(bool(difference & mask))
    return labels


def round_one_reference(inputs: Tensor, *, delta: int, target_bit: int) -> Tensor:
    """Key-free exact differential bit for one round.

    One SIMON round maps ``(L, R) -> (R ^ f(L) ^ k, L)``, so the difference
    ``(dR ^ (f(L) ^ f(L^dL)), dL)`` contains no round key at all. This is a
    mathematical identity, used as a positive control rather than as a model.
    """
    mask32 = (1 << 32) - 1

    def f(word: int) -> int:
        left = ((word << 1) | (word >> 31)) & mask32
        right = ((word << 8) | (word >> 24)) & mask32
        two = ((word << 2) | (word >> 30)) & mask32
        return (left & right) ^ two

    high_delta, low_delta = (delta >> 32) & mask32, delta & mask32
    mask = 1 << (63 - target_bit)
    bits: list[float] = []
    for plaintext in _pack(inputs):
        high = (plaintext >> 32) & mask32
        next_low = low_delta ^ f(high) ^ f(high ^ high_delta)
        next_high = high_delta
        difference = (next_high << 32) | next_low
        bits.append(float(bool(difference & mask)))
    return torch.tensor(bits, dtype=torch.float32).unsqueeze(1)


def make_null_labels(inputs: Tensor, *, null_seed: int, split: str) -> Tensor:
    """Matched random labels, domain-separated per split."""
    if split not in _SPLITS:
        raise ValueError("split must be train, dev or test")
    material = (b"cipher-experiments/diff-null/v1\0" + null_seed.to_bytes(4, "big")
                + b"\0" + split.encode("ascii"))
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))
    rows = int(inputs.shape[0])
    return torch.tensor([[float(rng.getrandbits(1))] for _ in range(rows)], dtype=torch.float32)


def bias_summary(labels: Tensor, *, keys_per_row: int) -> dict[str, float]:
    """Describe label bias so a constant predictor is never mistaken for a model.

    ``positive_rate`` near 0 or 1 means the task is trivial for a constant
    baseline; ``spread`` is the max-min positive rate over row groups (one group
    per key when ``keys_per_row`` > 1).
    """
    flat = labels.flatten().tolist()
    if not flat:
        raise ValueError("labels must be nonempty")
    if any(value not in (0.0, 1.0) for value in flat):
        raise ValueError("labels must be binary")
    rows = len(flat)
    keys_per_row = max(1, int(keys_per_row))
    rates = []
    for start in range(0, rows, keys_per_row):
        group = flat[start:start + keys_per_row]
        rates.append(sum(group) / len(group))
    return {
        "rows": float(rows),
        "positive_rate": sum(flat) / rows,
        "min_key_rate": min(rates),
        "max_key_rate": max(rates),
        "spread": max(rates) - min(rates),
    }


def key_marginal_probability(inputs: Tensor, *, delta: int, rounds: int, target_bit: int,
                             key_seed: int, train_key_count: int) -> Tensor:
    """Per-row positive fraction over the TRAIN key pool.

    This is the strongest key-agnostic predictor available to a model that has
    seen only TRAIN keys: it uses the row's plaintext and the TRAIN keys, never a
    held-out key or a held-out label. It is the baseline a differential model has
    to beat before "the model learned something beyond the bias" is supportable.
    """
    if type(train_key_count) is not int or train_key_count < 1:
        raise ValueError("train_key_count must be a positive integer")
    rows = int(inputs.shape[0])
    if not 1 <= rows <= _MAX_ROWS:
        raise ValueError(f"row count must be between 1 and {_MAX_ROWS}")
    keys = tuple(key_for(key_seed, index) for index in range(train_key_count))
    plaintexts = _pack(inputs)
    mask = 1 << (63 - target_bit)
    fractions = torch.empty((rows, 1), dtype=torch.float32)
    for offset, plaintext in enumerate(plaintexts):
        hits = 0
        for key in keys:
            difference = (_encrypt(plaintext, key, rounds)
                          ^ _encrypt(plaintext ^ delta, key, rounds))
            hits += int(bool(difference & mask))
        fractions[offset, 0] = hits / len(keys)
    return fractions

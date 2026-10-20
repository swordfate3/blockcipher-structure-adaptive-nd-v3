"""Unlabeled prediction inputs and caller-controlled label generation.

Inputs are generated independently of labels, so callers can build every split
before fitting. This API does not enforce checkpoint/TEST access ordering: a
caller must request TEST labels only after choosing and strictly loading its
checkpoint. TEST is pre-generated data, not a sealed confirmation set.
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
_MAX_ROUNDS = 44
_MAX_ROWS = 2_000_000


def make_inputs(train: int, dev: int, test: int, *, seed: int) -> dict[str, Tensor]:
    """Generate globally unique CPU-FP32 plaintext rows without making labels."""
    total = _counts(train, dev, test)
    _seed(seed)
    inputs = _binary_rows(_unique_blocks(total, seed))
    # Reuse the canonical split boundaries without constructing target labels.
    return {name: pair[0] for name, pair in _split(inputs, inputs[:, :0], train, dev).items()}


def _validate_inputs(inputs: Tensor) -> int:
    if not isinstance(inputs, Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    if inputs.device.type != "cpu":
        raise ValueError("inputs must be on CPU")
    if inputs.dtype != torch.float32:
        raise TypeError("inputs must have dtype torch.float32")
    if inputs.requires_grad:
        raise ValueError("inputs must be detached")
    if inputs.ndim != 2 or inputs.shape[1] != 64:
        raise ValueError("inputs must have shape [N,64]")
    rows = inputs.shape[0]
    if not 1 <= rows <= _MAX_ROWS:
        raise ValueError(f"inputs row count must be between 1 and {_MAX_ROWS}")
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("inputs must contain only finite values")
    if not bool(((inputs == 0) | (inputs == 1)).all()):
        raise ValueError("inputs must contain only binary values 0 and 1")
    return rows


def _plaintexts(inputs: Tensor) -> list[int]:
    """Pack FP32 bit rows into unsigned 64-bit Python values in bounded chunks."""
    result: list[int] = []
    for start in range(0, inputs.shape[0], 8192):
        bits = inputs[start : start + 8192].to(torch.int64)
        packed = torch.zeros(bits.shape[0], dtype=torch.int64)
        for column in range(64):
            packed = (packed << 1) | bits[:, column]
        result.extend(value if value >= 0 else value + (1 << 64) for value in packed.tolist())
    return result


def _null_rng(null_seed: int, split: str) -> random.Random:
    # Fixed domain separator and fixed-width seed encoding avoid ambiguous text
    # concatenation while keeping each split's stream independent.
    material = b"cipher-experiments/prediction-null/v1\0"
    material += null_seed.to_bytes(4, "big") + b"\0" + split.encode("ascii")
    stream_seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
    return random.Random(stream_seed)


def key_for_seed(key_seed: int) -> int:
    """Derive the 128-bit data-generation key for one seed.

    Audit tooling may call this to compare learned constants with the true round
    keys **after** fitting. It must not be used to initialise or constrain a
    model, and it stays outside the model and fitting packages.
    """
    _seed(key_seed, "key_seed")
    return random.Random(key_seed).getrandbits(128)


def make_labels(
    inputs: Tensor,
    *,
    dataset: str,
    rounds: int,
    target_bit: int,
    key_seed: int,
    label_mode: str = "cipher",
    null_seed: int = 191302,
    split: str,
) -> Tensor:
    """Create one detached CPU-FP32 label per input row.

    ``synthetic`` labels are the first input bit and never invoke a cipher.
    ``simon`` labels are the selected MSB-first ciphertext bit under a key
    generated locally from ``key_seed``; the key is never returned. ``null``
    labels use a deterministic local RNG stream independently domain-separated
    by the explicit train/dev/test split.
    """
    rows = _validate_inputs(inputs)
    if dataset not in ("synthetic", "simon"):
        raise ValueError("dataset must be synthetic or simon")
    if label_mode not in ("cipher", "null"):
        raise ValueError("label_mode must be cipher or null")
    if split not in _SPLITS:
        raise ValueError("split must be train, dev or test")
    _seed(key_seed, "key_seed")
    _seed(null_seed, "null_seed")
    if type(rounds) is not int:
        raise TypeError("rounds must be a built-in int")
    if not 1 <= rounds <= _MAX_ROUNDS:
        raise ValueError(f"rounds must be between 1 and {_MAX_ROUNDS}")
    if type(target_bit) is not int:
        raise TypeError("target_bit must be a built-in int")
    if not 0 <= target_bit < 64:
        raise ValueError("target_bit must be between 0 and 63")
    if label_mode == "null" and dataset != "simon":
        raise ValueError("null labels require dataset=simon")

    if label_mode == "null":
        rng = _null_rng(null_seed, split)
        return torch.tensor([[float(rng.getrandbits(1))] for _ in range(rows)], dtype=torch.float32)
    if dataset == "synthetic":
        return inputs[:, :1].clone().contiguous()

    key = key_for_seed(key_seed)
    mask = 1 << (63 - target_bit)
    labels = torch.empty((rows, 1), dtype=torch.float32)
    for start in range(0, rows, 8192):
        part = _plaintexts(inputs[start : start + 8192])
        labels[start : start + len(part), 0] = torch.tensor(
            [float(bool(_encrypt(plaintext, key, rounds) & mask)) for plaintext in part],
            dtype=torch.float32,
        )
    return labels

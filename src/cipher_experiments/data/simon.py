"""Bounded SIMON64/128 supervised data generation.

Labels are computed here by the local scalar cipher implementation; this
module is the only layer in this package that receives the data-generation key.
"""

from __future__ import annotations

import random

import torch
from torch import Tensor

from cipher_experiments.ciphers.simon import encrypt
from cipher_experiments.data.synthetic import (
    _binary_rows,
    _counts,
    _seed,
    _split,
    _unique_blocks,
)

_MAX_ROUNDS = 44


def make_simon(
    train: int,
    dev: int,
    test: int,
    *,
    seed: int,
    key_seed: int,
    rounds: int,
    target_bit: int,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Generate unique plaintext rows and selected ciphertext-bit labels.

    ``target_bit`` is an MSB-first index in ``0..63``. Each dataset uses one
    128-bit key sampled from a dedicated local RNG seeded by ``key_seed``.
    Inputs and labels are detached CPU FP32 tensors with shapes [N,64] and
    [N,1]. The key is not returned or retained in the result. This bounded
    utility is for caller-controlled engineering data, not formal fitting.
    """
    total = _counts(train, dev, test)
    _seed(seed)
    _seed(key_seed, "key_seed")
    if type(rounds) is not int:
        raise TypeError("rounds must be a built-in int")
    if not 1 <= rounds <= _MAX_ROUNDS:
        raise ValueError(f"rounds must be between 1 and {_MAX_ROUNDS}")
    if type(target_bit) is not int:
        raise TypeError("target_bit must be a built-in int")
    if not 0 <= target_bit < 64:
        raise ValueError("target_bit must be between 0 and 63")

    plaintexts = _unique_blocks(total, seed)
    key = random.Random(key_seed).getrandbits(128)
    mask = 1 << (63 - target_bit)
    labels = torch.tensor(
        [[float(bool(encrypt(plaintext, key, rounds) & mask))] for plaintext in plaintexts],
        dtype=torch.float32,
    )
    x = _binary_rows(plaintexts)
    return _split(x, labels, train, dev)

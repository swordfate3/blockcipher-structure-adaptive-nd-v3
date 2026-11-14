"""Torch-only arithmetic equivalent of speck.py; no device selection or secrets.

points: int64 tensor [N] of literal uint32 blocks. key_words: int64 [4] for
one key or [N,4] per-row keys, same official high-to-low order as scalar.
Caller chooses device, owns ephemeral key tensors, and accounts all calls.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _words(value: Tensor) -> None:
    if type(value) is not Tensor or value.dtype != torch.int64:
        raise ValueError('expected native int64 tensor')
    if bool(torch.any((value < 0) | (value > 65535))):
        raise ValueError('word outside uint16')


def enc_one_round(pair: tuple[Tensor, Tensor], key: Tensor | int) -> tuple[Tensor, Tensor]:
    x, y = pair
    x = (x >> 7) | ((x << 9) & 65535)
    x = ((x + y) & 65535) ^ key
    y = ((y << 2) & 65535) | (y >> 14)
    return x, y ^ x


def expand_key(key_words: Tensor, rounds: int) -> tuple[Tensor, ...]:
    if type(rounds) is not int or not 1 <= rounds <= 22:
        raise ValueError('rounds must be in 1..22')
    _words(key_words)
    if key_words.ndim not in (1, 2) or key_words.shape[-1] != 4:
        raise ValueError('keys must be [4] or [N,4]')
    keys = [key_words[..., 3]]
    lane = [key_words[..., i] for i in (2, 1, 0)]
    for i in range(rounds - 1):
        lane[i % 3], next_key = enc_one_round((lane[i % 3], keys[-1]), i)
        keys.append(next_key)
    return tuple(keys)


def encrypt_tensor(points: Tensor, key_words: Tensor, rounds: int) -> Tensor:
    if type(points) is not Tensor or points.dtype != torch.int64 or points.ndim != 1:
        raise ValueError('points must be native int64 vector')
    if not 1 <= points.numel() <= 50_000 or bool(torch.any((points < 0) | (points >= 2**32))):
        raise ValueError('points outside bounded uint32 batch')
    _words(key_words)
    if key_words.device != points.device or key_words.shape not in ((4,), (len(points), 4)):
        raise ValueError('key shape/device mismatch')
    pair = points >> 16, points & 65535
    for key in expand_key(key_words, rounds):
        pair = enc_one_round(pair, key)
    return (pair[0] << 16) | pair[1]

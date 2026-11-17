"""Keyless dedicated input layouts; raw MSB0 [B,4,2,64] only.

Source paths and precise SHA256s are recorded in the Task1 delivery report.
"""
from __future__ import annotations

import torch

from cipher_lab.representations.input import public_backstep, validate_bits


def relation_words(bits: torch.Tensor, cipher: str) -> torch.Tensor:
    validate_bits(bits)
    if cipher not in ("simon64", "simeck64"):
        raise ValueError("expected SIMON64 or SIMECK64")
    if bits.ndim != 4 or bits.shape[1:] != (4, 2, 64):
        raise ValueError("expected exactly four pairs of 64-bit ciphertexts")
    rotations = (1, 8, 2) if cipher == "simon64" else (0, 5, 1)

    def f(x: torch.Tensor) -> torch.Tensor:
        a, b, c = rotations
        return (torch.roll(x, -a, -1) & torch.roll(x, -b, -1)) ^ torch.roll(x, -c, -1)

    words = bits.bool().reshape(-1, 4, 4, 32)
    left, right, other_left, other_right = words.unbind(2)
    before, other_before = left ^ f(right), other_left ^ f(other_right)
    # The last feature is deliberately key-omitted D, not true decryption of r-2.
    return torch.stack(
        (
            left ^ other_left,
            right ^ other_right,
            left,
            right,
            other_left,
            other_right,
            before ^ other_before,
            right ^ f(before) ^ other_right ^ f(other_before),
        ),
        dim=2,
    ).float()


def component_input(bits: torch.Tensor) -> torch.Tensor:
    validate_bits(bits)
    raw = bits.float().reshape(-1, 4, 8, 16).flatten(0, 1)
    extra = (raw[:, :4] != raw[:, 4:]).float()
    return torch.cat((raw, extra), dim=1)


def author_pair_layout(bits: torch.Tensor, *, rounds: int) -> torch.Tensor:
    validate_bits(bits)
    viewed = public_backstep(bits, rounds)
    triplet = torch.cat((viewed, (viewed[:, :, :1] != viewed[:, :, 1:]).float()), dim=2)
    # Exact author reshape (3,4,16), NOT a transpose of (3,16,4).
    return triplet.reshape(bits.shape[0] * 4, 3, 4, 16).permute(0, 2, 1, 3).contiguous()

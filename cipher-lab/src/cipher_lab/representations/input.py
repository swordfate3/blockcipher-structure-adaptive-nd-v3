"""Audited 64-bit public input views, separate from routing and readout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from cipher_lab.representations.tasks import validate_public, validate_rounds


def validate_bits(bits: torch.Tensor) -> None:
    if (not isinstance(bits, torch.Tensor) or bits.layout != torch.strided
            or bits.is_complex()):
        raise ValueError("expected dense real binary bits")
    if bits.ndim != 4 or bits.shape[1:] != (4, 2, 64) or bits.shape[0] < 1:
        raise ValueError("expected bits [B,4,2,64], exactly four 64-bit pairs")
    if not bool(torch.isfinite(bits).all() & ((bits == 0) | (bits == 1)).all()):
        raise ValueError("ciphertext bits must be finite and binary")


def canonical_input(bits: torch.Tensor) -> torch.Tensor:
    validate_bits(bits)
    raw = bits.float()
    return torch.cat((raw, (raw[:, :, :1] != raw[:, :, 1:]).float()), dim=2)


def input_view(bits: torch.Tensor, public: torch.Tensor, *, rounds: int) -> torch.Tensor:
    """Apply SKINNY's public keyless last-round backstep exactly once, before XOR."""
    validate_bits(bits)
    validate_rounds(public, rounds)
    if bits.device != public.device:
        raise ValueError("bits and public must be on the same device")
    if bool(public[2] == 1 and public[-1] == 1):
        return public_backstep(bits, rounds)
    return bits


def public_rotations(public: torch.Tensor) -> tuple[int, int, int] | None:
    validate_public(public)
    if bool(public[2] == 1):
        return None
    return cast(tuple[int, int, int], tuple(int(float(x) * 32) for x in public[8:11]))


def _input_channel(raw: torch.Tensor, public: torch.Tensor) -> torch.Tensor:
    rotations = public_rotations(public)
    if rotations is None:
        return raw.new_zeros((raw.shape[0], 4, 1, 64))
    a, b, c = rotations

    def f(x: torch.Tensor) -> torch.Tensor:
        return (torch.roll(x, -a, -1) & torch.roll(x, -b, -1)) ^ torch.roll(x, -c, -1)

    words = raw[:, :, :2].bool().reshape(-1, 4, 4, 32)
    left, right, other_left, other_right = words.unbind(2)
    before, other_before = left ^ f(right), other_left ^ f(other_right)
    first = before ^ other_before
    second = right ^ f(before) ^ other_right ^ f(other_before)
    return torch.cat((first, second), dim=-1).unsqueeze(2).to(raw.dtype)


@dataclass(frozen=True)
class RepresentedInput:
    canonical: torch.Tensor
    channel: torch.Tensor


class InputRepresentation(nn.Module):
    """One canonical single view plus the public two-word relation channel."""

    def forward(self, bits: torch.Tensor, public: torch.Tensor, *, rounds: int) -> RepresentedInput:
        raw = canonical_input(input_view(bits, public, rounds=rounds))
        return RepresentedInput(raw, _input_channel(raw, public))


def public_backstep(bits: torch.Tensor, rounds: int) -> torch.Tensor:
    """Map [B,4,2,64] MSB-first binary input to the same shape/dtype/device.

    Apply inverse MixColumns and inverse ShiftRows, then XOR the public round constant.
    Inverse S-boxes apply ONLY to the lower half, where no key was injected.
    This is not full round decryption, and accepts no secret/sample metadata.
    """
    validate_bits(bits)
    if type(rounds) is not int or not 1 <= rounds <= 32:
        raise ValueError("SKINNY public rounds must be in 1..32")
    weights = torch.tensor([8, 4, 2, 1], device=bits.device)
    cells = (bits.reshape(bits.shape[0], 4, 2, 16, 4).long() * weights).sum(-1)
    c = cells.long()
    # Invert MC: output rows were (a^c^d, a, b^c, a^c).
    a = c[..., 4:8]
    third = c[..., 12:16] ^ a
    b = c[..., 8:12] ^ third
    d = c[..., :4] ^ c[..., 12:16]
    before_sr = torch.cat((a, b, third, d), dim=-1)
    s = before_sr[..., [0, 1, 2, 3, 5, 6, 7, 4, 10, 11, 8, 9, 15, 12, 13, 14]].clone()
    constant = 1
    for _ in range(rounds - 1):
        constant = ((constant << 1) & 63) ^ ((constant >> 5) & 1) ^ ((constant >> 4) & 1) ^ 1
    s[..., 0] ^= constant & 15
    s[..., 4] ^= constant >> 4
    s[..., 8] ^= 2
    inverse = torch.tensor([3, 4, 6, 8, 12, 10, 1, 14, 9, 2, 5, 7, 0, 11, 13, 15], device=s.device)
    # Last-round key injection affects only the top half. Do NOT undo its S-boxes.
    s[..., 8:] = inverse[s[..., 8:]]
    shifts = torch.arange(3, -1, -1, device=bits.device)
    return (((s[..., None] >> shifts) & 1).reshape(bits.shape)).to(bits.dtype)

"""Audited 64-bit public input views, separate from routing and readout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from cipher_experiments.ciphers.skinny_adapter import public_backstep
from cipher_experiments.ciphers.tasks import validate_public as validate_public


def validate_bits(bits: torch.Tensor) -> None:
    if bits.ndim != 4 or bits.shape[1:] != (4, 2, 64) or bits.shape[0] < 1:
        raise ValueError("expected bits [B,4,2,64], exactly four 64-bit pairs")
    if not bool(torch.isfinite(bits).all() & ((bits == 0) | (bits == 1)).all()):
        raise ValueError("ciphertext bits must be finite and binary")


def flat_to_bits(flat: torch.Tensor) -> torch.Tensor:
    """Explicit compatibility adapter; forward never guesses a flat layout."""
    if flat.ndim != 2 or flat.shape[1] != 512 or flat.shape[0] < 1:
        raise ValueError("flat adapter requires [B,512]")
    bits = flat.reshape(-1, 4, 2, 64)
    validate_bits(bits)
    return bits


def canonical_input(bits: torch.Tensor) -> torch.Tensor:
    validate_bits(bits)
    raw = bits.float()
    return torch.cat((raw, (raw[:, :, :1] != raw[:, :, 1:]).float()), dim=2)


def input_view(bits: torch.Tensor, public: torch.Tensor) -> torch.Tensor:
    """Apply SKINNY's public keyless r8 backstep exactly once, before XOR."""
    validate_bits(bits)
    validate_public(public)
    if bits.device != public.device:
        raise ValueError("bits and public must be on the same device")
    if bool(public[2] == 1 and public[-1] == 1):
        # The cipher adapter owns bit packing and the exact audited r8 formula.
        return public_backstep(bits)
    return bits


def public_rotations(public: torch.Tensor) -> tuple[int, int, int] | None:
    validate_public(public)
    if bool(public[2] == 1):
        return None
    return cast(tuple[int, int, int], tuple(int(float(x) * 32) for x in public[8:11]))


def _input_channel(raw: torch.Tensor, public: torch.Tensor, variant: str) -> torch.Tensor:
    if variant not in ("raw", "relation"):
        raise ValueError("only raw and relation representations are supported")
    rotations = public_rotations(public)
    if rotations is None:
        return raw.new_zeros((raw.shape[0], 4, 1, 64))
    if variant == "raw":
        return raw[:, :, 2:3]
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
    def __init__(self, variant: str = "relation") -> None:
        super().__init__()
        if variant not in ("raw", "relation"):
            raise ValueError("unknown representation")
        self.variant = variant

    def forward(self, bits: torch.Tensor, public: torch.Tensor) -> RepresentedInput:
        raw = canonical_input(input_view(bits, public))
        return RepresentedInput(raw, _input_channel(raw, public, self.variant))

    def dual_views(
        self, bits: torch.Tensor, public: torch.Tensor, *, view_mode: str
    ) -> tuple[RepresentedInput, RepresentedInput]:
        """TT=(T,T), CT=(C,T) for SKINNY; repeat the existing view otherwise.

        C is original ciphertext *including XOR*, not the ``raw`` channel control.
        Both slots share one public backstep call; T is never backstepped again.
        """
        if view_mode not in ("tt", "ct"):
            raise ValueError("dual views require tt or ct")
        transformed = cast(RepresentedInput, self(bits, public))
        if view_mode == "ct" and bool(public[2] == 1 and public[-1] == 1):
            original = canonical_input(bits)
            return RepresentedInput(
                original, _input_channel(original, public, self.variant)
            ), transformed
        return transformed, transformed


def input_channel(bits: torch.Tensor, public: torch.Tensor, *, variant: str) -> torch.Tensor:
    represented = cast(RepresentedInput, InputRepresentation(variant)(bits, public))
    return represented.channel


def public_relation_channel(bits: torch.Tensor, public: torch.Tensor) -> torch.Tensor:
    return input_channel(bits, public, variant="relation")

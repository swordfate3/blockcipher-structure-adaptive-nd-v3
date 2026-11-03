"""Exact binary public coordinates, interactions, and zero-round-constant reference.

These are key-free tensor operations, not observations of cipher state. In
particular, zero constants in every round do NOT mean a zero master key.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _binary_inputs(inputs: Tensor) -> Tensor:
    if not isinstance(inputs, Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    if (inputs.layout != torch.strided or inputs.ndim != 2
            or inputs.shape[0] < 1 or inputs.shape[1] != 64):
        raise ValueError("inputs must have nonempty shape [N,64]")
    if inputs.dtype not in (torch.bool, torch.float32):
        raise TypeError("inputs must have dtype torch.bool or torch.float32")
    if inputs.dtype == torch.float32:
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("inputs must be finite")
        if not bool(((inputs == 0) | (inputs == 1)).all()):
            raise ValueError("inputs must contain binary values")
    return inputs.to(dtype=torch.bool)


def _round_function(high: Tensor) -> Tensor:
    # Left word rotation is a negative column roll in MSB-first order.
    return (torch.roll(high, -1, dims=1) & torch.roll(high, -8, dims=1)) ^ torch.roll(
        high, -2, dims=1
    )


def _coordinates(bits: Tensor) -> Tensor:
    high, low = bits[:, :32], bits[:, 32:]
    return torch.cat((low ^ _round_function(high), high), dim=1)


def public_coordinates(inputs: Tensor) -> Tensor:
    """Return phi(P)=[Y XOR F(X),X], in MSB order as FP32[N,64]."""
    return _coordinates(_binary_inputs(inputs)).to(dtype=torch.float32)


def public_interactions(inputs: Tensor) -> Tensor:
    """Return phi followed by AND(ROL1(phi_high),ROL8(phi_high)), FP32[N,96]."""
    coordinates = _coordinates(_binary_inputs(inputs))
    high = coordinates[:, :32]
    interactions = torch.roll(high, -1, dims=1) & torch.roll(high, -8, dims=1)
    return torch.cat((coordinates, interactions), dim=1).to(dtype=torch.float32)


def zero_round_constant_reference(inputs: Tensor, *, rounds: int,
                                  target_bit: int = 0) -> Tensor:
    """Return bit0 after 6/8/10 public rounds with every round constant zero.

    This intentionally has no key schedule and is not a zero-master-key cipher.
    """
    if type(rounds) is not int:
        raise TypeError("rounds must be a built-in int")
    if rounds not in (6, 8, 10):
        raise ValueError("rounds must be 6, 8, or 10")
    if type(target_bit) is not int:
        raise TypeError("target_bit must be a built-in int")
    if target_bit != 0:
        raise ValueError("target_bit must be 0")
    bits = _binary_inputs(inputs)
    high, low = bits[:, :32], bits[:, 32:]
    for _ in range(rounds):
        high, low = low ^ _round_function(high), high
    return high[:, :1].to(dtype=torch.float32)


def zero_fixed_background(inputs: Tensor, *, plaintext_space: str,
                          fixed_value: int) -> Tensor:
    """Check the public fixed high word, then return a fresh FP32[zero,low].

    The protocol name high32-zero describes the fixed-half space; its public
    fixed_value can be nonzero. Unknown high bits are never silently discarded.
    """
    if type(plaintext_space) is not str:
        raise TypeError("plaintext_space must be a built-in str")
    if plaintext_space != "high32-zero":
        raise ValueError("background zeroing requires high32-zero")
    if type(fixed_value) is not int:
        raise TypeError("fixed_value must be a built-in int")
    if not 0 <= fixed_value <= 0xFFFFFFFF:
        raise ValueError("fixed_value must be uint32")
    bits = _binary_inputs(inputs)
    expected = torch.tensor(
        [(fixed_value >> shift) & 1 for shift in range(31, -1, -1)],
        dtype=torch.bool, device=inputs.device,
    )
    if not bool((bits[:, :32] == expected).all()):
        raise ValueError("input high32 does not match public fixed_value")
    return torch.cat((torch.zeros_like(bits[:, :32]), bits[:, 32:]), dim=1).to(
        dtype=torch.float32
    )

"""Batch-local public Boolean plan adapter. No network, labels or cipher calls."""
from __future__ import annotations

import torch
from torch import Tensor

from .simon_frontier import PublicFeaturePlan


def frontier_features(inputs: Tensor, plan: PublicFeaturePlan) -> Tensor:
    """CPU binary float32 [N,64] to [N,d], maskbit0 = rawcolumn0 (MSB).

    Evaluate each distinct public monomial and leaf once per batch. No full
    dataset cache, arithmetic matmul, random stream or differentiable network.
    Constant monomial mask0 and empty AND feature both mean Boolean one.
    """
    if type(plan) is not PublicFeaturePlan:
        raise ValueError('expected validated PublicFeaturePlan')
    if (type(inputs) is not Tensor or inputs.device.type != 'cpu'
            or inputs.layout != torch.strided or inputs.dtype != torch.float32
            or inputs.ndim != 2 or inputs.shape[1] != 64 or inputs.shape[0] < 1
            or not bool(((inputs == 0) | (inputs == 1)).all())):
        raise ValueError('expected binary CPU float32 [N,64]')
    bits = inputs.to(torch.bool)
    one = torch.ones(inputs.shape[0], dtype=torch.bool)
    monomials: dict[int, Tensor] = {}
    for polynomial in plan.leaves:
        for mask in polynomial:
            if mask not in monomials:
                value = one.clone()
                remainder = mask
                while remainder:
                    index = (remainder & -remainder).bit_length() - 1
                    value &= bits[:, index]
                    remainder &= remainder - 1
                monomials[mask] = value
    leaves = []
    for polynomial in plan.leaves:
        value = torch.zeros_like(one)
        for mask in polynomial:
            value ^= monomials[mask]
        leaves.append(value)
    output = torch.empty((inputs.shape[0], plan.dimension), dtype=torch.float32)
    for column, feature in enumerate(plan.features):
        value = one.clone()
        for index in feature:
            value &= leaves[index]
        output[:, column] = value
    return output

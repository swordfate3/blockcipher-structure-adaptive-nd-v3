"""Stable logits for the arithmetic mean of two Bernoulli probabilities."""
from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import logsigmoid

from ..prefix_bridge.data import Floats


def average_logits_torch(first: Tensor, second: Tensor) -> Tensor:
    """Average the probabilities while retaining gradients through both orders."""
    a, b = first.double(), second.double()
    pa, pb, qa, qb = logsigmoid(a), logsigmoid(b), logsigmoid(-a), logsigmoid(-b)
    logit = torch.logaddexp(pa, pb) - torch.logaddexp(qa, qb)
    # Near zero the subtraction above can erase the decision. If s = a+b >= 0,
    # p_a+p_b-1 = p_a*p_b*(1-exp(-s)); for s < 0 use the complementary class.
    # Recover that forward value while differentiating the two log class masses.
    with torch.no_grad():
        total = a + b
        magnitude = torch.exp(torch.where(total >= 0, pa + pb, qa + qb))
        magnitude = magnitude * -torch.expm1(-total.abs())
        near = logit.abs() < .25
        magnitude = torch.where(near, magnitude, 0.)
        tie_logit = 2 * torch.atanh(magnitude)
        floor = torch.finfo(first.dtype).tiny
        tie_logit = torch.where((tie_logit.to(first.dtype) == 0) & (total != 0),
                                floor, tie_logit)
        value = torch.where(near, total.sign() * tie_logit, logit)
    return (value + (logit - logit.detach())).to(first.dtype)


def average_logits(first: Floats, second: Floats) -> Floats:
    """NumPy scoring counterpart for finite float32 logits."""
    a, b = first.astype(np.float64), second.astype(np.float64)
    pa, pb = -np.logaddexp(0, -a), -np.logaddexp(0, -b)
    qa, qb = -np.logaddexp(0, a), -np.logaddexp(0, b)
    logit = np.logaddexp(pa, pb) - np.logaddexp(qa, qb)
    total = a + b
    near = np.abs(logit) < .25
    magnitude = np.exp(np.where(total >= 0, pa + pb, qa + qb)) * -np.expm1(-np.abs(total))
    tie_logit = 2 * np.arctanh(np.where(near, magnitude, 0.))
    tie_logit = np.where((tie_logit.astype(np.float32) == 0) & (total != 0),
                         np.finfo(np.float32).tiny, tie_logit)
    return cast(Floats, np.where(near, np.sign(total) * tie_logit, logit).astype(np.float32))

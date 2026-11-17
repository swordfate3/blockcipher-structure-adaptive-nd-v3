"""Versioned public Boolean coordinates; no cipher, labels or secret state."""
from __future__ import annotations

import torch
from torch import Tensor

FEATURE_ORDER = ('1', 'l', 'm', 'n', 'p', 'q', 's', 't', 'lm', 'mn', 'mq', 'pm',
                 'ln XOR lq XOR pn')


def simon_r3_residual13(inputs: Tensor) -> Tensor:
    if (type(inputs) is not Tensor or inputs.device.type != 'cpu'
            or inputs.layout != torch.strided or inputs.dtype != torch.float32
            or inputs.ndim != 2 or inputs.shape[1] != 64 or inputs.shape[0] < 1
            or not bool(((inputs == 0) | (inputs == 1)).all())):
        raise ValueError('expected binary CPU float32 [N,64]')
    bits = inputs.to(torch.bool)

    def x(j: int) -> Tensor:
        return bits[:, 31 - j % 32]

    def a(j: int) -> Tensor:
        return bits[:, 63 - j % 32] ^ (x(j-1) & x(j-8)) ^ x(j-2)

    l, m, n = a(29), a(22), a(15)  # noqa: E741 - published coordinate name
    s, t = a(28), a(21)
    p, q = x(30) ^ s, x(23) ^ t
    return torch.stack((torch.ones_like(l), l, m, n, p, q, s, t, l & m, m & n,
                        m & q, p & m, (l & n) ^ (l & q) ^ (p & n)), dim=1).float()

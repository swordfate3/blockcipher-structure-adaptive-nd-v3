"""Public-only Shared-M stems and suffix, with a dense per-pair bit head."""
from __future__ import annotations

import copy
from typing import cast

import torch
from torch import Tensor, nn

from cipher_lab.models.shared_m.components import GeometryStem, Residual
from cipher_lab.models.shared_m.router import PublicRouter, Route
from cipher_lab.models.shared_m.spec import STEM_NAMES, SharedMSpec
from cipher_lab.representations.tasks import public_features

from .config import BLOCK_BITS, WIDTH

_SPEC = SharedMSpec(96, 96, 8, (3, 5, 7, 9, 11), 64, 16)


class CubeSharedM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stems = nn.ModuleDict({name: GeometryStem(name, _SPEC) for name in STEM_NAMES})
        initial = copy.deepcopy(cast(GeometryStem, self.stems['local']).network.state_dict())
        for name in STEM_NAMES[1:]:
            cast(GeometryStem, self.stems[name]).network.load_state_dict(initial)
        self.register_buffer('public', torch.tensor(public_features('SIMON'), dtype=torch.float32))
        self.router = PublicRouter(_SPEC.router_hidden)
        self.suffix = nn.Sequential(
            *(Residual(k, _SPEC.channels, _SPEC.norm_groups) for k in _SPEC.suffix_kernels))
        features = _SPEC.channels * BLOCK_BITS
        self.base = nn.Linear(features, WIDTH)
        self.residual = nn.Sequential(nn.Linear(features, _SPEC.head_hidden), nn.ReLU(),
                                      nn.Linear(_SPEC.head_hidden, WIDTH))
        final = cast(nn.Linear, self.residual[-1])
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    public: Tensor

    def forward(self, inputs: Tensor) -> Tensor:
        if (not isinstance(inputs, Tensor) or inputs.layout != torch.strided or inputs.ndim != 2
                or inputs.shape[1] != WIDTH or not len(inputs) or len(inputs) % 4
                or inputs.dtype != torch.float32 or inputs.device != self.public.device
                or not bool(((inputs == 0) | (inputs == 1)).all())):
            raise ValueError(
                'expected binary same-device FP32 plaintext [N,128] with N divisible by 4')
        left, right = inputs.reshape(-1, 4, 2, BLOCK_BITS).unbind(dim=2)
        canonical = torch.stack((left, right, (left != right).to(inputs.dtype)), dim=2)
        relation = inputs.new_zeros((len(inputs) // 4, 4, 1, BLOCK_BITS))
        route = cast(Route, self.router(self.public))
        features = [cast(Tensor, self.stems[STEM_NAMES[index]](canonical, relation))
                    for index in route.indices.tolist()]
        mixed = route.weights[0] * features[0] + route.weights[1] * features[1]
        hidden = cast(Tensor, self.suffix(mixed.reshape(-1, _SPEC.channels, BLOCK_BITS)))
        rows = hidden.flatten(1)
        logits = cast(Tensor, self.base(rows)) + cast(Tensor, self.residual(rows))
        if logits.shape != (len(inputs), WIDTH) or not bool(logits.isfinite().all()):
            raise ValueError('invalid logits')
        return logits

"""Fixed shared-M predictor: MSB0 plaintext pairs to four per-pair relation logits.

Pairs are batch organization only. Geometry, normalization, suffix and readout
operate independently on each pair; only the public13 card controls routing.
"""
from __future__ import annotations

import copy
from typing import cast

import torch
from torch import Tensor, nn

from cipher_lab.models.shared_m.components import Residual
from cipher_lab.models.shared_m.router import PublicRouter, Route
from cipher_lab.models.shared_m.spec import STEM_NAMES, SharedMSpec
from cipher_lab.representations.neighborhoods import gather_neighborhood, neighborhood_indices
from cipher_lab.representations.tasks import validate_public


class _PairGeometryStem(nn.Module):
    indices: Tensor

    def __init__(self, kind: str, spec: SharedMSpec) -> None:
        super().__init__()
        self.register_buffer("indices", neighborhood_indices(kind))
        self.network = nn.Sequential(
            nn.Conv1d(27, spec.stem_width, 1),
            nn.GroupNorm(spec.norm_groups, spec.stem_width),
            nn.ReLU(),
            nn.Conv1d(spec.stem_width, spec.channels, 1),
            nn.GroupNorm(spec.norm_groups, spec.channels),
            nn.ReLU(),
        )

    def forward(self, raw: Tensor) -> Tensor:
        return cast(Tensor, self.network(gather_neighborhood(raw, self.indices)))


class _PairReadout(nn.Module):
    def __init__(self, spec: SharedMSpec) -> None:
        super().__init__()
        self.base = nn.Linear(spec.channels * 64, 4)
        self.residual = nn.Sequential(
            nn.Linear(spec.channels * 64, spec.head_hidden),
            nn.ReLU(),
            nn.Linear(spec.head_hidden, 4),
        )
        final = cast(nn.Linear, self.residual[2])
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        flat = hidden.flatten(1)
        return cast(Tensor, self.base(flat)) + cast(Tensor, self.residual(flat))


def _validate_pairs(pairs: Tensor) -> None:
    if (not isinstance(pairs, Tensor) or pairs.layout != torch.strided
            or pairs.is_complex()):
        raise ValueError("expected dense real binary pairs")
    if pairs.ndim != 4 or pairs.shape[1:] != (4, 2, 64) or pairs.shape[0] < 1:
        raise ValueError("expected pairs [B,4,2,64], exactly four 64-bit pairs")
    if not bool(torch.isfinite(pairs).all() & ((pairs == 0) | (pairs == 1)).all()):
        raise ValueError("pair bits must be finite and binary")


class SharedMRelations(nn.Module):
    """No-argument fixed architecture; returns logits [B,4,4], not probabilities."""

    def __init__(self) -> None:
        super().__init__()
        self.spec = SharedMSpec(96, 96, 8, (3, 5, 7, 9, 11), 64, 16)
        self.stems = nn.ModuleDict({
            name: _PairGeometryStem(name, self.spec) for name in STEM_NAMES
        })
        initial = copy.deepcopy(
            cast(_PairGeometryStem, self.stems["local"]).network.state_dict()
        )
        for name in STEM_NAMES[1:]:
            cast(_PairGeometryStem, self.stems[name]).network.load_state_dict(initial)
        self.router = PublicRouter(self.spec.router_hidden)
        self.suffix = nn.Sequential(*[
            Residual(kernel, self.spec.channels, self.spec.norm_groups)
            for kernel in self.spec.suffix_kernels
        ])
        self.readout = _PairReadout(self.spec)

    def forward(self, pairs: Tensor, public: Tensor) -> Tensor:
        _validate_pairs(pairs)
        weight = cast(nn.Conv1d, cast(_PairGeometryStem, self.stems["local"]).network[0]).weight
        if pairs.device != weight.device:
            raise ValueError("pairs and model must be on the same device")
        if isinstance(public, Tensor) and public.device != weight.device:
            raise ValueError("public card and model must be on the same device")
        validate_public(public)
        if public.dtype != weight.dtype:
            raise ValueError("public card and model must have same dtype")
        raw = torch.cat((pairs.to(weight.dtype),
                         (pairs[:, :, :1] != pairs[:, :, 1:]).to(weight.dtype)), dim=2)
        route = cast(Route, self.router(public))
        features = [cast(Tensor, self.stems[STEM_NAMES[index]](raw))
                    for index in route.indices.tolist()]
        mixed = route.weights[0] * features[0] + route.weights[1] * features[1]
        hidden = cast(Tensor, self.suffix(mixed))
        logits = cast(Tensor, self.readout(hidden))
        if logits.shape != (pairs.shape[0] * 4, 4) or not bool(torch.isfinite(logits).all()):
            raise ValueError("model produced nonfinite or invalid logits")
        return logits.reshape(pairs.shape[0], 4, 4)

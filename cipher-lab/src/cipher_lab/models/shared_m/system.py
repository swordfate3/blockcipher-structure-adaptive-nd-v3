"""Shared single-view model; public rounds affect representation, not routing."""
from __future__ import annotations

import copy
from typing import cast

import torch
from torch import Tensor, nn

from cipher_lab.models.shared_m.components import GeometryStem, Residual, SharedReadout
from cipher_lab.models.shared_m.router import PublicRouter, Route
from cipher_lab.models.shared_m.spec import STEM_NAMES, SharedMSpec
from cipher_lab.representations.input import InputRepresentation, RepresentedInput, validate_bits


class SharedM(nn.Module):
    def __init__(self, spec: SharedMSpec) -> None:
        super().__init__()
        if not isinstance(spec, SharedMSpec):
            raise ValueError("spec must be a Task1 SharedMSpec")
        self.spec = spec
        self.representation = InputRepresentation()
        self.stems = nn.ModuleDict({name: GeometryStem(name, spec) for name in STEM_NAMES})
        initial = copy.deepcopy(cast(GeometryStem, self.stems["local"]).network.state_dict())
        for name in STEM_NAMES[1:]:
            cast(GeometryStem, self.stems[name]).network.load_state_dict(initial)
        self.router = PublicRouter(spec.router_hidden)
        self.suffix = nn.Sequential(
            *[Residual(k, spec.channels, spec.norm_groups) for k in spec.suffix_kernels])
        self.readout = SharedReadout(spec)

    def forward(self, bits: Tensor, public: Tensor, *, rounds: int) -> Tensor:
        validate_bits(bits)
        weight = cast(nn.Conv1d, cast(GeometryStem, self.stems["local"]).network[0]).weight
        if bits.device != weight.device:
            raise ValueError("bits and model must be on the same device")
        represented = cast(RepresentedInput, self.representation(bits, public, rounds=rounds))
        route = cast(Route, self.router(public))
        features = [cast(Tensor, self.stems[STEM_NAMES[index]](
            represented.canonical.to(weight.dtype), represented.channel.to(weight.dtype)))
            for index in route.indices.tolist()]
        mixed = route.weights[0] * features[0] + route.weights[1] * features[1]
        hidden = cast(Tensor, self.suffix(mixed.reshape(-1, self.spec.channels, 64)))
        logits = cast(Tensor, self.readout(hidden.reshape(-1, 4, self.spec.channels, 64)))
        if logits.shape != (bits.shape[0], 1) or not bool(torch.isfinite(logits).all()):
            raise ValueError("model produced nonfinite or invalid logits")
        return logits

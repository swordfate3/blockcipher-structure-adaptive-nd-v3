"""The six-channel PRESENT Shared-M; no state synthesis or training dependencies."""
from __future__ import annotations

from hashlib import sha256
from typing import cast

import torch
from torch import Tensor, nn

from cipher_lab.models.shared_m.components import GeometryStem, Residual, SharedReadout
from cipher_lab.models.shared_m.router import PublicRouter, Route
from cipher_lab.models.shared_m.spec import STEM_NAMES, SharedMSpec
from cipher_lab.representations.neighborhoods import gather_neighborhood, gather_relation
from cipher_lab.representations.tasks import public_features

from . import config

_SPEC = SharedMSpec(config.CHANNELS, config.CHANNELS, config.NORM_GROUPS,
                    config.SUFFIX_KERNELS, config.HEAD_HIDDEN, config.ROUTER_HIDDEN)


class PrefixStem(GeometryStem):
    def __init__(self, name: str) -> None:
        super().__init__(name, _SPEC)
        self.network[0] = nn.Conv1d(config.STEM_INPUTS, _SPEC.stem_width, 1)

    def forward(self, raw: Tensor, channel: Tensor) -> Tensor:
        s2 = gather_neighborhood(raw[:, :, :3], self.indices)
        final = gather_neighborhood(raw[:, :, 3:], self.indices)
        base = self.network[0](torch.cat((s2, final), dim=1))
        added = self.relation_projection(gather_relation(channel, self.indices))
        return cast(Tensor, self.network[1:](base + added)).reshape(-1, 4, self.channels, 64)


class PrefixSharedM(nn.Module):
    public: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.stems = nn.ModuleDict({name: PrefixStem(name) for name in STEM_NAMES})
        initial = cast(PrefixStem, self.stems['local']).network.state_dict()
        for name in STEM_NAMES[1:]:
            cast(PrefixStem, self.stems[name]).network.load_state_dict(initial, strict=True)
        public = torch.tensor(public_features('PRESENT'), dtype=torch.float32)
        self.register_buffer('public', public)
        self.router = PublicRouter(_SPEC.router_hidden)
        self.suffix = nn.Sequential(*[Residual(k, _SPEC.channels, _SPEC.norm_groups)
                                      for k in _SPEC.suffix_kernels])
        self.readout = SharedReadout(_SPEC)

    def forward(self, bits: Tensor) -> Tensor:
        if (not isinstance(bits, Tensor) or bits.layout != torch.strided
                or bits.shape[1:] != (4, 6, 64) or bits.ndim != 4 or not bits.shape[0]
                or bits.dtype != torch.float32 or bits.device != self.public.device
                or not bool(((bits == 0) | (bits == 1)).all())):
            raise ValueError('INPUT_BITS')
        relation = bits.new_zeros((bits.shape[0], 4, 1, 64))
        route = cast(Route, self.router(self.public))
        features = [cast(Tensor, self.stems[STEM_NAMES[i]](bits, relation))
                    for i in route.indices.tolist()]
        mixed = route.weights[0] * features[0] + route.weights[1] * features[1]
        hidden = cast(Tensor, self.suffix(mixed.reshape(-1, _SPEC.channels, 64)))
        logits = cast(Tensor, self.readout(hidden.reshape(-1, 4, _SPEC.channels, 64)))
        if logits.shape != (bits.shape[0], 1) or not bool(torch.isfinite(logits).all()):
            raise ValueError('MODEL_LOGITS')
        return logits


def build(seed: int) -> PrefixSharedM:
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.default_generator.manual_seed(seed)
        return PrefixSharedM().float()


def initial_sha256(model: PrefixSharedM) -> str:
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().numpy()
        digest.update(f'{name}:{a.dtype.name}:{list(a.shape)}\n'.encode('ascii'))
        digest.update(a.astype(a.dtype.newbyteorder('<'), copy=False).tobytes())
    return digest.hexdigest()

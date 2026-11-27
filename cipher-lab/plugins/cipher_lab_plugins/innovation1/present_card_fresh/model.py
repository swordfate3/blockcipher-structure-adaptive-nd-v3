"""The original three-channel pair path with the pinned public nibble table."""
from __future__ import annotations

from hashlib import sha256
from typing import cast

import torch
from torch import Tensor, nn

from cipher_lab.models.shared_m.components import GeometryStem
from cipher_lab.models.shared_m.router import Route
from cipher_lab.models.shared_m.spec import STEM_NAMES
from cipher_lab.representations.tasks import public_features

from ..config import Recipe, Seeds
from ..four_single.models import SinglePairM
from ..four_single.models import geometry as pair_geometry
from ..layout_probe.models import require_fp32
from ..models import construct
from . import config
from .layout import NIBBLE_RULE, verified_card
from .present_public import PRESENT_P_LAYER


class CardM(SinglePairM):
    def _route(self, public: Tensor) -> Route:
        return Route(torch.tensor([STEM_NAMES.index(name) for name, _ in config.ROUTE],
                                  device=public.device, dtype=torch.long),
                     public.new_tensor([weight for _, weight in config.ROUTE]))


class Classifier(nn.Module):
    public: Tensor

    def __init__(self, pair: CardM) -> None:
        super().__init__()
        self.pair = pair
        public = torch.tensor(public_features('PRESENT'), dtype=torch.float32)
        self.register_buffer('public', public)

    def forward(self, bits: Tensor) -> Tensor:
        if (not isinstance(bits, Tensor) or bits.layout != torch.strided
                or bits.ndim != 4 or bits.shape[1:] != (1, 2, 64)
                or not len(bits) or len(bits) % 4 or bits.device != self.public.device
                or bits.dtype != torch.float32 or not bool(((bits == 0) | (bits == 1)).all())):
            raise ValueError('CARD_INPUT_BITS')
        return self.pair.forward(bits, self.public, rounds=config.ROUNDS[0])


def build() -> Classifier:
    require_fp32()
    with torch.device('cpu'):
        # Same scoped Seed/RNG/SharedM constructor as the reference builder.
        pair = CardM(construct(Recipe(seeds=Seeds(init=config.INIT_SEED)), 'M-best'))
        pair.shared.router.requires_grad_(False)
        indices = cast(GeometryStem, pair.shared.stems['nibble']).indices
        structure = verified_card(PRESENT_P_LAYER, NIBBLE_RULE,
                                  model_default=tuple(tuple(row) for row in indices.tolist()))
        indices.copy_(torch.tensor(structure, dtype=torch.long, device=indices.device))
        net = Classifier(pair)
    if sum(p.numel() for p in net.parameters()) != config.PARAMETERS:
        raise ValueError('MODEL_PARAMETERS')
    return net


def geometry(net: Classifier) -> dict[str, str]:
    return {**pair_geometry(net.pair), 'public': sha256(
        net.public.detach().cpu().numpy().astype('<f4').tobytes()).hexdigest()}

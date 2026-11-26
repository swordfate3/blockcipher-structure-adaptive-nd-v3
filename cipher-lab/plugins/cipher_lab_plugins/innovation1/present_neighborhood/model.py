"""Use the original single-pair feature operations and replace only one table."""
from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from cipher_lab.models.shared_m.components import GeometryStem
from cipher_lab.models.shared_m.router import Route
from cipher_lab.models.shared_m.spec import STEM_NAMES

from ..four_single.models import SinglePairM
from ..four_single.models import build as build_pair
from .config import ROUTE, SLOTS, NeighborhoodRecipe
from .layout import remap


class NeighborhoodM(SinglePairM):
    def _route(self, public: Tensor) -> Route:
        # Public card remains unchanged for the representation; neither the
        # value of the card nor any learned router parameters select the stems.
        return Route(torch.tensor([STEM_NAMES.index(name) for name, _ in ROUTE],
                                  device=public.device, dtype=torch.long),
                     public.new_tensor([weight for _, weight in ROUTE]))


def build(recipe: NeighborhoodRecipe, slot: str) -> SinglePairM:
    if slot not in SLOTS:
        raise ValueError('MODEL_SLOT')
    model = build_pair(recipe, NeighborhoodM)
    model.shared.router.requires_grad_(False)
    indices = cast(GeometryStem, model.shared.stems['nibble']).indices
    structure = remap(tuple(tuple(row) for row in indices.tolist()))
    if slot == SLOTS[1]:
        # Keep the persistent buffer registered at the same path and do not
        # permute the input, output or the word stem's neighborhoods.
        indices.copy_(torch.tensor(structure, dtype=torch.long, device=indices.device))
    return model

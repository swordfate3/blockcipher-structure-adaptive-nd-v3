"""Single-pair readout and a card-qualified, nonmutating gather override."""
from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from cipher_lab.models.shared_m import SharedM
from cipher_lab.models.shared_m.components import GeometryStem
from cipher_lab.models.shared_m.router import Route
from cipher_lab.models.shared_m.spec import STEM_NAMES
from cipher_lab.representations.neighborhoods import gather_neighborhood, gather_relation
from cipher_lab.representations.tasks import public_features, validate_public

from ..layout_probe.layout import nibble_indices
from ..layout_probe.models import require_fp32
from ..models import construct
from ..single_pair.models import computation_slots, shared_logits
from ..single_train.weights import decode_state

if TYPE_CHECKING:
    from .config import FourSingleRecipe

FIXED_BUFFERS = frozenset({'skinny_column', *(f'shared.stems.{s}.indices' for s in STEM_NAMES)})


class SinglePairM(nn.Module):
    skinny_column: Tensor

    def __init__(self, shared: nn.Module) -> None:
        super().__init__()
        self.shared = cast(SharedM, shared)
        self.register_buffer('skinny_column', nibble_indices('column'))

    def _column(self, raw: Tensor, channel: Tensor) -> Tensor:
        # Same trained GeometryStem parameters, but never assign to its row buffer.
        stem = cast(GeometryStem, self.shared.stems['nibble'])
        base = stem.network[0](gather_neighborhood(raw, self.skinny_column))
        added = stem.relation_projection(gather_relation(channel, self.skinny_column))
        return cast(Tensor, stem.network[1:](base + added)).reshape(-1, 4, stem.channels, 64)

    def _route(self, public: Tensor) -> Route:
        return cast(Route, self.shared.router(public))

    def forward(self, bits: Tensor, public: Tensor, *, rounds: int) -> Tensor:
        slots = computation_slots(bits)
        weight = next(self.shared.parameters())
        if (bits.device != weight.device or public.device != weight.device
                or bits.dtype != torch.float32 or public.dtype != torch.float32
                or weight.dtype != torch.float32):
            raise ValueError('MODEL_FP32_DEVICE')
        validate_public(public)
        skinny = torch.equal(public, public.new_tensor(public_features('SKINNY')))
        logits = cast(Tensor, shared_logits(self.shared, slots, public, rounds=rounds,
                                           nibble=self._column if skinny else None,
                                           route_for=self._route))
        if logits.shape != (bits.shape[0], 1) or not bool(torch.isfinite(logits).all()):
            raise ValueError('SINGLE_PAIR_LOGITS')
        return logits

    @torch.inference_mode()
    def route_weights(self, public: Tensor) -> dict[str, float]:
        route = self._route(public)
        return {STEM_NAMES[i]: float(w) for i, w in zip(route.indices.tolist(),
                                                       route.weights.tolist(), strict=True)}


def build(recipe: FourSingleRecipe) -> SinglePairM:
    from .config import PARAMETERS, SLOT

    require_fp32()
    with torch.device('cpu'):
        model = SinglePairM(construct(recipe.streams, SLOT))
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ValueError('MODEL_PARAMETERS')
    return model


def geometry(model: SinglePairM) -> dict[str, str]:
    buffers = dict(model.named_buffers())
    if set(buffers) != FIXED_BUFFERS:
        raise ValueError('MODEL_BUFFERS')
    return {name: sha256(value.detach().cpu().numpy().astype('<i8').tobytes()).hexdigest()
            for name, value in buffers.items()}


def load_state(model: SinglePairM, raw: bytes, expected: dict[str, object]) -> None:
    if expected.get('geometry') != geometry(model):
        raise ValueError('CHECKPOINT_INVALID')
    state = decode_state(raw, lambda: expected, model.state_dict(), FIXED_BUFFERS)
    model.load_state_dict(state, strict=True)

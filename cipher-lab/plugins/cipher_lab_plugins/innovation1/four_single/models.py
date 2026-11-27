"""Single-pair readout and a card-qualified, nonmutating gather override."""
from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from typing import cast

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
from .config import PARAMETERS, SLOT, FourSingleRecipe

FIXED_BUFFERS = frozenset({'skinny_column', *(f'shared.stems.{s}.indices' for s in STEM_NAMES)})


class SinglePairM(nn.Module):
    skinny_column: Tensor

    def __init__(self, shared: nn.Module) -> None:
        super().__init__()
        self.shared = cast(SharedM, shared)
        self.register_buffer('skinny_column', nibble_indices('column'))

    def _column(self, raw: Tensor, channel: Tensor) -> Tensor:
        return self._gather(raw, channel, self.skinny_column)

    def _gather(self, raw: Tensor, channel: Tensor, table: Tensor) -> Tensor:
        # Use the shared stem with a task's fixed indices without rewriting the stem buffer.
        stem = cast(GeometryStem, self.shared.stems['nibble'])
        base = stem.network[0](gather_neighborhood(raw, table))
        added = stem.relation_projection(gather_relation(channel, table))
        return cast(Tensor, stem.network[1:](base + added)).reshape(-1, 4, stem.channels, 64)

    def _nibble(self, public: Tensor) -> Callable[[Tensor, Tensor], Tensor] | None:
        skinny = torch.equal(public, public.new_tensor(public_features('SKINNY')))
        return self._column if skinny else None

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
        logits = cast(Tensor, shared_logits(self.shared, slots, public, rounds=rounds,
                                           nibble=self._nibble(public),
                                           route_for=self._route))
        if logits.shape != (bits.shape[0], 1) or not bool(torch.isfinite(logits).all()):
            raise ValueError('SINGLE_PAIR_LOGITS')
        return logits

    @torch.inference_mode()
    def route_weights(self, public: Tensor) -> dict[str, float]:
        route = self._route(public)
        return {STEM_NAMES[i]: float(w) for i, w in zip(route.indices.tolist(),
                                                       route.weights.tolist(), strict=True)}


def build(recipe: FourSingleRecipe, cls: type[SinglePairM] = SinglePairM) -> SinglePairM:
    require_fp32()
    with torch.device('cpu'):
        model = cls(construct(recipe.streams, SLOT))
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ValueError('MODEL_PARAMETERS')
    return model


def _fixed_names(model: SinglePairM) -> frozenset[str]:
    extra: frozenset[str] = getattr(type(model), 'extra_fixed_buffers', frozenset())
    return FIXED_BUFFERS | extra


def geometry(model: SinglePairM) -> dict[str, str]:
    buffers = dict(model.named_buffers())
    if set(buffers) != _fixed_names(model):
        raise ValueError('MODEL_BUFFERS')
    return {name: sha256(value.detach().cpu().numpy().astype('<i8').tobytes()).hexdigest()
            for name, value in buffers.items()}


def load_state(model: SinglePairM, raw: bytes, expected: dict[str, object]) -> None:
    if expected.get('geometry') != geometry(model):
        raise ValueError('CHECKPOINT_INVALID')
    state = decode_state(raw, lambda: expected, model.state_dict(), _fixed_names(model))
    model.load_state_dict(state, strict=True)

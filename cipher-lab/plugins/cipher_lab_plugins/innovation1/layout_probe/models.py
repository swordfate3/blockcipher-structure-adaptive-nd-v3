"""Fresh fixed-route scan controls using innovation1's existing pair operators."""
from __future__ import annotations

from copy import deepcopy
from typing import cast

import torch
from torch import Tensor, nn

from cipher_lab.models.shared_m import SharedM
from cipher_lab.models.shared_m.components import GeometryStem
from cipher_lab.models.shared_m.router import Route
from cipher_lab.models.shared_m.spec import STEM_NAMES
from cipher_lab.representations.tasks import public_features, validate_public

from .. import models as production
from ..config import Recipe, Seeds
from ..single_pair.models import computation_slots, shared_logits
from .layout import nibble_indices


def _validate_skinny(public: Tensor) -> None:
    validate_public(public)
    if not torch.equal(public, public.new_tensor(public_features('SKINNY'))):
        raise ValueError('layout probe requires the SKINNY public card')


class FixedSkinnyRouter(nn.Module):
    """Parameterless, with no persistent state and no task selection by score."""

    def forward(self, public: Tensor) -> Route:
        _validate_skinny(public)
        return Route(public.new_tensor([STEM_NAMES.index('nibble'), STEM_NAMES.index('word')],
                                       dtype=torch.long),
                     public.new_full((2,), 0.5))


class LayoutProbe(nn.Module):
    """Own a fresh M's components, but score one logit per pair, never per four-slot group."""

    def __init__(self, shared: SharedM) -> None:
        super().__init__()
        self.shared = shared
        self.shared.add_module('router', FixedSkinnyRouter())

    def forward(self, bits: Tensor, public: Tensor, *, rounds: int) -> Tensor:
        """Score [N,1,2,64] (positive N divisible by four) with an FP32 public card."""
        slots = computation_slots(bits)
        weight = next(self.shared.parameters())
        if bits.device != weight.device or public.device != weight.device:
            raise ValueError('bits, public card and model must be on the same device')
        # The reused pair path consumes the representation's FP32 tensors directly.
        if public.dtype != torch.float32 or weight.dtype != torch.float32:
            raise ValueError('layout probe requires FP32 public and model dtype')
        _validate_skinny(public)
        logits = cast(Tensor, shared_logits(self.shared, slots, public, rounds=rounds))
        if logits.shape != (bits.shape[0], 1) or not bool(torch.isfinite(logits).all()):
            raise ValueError('layout probe produced nonfinite or invalid pair logits')
        return logits


def require_fp32() -> None:
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('layout probe requires FP32 default dtype before construction')


def build_pair(*, init_seed: int) -> tuple[LayoutProbe, LayoutProbe]:
    """Initialize once on CPU; return independent (row, column) models, never restored."""
    if type(init_seed) is not int or not 0 <= init_seed < 2**63:
        raise ValueError('init_seed must be a public nonnegative 63-bit integer')
    require_fp32()
    with torch.device('cpu'):
        shared = cast(SharedM, production.construct(Recipe(seeds=Seeds(init=init_seed)), 'M-best'))
        row = LayoutProbe(shared)
        column = deepcopy(row)
        cast(GeometryStem, column.shared.stems['nibble']).indices.copy_(nibble_indices('column'))
    return row, column

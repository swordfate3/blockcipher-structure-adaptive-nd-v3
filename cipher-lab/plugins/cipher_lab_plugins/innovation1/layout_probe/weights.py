"""Layout-bound checkpoint identity, reusing the single-pair bounded decoder."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor

from ..single_train.config import SingleRecipe
from ..single_train.weights import bound_identity, decode_state
from .config import FIXED, METHOD, RUN_ID, SLOTS, LayoutRecipe

FIXED_BUFFERS = frozenset(f'shared.stems.{stem}.indices' for stem in ('local', 'word', 'nibble'))


def identity(recipe: SingleRecipe, source: dict[str, str], backend: str,
             slot: str, epoch: int, score: float) -> dict[str, Any]:
    if type(recipe) is not LayoutRecipe:
        raise ValueError('CHECKPOINT_IDENTITY')
    return bound_identity(recipe, source, backend, slot, epoch, score, slots=SLOTS,
                          fixed=FIXED, schema=2, method=METHOD, run_id=RUN_ID)


def decode(raw: bytes, recipe: SingleRecipe, source: dict[str, str], backend: str,
           slot: str, epoch: int, score: float,
           template: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return decode_state(raw, lambda: identity(recipe, source, backend, slot, epoch, score),
                        template, FIXED_BUFFERS)

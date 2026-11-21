"""Layout-bound checkpoint identity, reusing the single-pair bounded decoder."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor

from ..config import validate_run_id
from ..single_train.config import BACKEND, SingleRecipe
from ..single_train.weights import bound_identity, decode_state
from .config import LAYOUT_NAMESPACES, SLOTS, LayoutRecipe, official

FIXED_BUFFERS = frozenset(f'shared.stems.{stem}.indices' for stem in ('local', 'word', 'nibble'))


def identity(recipe: SingleRecipe, source: dict[str, str], backend: str,
             slot: str, epoch: int, score: float) -> dict[str, Any]:
    if (type(recipe) is not LayoutRecipe or recipe.run_id is None
            or recipe.namespace != LAYOUT_NAMESPACES.get(recipe.method)):
        raise ValueError('CHECKPOINT_IDENTITY')
    validate_run_id(recipe.run_id)
    fixed = official(recipe, recipe.run_id) if backend == BACKEND else recipe
    metadata = bound_identity(recipe, source, backend, slot, epoch, score, slots=SLOTS,
                              fixed=fixed, schema=recipe.artifact_schema,
                              method=recipe.method, run_id=recipe.run_id)
    if recipe.loss_key == 'mse':
        if score > 1.:
            raise ValueError('CHECKPOINT_IDENTITY')
        metadata['mse'] = metadata.pop('score')
    return metadata


def decode(raw: bytes, recipe: SingleRecipe, source: dict[str, str], backend: str,
           slot: str, epoch: int, score: float,
           template: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return decode_state(raw, lambda: identity(recipe, source, backend, slot, epoch, score),
                        template, FIXED_BUFFERS)

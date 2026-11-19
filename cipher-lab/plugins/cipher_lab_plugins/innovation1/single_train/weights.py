"""New checkpoint domain: bounded archive, exact new identity, strict CPU state."""
from __future__ import annotations

import io
import math
import re
from collections.abc import Callable, Collection, Mapping
from typing import Any

import torch

from ..checkpoint import preflight
from ..contracts import MAX_CHECKPOINT
from .config import BACKEND, FIXED, METHOD, RUN_ID, SLOTS, SingleRecipe, canonical

_FIXED_M = ('stems.local.indices', 'stems.word.indices', 'stems.nibble.indices')


def identity(recipe: SingleRecipe, source: dict[str, str], backend: str,
             slot: str, epoch: int, score: float) -> dict[str, Any]:
    return bound_identity(recipe, source, backend, slot, epoch, score, slots=SLOTS,
                          fixed=FIXED, schema=1, method=METHOD, run_id=RUN_ID)


def bound_identity(recipe: SingleRecipe, source: dict[str, str], backend: str,
                   slot: str, epoch: int, score: float, *, slots: tuple[str, str],
                   fixed: SingleRecipe, schema: int, method: str, run_id: str) -> dict[str, Any]:
    if (slot not in slots or type(epoch) is not int
            or not 1 <= epoch <= recipe.training.epochs
            or type(score) not in (int, float) or not math.isfinite(score) or score < 0
            or backend not in (BACKEND, 'artificial') or (backend == BACKEND and recipe != fixed)
            or not source or any(type(k) is not str or type(v) is not str
                                 or not re.fullmatch('[a-f0-9]{64}', v)
                                 for k, v in source.items())):
        raise ValueError('CHECKPOINT_IDENTITY')
    return dict(schema=schema, method=method, run_id=run_id, recipe=recipe.to_dict(),
                source=dict(source), backend=backend, slot=slot, epoch=epoch, score=float(score))


def decode(raw: bytes, recipe: SingleRecipe, source: dict[str, str], backend: str,
           slot: str, epoch: int, score: float,
           template: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return decode_state(raw, lambda: identity(recipe, source, backend, slot, epoch, score),
                        template, _FIXED_M if slot == SLOTS[0] else ())


def decode_state(raw: bytes, expected: Callable[[], dict[str, Any]],
                 template: Mapping[str, torch.Tensor],
                 fixed_names: Collection[str]) -> dict[str, torch.Tensor]:
    preflight(raw, MAX_CHECKPOINT)
    metadata = expected()
    try:
        value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if (type(value) is not dict or set(value) != {'metadata', 'state'}
                or canonical(value['metadata']) != canonical(metadata)
                or type(value['state']) is not dict or set(value['state']) != set(template)):
            raise ValueError
        state = value['state']
        for name, reference in template.items():
            tensor = state[name]
            if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu'
                    or tensor.layout != torch.strided or tensor.shape != reference.shape
                    or tensor.dtype != reference.dtype or not bool(torch.isfinite(tensor).all())):
                raise ValueError
        if any(
                name not in template or not torch.equal(state[name], template[name])
                for name in fixed_names):
            raise ValueError
        return dict(state)
    except Exception:
        raise ValueError('CHECKPOINT_INVALID') from None

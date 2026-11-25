"""Checkpoint domain for the masked PRESENT models, not the old five slots."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from torch import Tensor

from cipher_lab.artifacts import _source, _token

from ..checkpoint import canonical, preflight
from ..single_train.weights import decode_state
from . import config


def identity(slot: str, source: Mapping[str, str], run_id: str,
             epoch: int, bce: float) -> dict[str, Any]:
    r, mask = config.slot_input(slot)
    _source(dict(source))
    _token(run_id)
    if (type(epoch) is not int or not 1 <= epoch <= config.EPOCHS
            or type(bce) not in (float, int) or not math.isfinite(bce) or bce < 0):
        raise ValueError('CHECKPOINT_IDENTITY')
    return dict(schema=1, method=config.METHOD, slot=slot, round=r, mask=mask,
                s2_boundary=config.S2_BOUNDARY, recipe=config.recipe(), source=dict(source),
                run_id=run_id, epoch=epoch, bce=float(bce))


def require_identity(slot: str, metadata: dict[str, Any]) -> None:
    """Accept no extra checkpoint metadata (including hidden panel arrays)."""
    try:
        expected = identity(slot, metadata['source'], metadata['run_id'],
                            metadata['epoch'], metadata['bce'])
        if canonical(expected) != canonical(metadata):
            raise ValueError
    except (TypeError, KeyError, ValueError, OverflowError):
        raise ValueError('CHECKPOINT_IDENTITY') from None


def decode(raw: bytes, metadata: dict[str, Any], template: Mapping[str, Tensor],
           fixed: tuple[str, ...]) -> dict[str, Tensor]:
    preflight(raw, config.CHECKPOINT_BYTES)
    return decode_state(raw, lambda: metadata, template, fixed)

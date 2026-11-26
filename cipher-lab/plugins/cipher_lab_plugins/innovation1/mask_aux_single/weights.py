"""Pair-specific checkpoint identity; strict state validation uses the existing codec."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from cipher_lab.artifacts import _source, _token

from ..checkpoint import canonical
from . import config


def identity(slot: str, source: Mapping[str, str], run_id: str,
             epoch: int, bce: float) -> dict[str, Any]:
    r, mask = config.slot_input(slot)
    _source(dict(source))
    _token(run_id)
    if (type(epoch) is not int or not 1 <= epoch <= config.EPOCHS
            or type(bce) not in (int, float) or not math.isfinite(bce) or bce < 0):
        raise ValueError('CHECKPOINT_IDENTITY')
    return dict(schema=1, method=config.METHOD, slot=slot, round=r, mask=mask,
                layout=('pair-classifier+position-aux' if slot == config.AUXILIARY
                        else 'pair-classifier'),
                recipe=config.recipe(), source=dict(source), run_id=run_id, epoch=epoch,
                step=epoch * config.steps_per_epoch(), bce=float(bce))


def require_identity(slot: str, metadata: dict[str, Any]) -> None:
    try:
        expected = identity(slot, metadata['source'], metadata['run_id'],
                            metadata['epoch'], metadata['bce'])
        if canonical(expected) != canonical(metadata):
            raise ValueError
    except (TypeError, KeyError, ValueError, OverflowError):
        raise ValueError('CHECKPOINT_IDENTITY') from None

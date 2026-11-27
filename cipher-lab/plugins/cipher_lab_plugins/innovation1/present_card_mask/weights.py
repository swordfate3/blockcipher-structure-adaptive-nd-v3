"""Bind checkpoints to the distinct masked-card recipe and complete state."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..checkpoint import canonical
from ..present_card_fresh import config as pure_config
from ..present_card_fresh import weights as pure
from . import config


def identity(slot: str, source: Mapping[str, str], run_id: str, epoch: int, bce: float,
             initial: str, geometry: Mapping[str, str]) -> dict[str, Any]:
    if slot != config.SLOT:
        raise ValueError('CHECKPOINT_IDENTITY')
    base = pure.identity(pure_config.SLOT, source, run_id, epoch, bce, initial, geometry)
    return {**base, 'method': config.METHOD, 'recipe': config.recipe(), 'slot': slot}


def require_identity(slot: str, metadata: dict[str, Any]) -> None:
    try:
        expected = identity(slot, metadata['source'], metadata['run_id'], metadata['epoch'],
                            metadata['bce'], metadata['initial_sha256'], metadata['geometry'])
        if canonical(expected) != canonical(metadata):
            raise ValueError
    except (AttributeError, TypeError, KeyError, ValueError, OverflowError):
        raise ValueError('CHECKPOINT_IDENTITY') from None

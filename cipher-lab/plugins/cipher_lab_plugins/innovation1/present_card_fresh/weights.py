"""Exact fresh-card checkpoint metadata; the bridge owns the bounded state codec."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from cipher_lab.artifacts import _source, _token

from ..checkpoint import canonical
from . import config
from .layout import STRUCTURE_SHA256

_GEOMETRY = {'public', 'skinny_column', *(f'shared.stems.{s}.indices'
                                       for s in ('local', 'word', 'nibble'))}


def identity(slot: str, source: Mapping[str, str], run_id: str, epoch: int, bce: float,
             initial: str, geometry: Mapping[str, str]) -> dict[str, Any]:
    source = dict(source)
    _source(source)
    _token(run_id)
    if (slot != config.SLOT or type(bce) not in (int, float) or not math.isfinite(bce) or bce < 0
            or re.fullmatch('[0-9a-f]{64}', initial) is None or geometry.keys() != _GEOMETRY
            or geometry['shared.stems.nibble.indices'] != STRUCTURE_SHA256
            or any(re.fullmatch('[0-9a-f]{64}', digest) is None for digest in geometry.values())):
        raise ValueError('CHECKPOINT_IDENTITY')
    return dict(schema=1, method=config.METHOD, slot=slot, round=config.ROUNDS[0],
                recipe=config.recipe(), source=source, run_id=run_id, epoch=epoch,
                step=config._steps(epoch), bce=float(bce), initial_sha256=initial,
                geometry=dict(geometry))


def require_identity(slot: str, metadata: dict[str, Any]) -> None:
    try:
        expected = identity(slot, metadata['source'], metadata['run_id'], metadata['epoch'],
                            metadata['bce'], metadata['initial_sha256'], metadata['geometry'])
        if canonical(expected) != canonical(metadata):
            raise ValueError
    except (AttributeError, TypeError, KeyError, ValueError, OverflowError):
        raise ValueError('CHECKPOINT_IDENTITY') from None

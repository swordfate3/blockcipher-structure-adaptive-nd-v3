"""Endpoint order transforms for PRESENT card batches."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..prefix_bridge.data import Floats


def swap_all(inputs: Floats) -> Floats:
    """Return an owned contiguous endpoint reversal for DEV diagnostics."""
    return inputs[:, :, ::-1, :].copy(order='C')


def swap_half(inputs: Floats, *, seed: int, step: int) -> Floats:
    """Sample a fresh independent endpoint-order decision for each training row."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, step])))
    swap = rng.random(len(inputs)) < 0.5
    result = inputs.copy(order='C')
    result[swap] = inputs[swap, :, ::-1, :]
    return result

"""Keep side draws aligned while changing the hidden coordinates."""
from __future__ import annotations

import numpy as np

from ..mask_aux.masking import masked_view
from ..prefix_bridge.data import Floats, Indices
from ..present_card_mask import config as parent
from ..present_card_mask.masking import prepare as random_view
from . import config


def prepare(inputs: Floats, step: int, slot: str) -> tuple[Floats, Indices, Floats]:
    if slot == config.RANDOM:
        return random_view(inputs, step)
    if slot not in (config.FIRST, config.FOURTH):
        raise ValueError('INVALID_MASK_SLOT')
    if (type(step) is not int or step < 1 or inputs.ndim != 4
            or inputs.shape[1:] != (1, 2, 64) or inputs.dtype != np.float32
            or not len(inputs) or len(inputs) % 4 or not np.all((inputs == 0) | (inputs == 1))):
        raise ValueError('INVALID_MASK_INPUT')
    rng = np.random.default_rng(np.random.SeedSequence([parent.MASK_SEED, step]))
    # Advance past the location draw so the side draws match the random arm.
    rng.permuted(np.broadcast_to(np.arange(64), (len(inputs), 64)), axis=1)
    sides = rng.integers(0, 2, size=(len(inputs), parent.MASK_SITES))
    sites = np.arange(0 if slot == config.FIRST else 3, 64, 4)
    mask = np.zeros((len(inputs), 2, 64), dtype=np.bool_)
    mask[np.arange(len(inputs))[:, None], sides, sites] = True
    pairs = inputs.reshape(-1, 4, 2, 64)
    full = np.zeros((*pairs.shape[:2], 6, 64), dtype=np.float32)
    full[:, :, 3:5] = pairs
    full[:, :, 5] = pairs[:, :, 0] != pairs[:, :, 1]
    view, index, targets = masked_view(full, mask.reshape(pairs.shape))
    return view.reshape(-1, 1, 6, 64), index, targets

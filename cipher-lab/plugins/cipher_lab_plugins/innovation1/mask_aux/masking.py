"""Independent, public endpoint masking; targets are returned separately from inputs."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..prefix_bridge.data import Floats, Indices
from . import config

Mask = NDArray[np.bool_]


def mask_for_step(groups: int, step: int) -> Mask:
    if type(groups) is not int or groups <= 0 or type(step) is not int or step <= 0:
        raise ValueError('INVALID_MASK_STEP')
    rng = np.random.default_rng(np.random.SeedSequence([config.MASK_SEED, step]))
    # Independently permute the 64 sites in each pair. Endpoint draws do not
    # depend on any values (including the group class).
    pairs = groups * 4
    sites = rng.permuted(np.broadcast_to(np.arange(64), (pairs, 64)), axis=1)
    endpoints = rng.integers(0, 2, size=(pairs, config.MASK_SITES))
    mask = np.zeros((pairs, 2, 64), dtype=np.bool_)
    mask[np.arange(pairs)[:, None], endpoints, sites[:, :config.MASK_SITES]] = True
    return mask.reshape(groups, 4, 2, 64)


def masked_view(full: Floats, mask: Mask) -> tuple[Floats, Indices, Floats]:
    if (full.ndim != 4 or full.shape[1:] != (4, 6, 64) or full.dtype != np.float32
            or not len(full) or np.any(full[:, :, :3]) or mask.dtype != np.bool_
            or mask.shape != (len(full), 4, 2, 64)
            or np.any(mask.sum(axis=2) > 1)
            or not np.all(mask.sum(axis=(2, 3)) == config.MASK_SITES)):
        raise ValueError('INVALID_MASK_VIEW')
    xor_mask = mask.any(axis=2)
    targets = full[:, :, 3:5][mask]
    masked = full.copy()
    masked[:, :, :2] = mask
    masked[:, :, 2] = xor_mask
    masked[:, :, 3:5][mask] = 0
    masked[:, :, 5][xor_mask] = 0
    return masked, np.flatnonzero(mask), targets

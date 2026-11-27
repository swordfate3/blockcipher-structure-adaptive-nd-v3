"""Build a masked auxiliary view from complete endpoint rows."""
from __future__ import annotations

import numpy as np

from ..mask_aux.masking import mask_for_step, masked_view
from ..prefix_bridge.data import Floats, Indices


def prepare(inputs: Floats, step: int) -> tuple[Floats, Indices, Floats]:
    if (inputs.ndim != 4 or inputs.shape[1:] != (1, 2, 64) or inputs.dtype != np.float32
            or not len(inputs) or len(inputs) % 4 or not np.all((inputs == 0) | (inputs == 1))):
        raise ValueError('INVALID_MASK_INPUT')
    pairs = inputs.reshape(-1, 4, 2, 64)
    full = np.zeros((*pairs.shape[:2], 6, 64), dtype=np.float32)
    full[:, :, 3:5] = pairs
    full[:, :, 5] = pairs[:, :, 0] != pairs[:, :, 1]
    view, indices, targets = masked_view(full, mask_for_step(len(pairs), step))
    return view.reshape(-1, 1, 6, 64), indices, targets

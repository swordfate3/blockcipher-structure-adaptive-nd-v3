"""Swap sensitivity on DEV using a restored classifier and unchanged labels."""
from __future__ import annotations

import numpy as np

from cipher_lab.execution import Meter
from cipher_lab_plugins.direct_block.fit import BatchPanel, Engine, batch_indices, predict_batch

from ..prefix_bridge.metrics import score
from .views import swap_all


def evaluate(engine: Engine, dev: BatchPanel, batch_size: int, meter: Meter) -> dict[str, object]:
    labels = np.empty((len(dev), 1), dtype=np.uint8)
    original = np.empty(labels.shape, dtype=np.float32)
    swapped = np.empty_like(original)
    for at in batch_indices(len(dev), batch_size):
        x, y = dev.batch(at)
        original[at] = predict_batch(engine, x, y, meter)
        swapped[at] = predict_batch(engine, swap_all(x), y, meter)
        labels[at] = y
    first, second = score(original, labels), score(swapped, labels)
    flips = int(np.count_nonzero((original >= 0) != (swapped >= 0)))
    return {'pairs': len(labels), 'flip_count': flips, 'flip_fraction': flips / len(labels),
            'mean_abs_logit_delta': float(np.abs(original.astype(np.float64) - swapped).mean()),
            'original': {key: first.saved[key] for key in ('accuracy', 'bce')},
            'swapped': {key: second.saved[key] for key in ('accuracy', 'bce')}}

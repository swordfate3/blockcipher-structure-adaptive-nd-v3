"""Score both TEST views in one pass over the shared panel."""
from __future__ import annotations

import numpy as np

from cipher_lab.execution import Meter
from cipher_lab_plugins.direct_block.fit import (
    BatchPanel,
    Engine,
    Labels,
    batch_indices,
    predict_batch,
)

from ..prefix_bridge.metrics import Score, score
from ..present_card_swap.views import swap_all
from .probability import average_logits


def evaluate(engine: Engine, panel: BatchPanel, batch_size: int,
             meter: Meter) -> tuple[Score, Score, Labels]:
    labels = np.empty((len(panel), 1), dtype=np.uint8)
    original = np.empty(labels.shape, dtype=np.float32)
    dual = np.empty_like(original)
    for at in batch_indices(len(panel), batch_size):
        inputs, y = panel.batch(at)
        if y.shape != (len(at), 1) or y.dtype != np.uint8 or (y > 1).any():
            raise ValueError('INVALID_TEST_GROUPS')
        first = predict_batch(engine, inputs, y, meter)
        original[at] = first
        second = predict_batch(engine, swap_all(inputs), y, meter)
        dual[at] = average_logits(original[at], second)
        labels[at] = y
    return score(original, labels), score(dual, labels), labels

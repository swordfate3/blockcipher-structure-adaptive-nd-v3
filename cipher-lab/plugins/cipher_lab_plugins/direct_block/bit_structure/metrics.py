"""Binary prediction metrics for aligned single-column arrays."""
from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .data import Bytes, Floats, _check_labels

Correct = NDArray[np.bool_]
Summary = dict[str, int | float]


def train_constant(labels: Bytes) -> float:
    _check_labels(labels, len(labels))
    return (int(np.count_nonzero(labels)) + 0.5) / (len(labels) + 1)


def _summary(correct: int, rows: int, bce: float) -> Summary:
    return {'accuracy': correct / rows, 'bce': bce, 'correct': correct, 'rows': rows}


def score(logits: Floats, labels: Bytes) -> tuple[Summary, Correct]:
    rows = len(labels)
    _check_labels(labels, rows)
    if logits.dtype != np.float32 or logits.shape != labels.shape or not np.isfinite(logits).all():
        raise ValueError('invalid logits')
    logits64 = logits.astype(np.float64)
    signed = np.where(labels == 1, -logits64, logits64)
    bce = float(np.logaddexp(0.0, signed).mean())
    good = cast(Correct, (logits > 0) == labels)
    return _summary(int(np.count_nonzero(good)), rows, bce), good


def constant_score(probability: float, labels: Bytes) -> Summary:
    rows = len(labels)
    _check_labels(labels, rows)
    if not 0 < probability < 1:
        raise ValueError('constant probability must be in (0, 1)')
    ones = int(np.count_nonzero(labels))
    zeros = rows - ones
    bce = -(ones * math.log(probability) + zeros * math.log1p(-probability)) / rows
    result = _summary(ones if probability > 0.5 else zeros, rows, bce)
    result['probability'] = probability
    return result


def paired(left_correct: Correct, right_correct: Correct) -> dict[str, int]:
    if (left_correct.dtype != np.bool_ or right_correct.dtype != np.bool_
            or left_correct.shape != right_correct.shape):
        raise ValueError('invalid paired correctness arrays')
    both = int(np.count_nonzero(left_correct & right_correct))
    left_only = int(np.count_nonzero(left_correct)) - both
    right_only = int(np.count_nonzero(right_correct)) - both
    return {'both_correct': both, 'left_only': left_only, 'right_only': right_only,
            'both_wrong': left_correct.size - both - left_only - right_only}

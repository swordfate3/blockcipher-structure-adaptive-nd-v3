"""Full-word bit accuracy, constant baseline and frozen round decisions."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from . import config
from .data import Bytes, Floats, check_labels


def _score(matches: NDArray[np.bool_], bce: float) -> dict[str, object]:
    per_bit = [int(value) for value in matches.sum(axis=0)]
    rows = len(matches)
    high, low = sum(per_bit[:16]), sum(per_bit[16:])
    accuracy = (high + low) / (32 * rows)
    return {'rows': rows, 'correct_bits': high + low, 'accuracy': accuracy, 'bce': bce,
            'high_correct': high, 'low_correct': low, 'high_accuracy': high / (16 * rows),
            'low_accuracy': low / (16 * rows), 'per_bit_correct': per_bit,
            'ci_low': max(0.0, accuracy - config.RADIUS),
            'ci_high': min(1.0, accuracy + config.RADIUS)}


def score(logits: Floats, labels: Bytes) -> dict[str, object]:
    check_labels(labels, len(labels))
    if logits.dtype != np.float32 or logits.shape != labels.shape or not np.isfinite(logits).all():
        raise ValueError('invalid full-word logits')
    signed = np.where(labels == 0, logits, -logits).astype(np.float64)
    return _score((logits > 0) == labels, float(np.logaddexp(0.0, signed).mean()))


def constant_score(train_labels: Bytes, test_labels: Bytes) -> dict[str, object]:
    check_labels(train_labels, len(train_labels))
    check_labels(test_labels, len(test_labels))
    positives = train_labels.sum(axis=0, dtype=np.int64)
    probability = (positives + 0.5) / (len(train_labels) + 1)
    losses = np.where(test_labels == 1, -np.log(probability), -np.log1p(-probability))
    result = _score((positives > len(train_labels) / 2) == test_labels, float(losses.mean()))
    del result['ci_low'], result['ci_high']
    return result


def classify(depth: int, scores: Mapping[str, object]) -> str:
    if (type(depth) is not int or depth not in config.ROUNDS
            or scores.keys() != set(config.PANEL_SLOTS)):
        raise ValueError('invalid round or score slots')
    intervals: dict[str, tuple[float, float]] = {}
    for name, value in scores.items():
        if not isinstance(value, Mapping):
            raise ValueError('invalid score')
        low, high = value.get('ci_low'), value.get('ci_high')
        if not (type(low) is int or type(low) is float):
            raise ValueError('invalid lower score bound')
        if not (type(high) is int or type(high) is float):
            raise ValueError('invalid upper score bound')
        if not math.isfinite(low) or not math.isfinite(high) or not 0 <= low <= high <= 1:
            raise ValueError('invalid score interval')
        intervals[name] = low, high
    if depth == config.ROUNDS[0]:
        passes = all(bounds[0] > .51 for slot, bounds in intervals.items()
                     if slot.endswith('-correct'))
        return 'CONTINUE' if passes else 'POSITIVE_CONTROL_FAILED'
    if all(.49 <= low <= high <= .51 for low, high in intervals.values()):
        return 'NEAR_CHANCE'
    return 'ROUND_CEILING' if depth == config.ROUNDS[-1] else 'CONTINUE'

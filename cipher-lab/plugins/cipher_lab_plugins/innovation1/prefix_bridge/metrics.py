"""Group classification and paired correctness, not bit or pair accuracy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.float32]
Labels = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class Score:
    saved: dict[str, object]
    correct: NDArray[np.bool_]


def score(logits: Floats, labels: Labels) -> Score:
    if (logits.dtype != np.float32 or labels.dtype != np.uint8 or logits.shape != labels.shape
            or logits.ndim != 2 or logits.shape[1] != 1 or not len(logits)
            or not np.isfinite(logits).all() or not (labels <= 1).all()):
        raise ValueError('INVALID_TEST_GROUPS')
    y, z = labels[:, 0], logits[:, 0]
    positive, guess = y == 1, z >= 0
    correct = positive == guess
    tp, tn = int((positive & guess).sum()), int((~positive & ~guess).sum())
    positives = int(positive.sum())
    negatives = len(y) - positives
    return Score({
        'groups': len(y), 'correct_groups': tp + tn, 'accuracy': (tp + tn) / len(y),
        'bce': float(np.logaddexp(0., np.where(positive, -z, z).astype(np.float64)).mean()),
        'by_label': {'0': {'groups': negatives, 'correct': tn},
                     '1': {'groups': positives, 'correct': tp}},
        'confusion': {'tn': tn, 'fp': negatives - tn, 'fn': positives - tp, 'tp': tp},
    }, correct)


def compare(joint: Score, control: Score) -> dict[str, int | float]:
    a, b = joint.correct, control.correct
    if not len(a) or a.shape != b.shape:
        raise ValueError('UNALIGNED_TEST_GROUPS')
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    return {'percentage_points': 100 * (only_a - only_b) / len(a),
            'both_correct': int((a & b).sum()), 'joint_only_correct': only_a,
            'control_only_correct': only_b, 'both_wrong': int((~a & ~b).sum())}

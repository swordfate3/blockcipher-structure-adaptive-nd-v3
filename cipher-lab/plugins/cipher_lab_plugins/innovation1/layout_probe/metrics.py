"""Fixed-class, paired TEST uncertainty from aligned labels and logits only."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

Z_95 = 1.95996398454


def paired_statistics(labels: Any, row_logits: Any, column_logits: Any) -> dict[str, Any]:
    y, a, b = (np.asarray(v, dtype=np.float64) for v in (labels, row_logits, column_logits))
    if (y.ndim != 2 or y.shape[1] != 1 or y.shape != a.shape or y.shape != b.shape
            or not np.isin(y, (0, 1)).all() or not np.isfinite(a).all()
            or not np.isfinite(b).all()):
        raise ValueError('PAIRED_SCORE_INVALID')
    a_neg, b_neg = np.where(y == 1, -a, a), np.where(y == 1, -b, b)
    delta = np.logaddexp(0., b_neg) - np.logaddexp(0., a_neg)
    strata = {}
    for label in (0, 1):
        values = delta[y == label]
        n = int(values.size)
        if n < 2 or n * 2 != y.size:
            raise ValueError('PAIRED_SCORE_QUOTA')
        strata[str(label)] = dict(n=n, mean=float(values.mean()),
                                  sample_variance=float(values.var(ddof=1)))
    mean = sum(.5 * s['mean'] for s in strata.values())
    se = .5 * math.sqrt(sum(s['sample_variance'] / s['n'] for s in strata.values()))
    lo, hi = mean - Z_95 * se, mean + Z_95 * se
    if not all(math.isfinite(v) for v in (mean, se, lo, hi)):
        raise ValueError('PAIRED_SCORE_INVALID')
    delta_acc = (np.sum((b >= 0) == y) - np.sum((a >= 0) == y)) / y.size
    return dict(delta_bce=mean, standard_error=se, ci95=[lo, hi], strata=strata,
                direction='lower' if hi < 0 else 'higher' if lo > 0 else 'uncertain',
                delta_acc_pp=100. * float(delta_acc))

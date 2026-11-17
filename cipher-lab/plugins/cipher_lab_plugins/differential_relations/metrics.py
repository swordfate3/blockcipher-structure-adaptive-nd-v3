"""TRAIN-only baselines and numerical statistics; no oracle, model or sampling."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
U64 = NDArray[np.uint64]
U8 = NDArray[np.uint8]


def _matrix(values: NDArray[Any]) -> F64:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4 or len(a) < 1 or not np.all(np.isfinite(a)):
        raise ValueError('EXPECTED_FINITE_N_BY_4')
    return a


def _binary(values: NDArray[Any]) -> F64:
    a = _matrix(values)
    if not np.all((a == 0) | (a == 1)):
        raise ValueError('EXPECTED_BINARY_LABELS')
    return a


def logloss(logits: NDArray[Any], labels: NDArray[Any]) -> F64:
    z, y = _matrix(logits), _binary(labels)
    if z.shape != y.shape:
        raise ValueError('SHAPE_MISMATCH')
    # Avoid inf-inf and avoid cancellation for large positive correct logits.
    return np.logaddexp(0., np.where(y == 1, -z, z))


def _words(orbits: U64) -> U64:
    if (orbits.dtype != np.uint64 or orbits.ndim != 1 or len(orbits) == 0
            or np.any(orbits & np.uint64(64))):
        raise ValueError('EXPECTED_CANONICAL_ORBITS')
    return orbits


@dataclass(frozen=True, slots=True)
class Baselines:
    constant: F64
    stump_bits: NDArray[np.int64]
    stump_probabilities: F64
    train_rows: int

    def probabilities(self, orbits: U64) -> tuple[F64, F64]:
        words = _words(orbits)
        cells = ((words[:, None] >> self.stump_bits.astype(np.uint64)) & 1).astype(np.int64)
        stump = self.stump_probabilities[np.arange(4), cells]
        return np.broadcast_to(self.constant, (len(words), 4)).copy(), stump

    def logits(self, orbits: U64) -> tuple[F64, F64]:
        c, s = self.probabilities(orbits)
        return np.log(c) - np.log1p(-c), np.log(s) - np.log1p(-s)


def fit_baselines(train_orbits: U64, train_labels: U8) -> Baselines:
    words, y = _words(train_orbits), _binary(train_labels)
    if len(words) != len(y):
        raise ValueError('SHAPE_MISMATCH')
    n = len(y)
    ones = y.sum(axis=0, dtype=np.float64)
    constant = (ones + 1.) / (n + 2.)
    bits = np.zeros(4, dtype=np.int64)
    probabilities = np.empty((4, 2), dtype=np.float64)
    best = np.full(4, np.inf)
    for bit in range(64):
        if bit == 6:
            continue
        cell = ((words >> np.uint64(bit)) & 1).astype(np.bool_)
        n1 = int(cell.sum())
        # Aggregate TRAIN BCE, without expanding predictions or consulting any other split.
        positive1 = y[cell].sum(axis=0, dtype=np.float64)
        counts = np.array([n - n1, n1], dtype=np.float64)
        positive = np.stack((ones - positive1, positive1), axis=1)
        q = (positive + 1.) / (counts[None, :] + 2.)
        score = -(positive * np.log(q) + (counts[None, :] - positive) * np.log1p(-q)).sum(1) / n
        improve = score < best  # Ascending LSB0 iteration preserves smallest exact tie.
        bits[improve] = bit
        probabilities[improve] = q[improve]
        best[improve] = score[improve]
    for a in (constant, bits, probabilities):
        a.setflags(write=False)
    return Baselines(constant, bits, probabilities, n)


def relation_metrics(logits: NDArray[Any], labels: NDArray[Any]) -> list[dict[str, Any]]:
    loss = logloss(logits, labels)
    pred, y = _matrix(logits) > 0, _binary(labels).astype(bool)
    results: list[dict[str, Any]] = []
    for j in range(4):
        p, t = pred[:, j], y[:, j]
        tn, fp = int(np.sum(~p & ~t)), int(np.sum(p & ~t))
        fn, tp = int(np.sum(~p & t)), int(np.sum(p & t))
        r0 = tn / (tn + fp) if tn + fp else None
        r1 = tp / (tp + fn) if tp + fn else None
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        reasons: dict[str, str] = {}
        if r0 is None:
            reasons['recall_0'] = 'class_0_absent'
        if r1 is None:
            reasons['recall_1'] = 'class_1_absent'
        if r0 is None or r1 is None:
            reasons['balanced_accuracy'] = 'class_absent'
        if denom == 0:
            reasons['mcc'] = 'zero_marginal_product'
        results.append({
            'rows': len(y), 'ones': tp + fn, 'bce': float(loss[:, j].mean()),
            'accuracy': (tp + tn) / len(y),
            'balanced_accuracy': (r0 + r1) / 2 if r0 is not None and r1 is not None else None,
            'recall_0': r0, 'recall_1': r1,
            'mcc': (tp * tn - fp * fn) / denom if denom else None,
            'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
            'undefined_reasons': reasons,
        })
    return results


def _paired(model: F64, base: F64) -> dict[str, Any]:
    # The unit is an orbit, never four independent relations.
    with np.errstate(over='ignore', invalid='ignore'):
        diff = (base - model).mean(axis=1, dtype=np.float64)
        lb, lm = float(base.mean()), float(model.mean())
    result: dict[str, Any] = {
        'baseline_bce': lb if math.isfinite(lb) else None,
        'mean_difference': None, 'relative_improvement': None,
        'ci_low': None, 'ci_high': None, 'confidence': .975, 'sample_sd': None,
        'max_abs_difference': None, 'first_half_mean': None, 'second_half_mean': None,
        'max_centered_square_fraction': None, 'uncertain': True, 'reasons': [],
    }
    if not np.isfinite(diff).all() or not math.isfinite(lb) or not math.isfinite(lm):
        result['reasons'] = ['nonfinite']
        return result
    # Subtract an anchor first to avoid a constant panel acquiring roundoff variance.
    mean = float(diff[0]) + float((diff - diff[0]).mean())
    n = len(diff)
    half = n // 2
    result.update(mean_difference=mean, relative_improvement=(lb - lm) / lb if lb > 0 else None,
                  max_abs_difference=float(np.max(np.abs(diff))),
                  first_half_mean=float(diff[:half].mean()) if half else None,
                  second_half_mean=float(diff[half:].mean()))
    if n < 2:
        result['reasons'] = ['insufficient_orbits']
        return result
    with np.errstate(over='ignore', invalid='ignore'):
        squares = np.square(diff - mean)
        ss = float(squares.sum(dtype=np.float64))
    if not math.isfinite(ss):
        result['reasons'] = ['nonfinite_variance']
        return result
    if ss == 0:
        result.update(sample_sd=0., reasons=['zero_variance'])
        return result
    sd = math.sqrt(ss / (n - 1))
    fraction = float(squares.max()) / ss
    radius = statistics.NormalDist().inv_cdf(.9875) * sd / math.sqrt(n)
    result.update(sample_sd=sd, ci_low=mean - radius, ci_high=mean + radius,
                  max_centered_square_fraction=fraction, uncertain=fraction > .01,
                  reasons=['extreme_orbit_dominance'] if fraction > .01 else [])
    return result


def paired_bce_comparison(model_losses: NDArray[Any], constant_losses: NDArray[Any],
                          stump_losses: NDArray[Any]) -> dict[str, Any]:
    arrays = [np.asarray(a, dtype=np.float64) for a in
              (model_losses, constant_losses, stump_losses)]
    if (any(a.ndim != 2 or a.shape[1] != 4 or len(a) == 0 for a in arrays)
            or any(a.shape != arrays[0].shape for a in arrays)):
        raise ValueError('SHAPE_MISMATCH')
    if any(np.any(a < 0) for a in arrays):
        raise ValueError('NEGATIVE_LOSS')
    c, s = (_paired(arrays[0], a) for a in arrays[1:])
    value = float(arrays[0].mean())
    signal = all(not row['uncertain'] and row['ci_low'] is not None and row['ci_low'] > 0
                 and row['relative_improvement'] is not None
                 and row['relative_improvement'] >= .01 for row in (c, s))
    return {'model_bce': value if math.isfinite(value) else None,
            'comparisons': {'constant': c, 'stump': s}, 'main_signal': signal}


def mcnemar_exact(model_only_correct: int, baseline_only_correct: int) -> float:
    a, b = model_only_correct, baseline_only_correct
    if any(type(v) is not int or v < 0 for v in (a, b)):
        raise ValueError('INVALID_DISCORDANT_COUNT')
    n, k = a + b, min(a, b)
    if n == 0 or 2 * k >= n - 1:
        return 1.
    # Exact binomial-tail definition, evaluated in log space with a scaled recurrence.
    # No factorials/2**n float conversion, no asymptotic chi-square approximation.
    logmass = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2.)
    term = total = 1.
    for i in range(k, 0, -1):
        term *= i / (n - i + 1)
        total += term
        ratio = (i - 1) / (n - i + 2)
        # Bound the uncomputed lower tail by a geometric series at its largest ratio.
        if ratio < 1 and term * ratio / (1 - ratio) <= total * 1e-16:
            break
    return min(1., math.exp(logmass + math.log(total) + math.log(2.)))


def accuracy_comparisons(model_logits: NDArray[Any], constant_logits: NDArray[Any],
                         stump_logits: NDArray[Any], labels: NDArray[Any]
                         ) -> list[dict[str, Any]]:
    y = _binary(labels).astype(bool)
    model = _matrix(model_logits)
    if model.shape != y.shape:
        raise ValueError('SHAPE_MISMATCH')
    correct = (model > 0) == y
    rows: list[dict[str, Any]] = []
    for name, values in (('constant', constant_logits), ('stump', stump_logits)):
        base = _matrix(values)
        if base.shape != y.shape:
            raise ValueError('SHAPE_MISMATCH')
        bc = (base > 0) == y
        for j in range(4):
            a = int(np.sum(correct[:, j] & ~bc[:, j]))
            b = int(np.sum(~correct[:, j] & bc[:, j]))
            rows.append({'baseline': name, 'relation': f'R{j + 1}', 'rows': len(y),
                         'model_only_correct': a, 'baseline_only_correct': b,
                         'accuracy_improvement': (a - b) / len(y),
                         'p': mcnemar_exact(a, b)})
    order = sorted(range(8), key=lambda i: rows[i]['p'])
    previous = 0.
    for rank, index in enumerate(order):
        row = rows[index]
        previous = max(previous, min(1., (8 - rank) * row['p']))
        row.update(adjusted_p=previous, alpha=.05,
                   accuracy_signal=previous < .05 and row['accuracy_improvement'] >= .01)
    return rows

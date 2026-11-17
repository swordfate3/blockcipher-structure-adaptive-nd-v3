"""Public residual transforms and paired statistics on explicit saved outputs only.

No model forward, private data access, label generation, fitting or persistence.
Exact conditional tails use a log-PMF and decreasing recurrence on the smaller
side; no factorial/binomial integers or cancellation in small positive tails.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from cipher_lab_plugins.prediction_common.metrics import TrainBaseline, _tensor, binary_metrics


def absolute_logits(residual_logits: Tensor, reference: Tensor) -> Tensor:
    """Flip once. In particular +/-0 remains an absolute zero prediction."""
    _tensor(residual_logits, (1,))
    _tensor(reference, (1,), binary=True)
    if residual_logits.shape != reference.shape:
        raise ValueError('reference row mismatch')
    return (1 - 2 * reference) * residual_logits


def binomial_tail(b: int, c: int) -> float:
    """P[Binomial(b+c, .5) >= b], including b=c=0 -> 1.

    Only a scalar recurrence is needed near the centre (O(sqrt(m)) effective
    terms), and extreme underflow correctly rounds to zero in float64. Symmetry
    avoids summing the long, increasing side of the distribution.
    """
    if (type(b) is not int or type(c) is not int or min(b, c) < 0 or b+c > 2000000):
        raise ValueError('invalid discordant counts')
    m = b+c
    if b == 0:
        return 1.

    def upper(k: int) -> float:
        log_mass = (math.lgamma(m+1) - math.lgamma(k+1) - math.lgamma(m-k+1)
                    - m * math.log(2.))
        term = math.exp(log_mass)
        terms = [term]
        total = term
        for j in range(k, m):
            term *= (m-j) / (j+1)
            terms.append(term)
            total += term
            if term <= total * 1e-17:
                break
        return math.fsum(terms)

    result = upper(b) if b > m/2 else 1. - upper(m-b+1)
    return max(0., min(1., result))


def paired_accuracy(direct_logits: Tensor, residual_absolute_logits: Tensor,
                    labels: Tensor) -> dict[str, object]:
    """Paired absolute-bit ACC plus residual-minus-direct per-row BCE aggregates.

    Normal interval is descriptive only and omitted unless both discordances
    are >=25. The exact one-sided conditional tail is always the test statistic.
    """
    for value in (direct_logits, residual_absolute_logits):
        _tensor(value, (1,))
    _tensor(labels, (1,), binary=True)
    if direct_logits.shape != labels.shape or residual_absolute_logits.shape != labels.shape:
        raise ValueError('paired row mismatch')
    with torch.no_grad():
        direct = (direct_logits > 0) == labels
        residual = (residual_absolute_logits > 0) == labels
        b = int((~direct & residual).sum())
        c = int((direct & ~residual).sum())
        both_correct = int((direct & residual).sum())
        both_wrong = int((~direct & ~residual).sum())
        n = len(labels)
        delta = (b-c)/n
        interval = None
        if b >= 25 and c >= 25:
            variance = ((b+c) - (b-c)**2/n) / (n-1)
            radius = 1.959963984540054 * math.sqrt(variance/n)
            interval = [max(-1., delta-radius), min(1., delta+radius)]
        loss = (F.binary_cross_entropy_with_logits(residual_absolute_logits.double(),
                    labels.double(), reduction='none')
                - F.binary_cross_entropy_with_logits(direct_logits.double(), labels.double(),
                                                     reduction='none'))
        return {'n': n, 'b': b, 'c': c, 'both_correct': both_correct,
                'both_wrong': both_wrong, 'sum_d': b-c, 'sum_d2': b+c, 'delta': delta,
                'p_one_sided': binomial_tail(b, c), 'normal_95_interval': interval,
                'loss_difference': {'direction': 'residual-minus-direct',
                                    'sum': float(loss.sum()),
                                    'sum_squares': float(loss.square().sum()),
                                    'mean': float(loss.mean())}}


def reference_diagnostics(labels: Tensor, reference: Tensor,
                          train_residual: TrainBaseline) -> dict[str, object]:
    """Freeze flip/probability on TRAIN, then apply unchanged to any explicit split."""
    train_residual.__post_init__()
    flip = train_residual.majority
    probability = train_residual.probability
    logits = absolute_logits(torch.full_like(reference, math.log(probability/(1-probability))),
                             reference)
    fixed = absolute_logits(torch.full_like(reference, 1. if flip else -1.), reference)
    return {'train_residual_constant': flip,
            'train_residual_probability': probability,
            'fixed_flip': binary_metrics(fixed, labels, majority=flip, probability=probability),
            'residual_probability': binary_metrics(logits, labels, majority=flip,
                                                   probability=probability)}

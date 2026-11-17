"""Saved/artificial group scores only; no model/data generation dependencies.

Accuracy bootstrap compresses each class to paired correctness differences
(-1,0,+1). Multinomial resampling is exactly the pushforward distribution of
paired group-index resampling for this statistic, not a normal approximation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vectors(labels: ArrayLike, shared: ArrayLike, dedicated: ArrayLike
             ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    arrays = [np.asarray(x, dtype=np.float64) for x in (labels, shared, dedicated)]
    if any(a.ndim not in (1, 2) or (a.ndim == 2 and a.shape[1] != 1) for a in arrays):
        raise ValueError('one group label/logit per row required')
    y, m, d = (a.reshape(-1) for a in arrays)
    if (y.size < 2 or y.size != m.size or y.size != d.size
            or not all(np.isfinite(a).all() for a in arrays)
            or not np.isin(y, [0, 1]).all()
            or np.count_nonzero(y == 0) != np.count_nonzero(y == 1)):
        raise ValueError('finite aligned logits and balanced binary group labels required')
    return y, m, d


def difference_counts(labels: ArrayLike, shared: ArrayLike, dedicated: ArrayLike
                      ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Negative then positive class; bins are M correctness minus D: -1,0,+1."""
    y, m, d = _vectors(labels, shared, dedicated)
    delta = ((m >= 0) == y).astype(np.int8) - ((d >= 0) == y).astype(np.int8)
    counts = [tuple(int(np.count_nonzero(delta[y == cls] == value)) for value in (-1, 0, 1))
              for cls in (0, 1)]
    return (counts[0][0], counts[0][1], counts[0][2]), (
        counts[1][0], counts[1][1], counts[1][2])


def bootstrap_interval(counts: tuple[tuple[int, int, int], tuple[int, int, int]], *,
                       replicates: int = 20000, seed: int = 26111744,
                       chunk_size: int = 2048) -> list[float]:
    """98.75% linear percentile CI in accuracy units; O(R+chunk_size) storage."""
    if (len(counts) != 2 or any(len(c) != 3 for c in counts)
            or any(type(c) is not int or c < 0 for row in counts for c in row)
            or sum(counts[0]) <= 0 or sum(counts[0]) != sum(counts[1])
            or type(replicates) is not int or replicates <= 0
            or type(chunk_size) is not int or chunk_size <= 0
            or type(seed) is not int or seed < 0):
        raise ValueError('balanced counts, positive R/chunk and public seed required')
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(replicates, dtype=np.float64)
    n = sum(counts[0])
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        totals = np.zeros(stop - start, dtype=np.int64)
        for row in counts:
            draws = rng.multinomial(n, np.asarray(row, dtype=np.float64) / n,
                                    size=stop - start)
            totals += draws[:, 2] - draws[:, 0]
        samples[start:stop] = totals / (2 * n)
    return [float(x) for x in np.quantile(samples, [0.00625, 0.99375], method='linear')]


def compare(labels: ArrayLike, shared: ArrayLike, dedicated: ArrayLike, *,
            replicates: int = 20000, seed: int = 26111744) -> dict[str, Any]:
    """Group ACC/BCE, paired accuracy difference and Bonferroni percentile CI.

    Caller supplies a per-task public bootstrap seed (data.derive_seed supports
    purpose='bootstrap'). No claim about initialization or worst-key variation.
    """
    y, m, d = _vectors(labels, shared, dedicated)

    def scores(logits: NDArray[np.float64]) -> dict[str, float]:
        # Stable BCE avoids subtractive cancellation for large positive logits.
        loss = np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
        return dict(accuracy=float(np.mean((logits >= 0) == y)), bce=float(np.mean(loss)))

    ms, ds = scores(m), scores(d)
    counts = difference_counts(y, m, d)
    return dict(shared=ms, dedicated=ds,
                accuracy_difference=ms['accuracy'] - ds['accuracy'],
                difference_counts=[list(row) for row in counts],
                interval=bootstrap_interval(counts, replicates=replicates, seed=seed),
                confidence=0.9875, family_alpha=0.05, comparisons=4, replicates=replicates,
                seed=seed, quantile='linear', unit='group-accuracy',
                method='fixed-class-quota-paired-group-multinomial')

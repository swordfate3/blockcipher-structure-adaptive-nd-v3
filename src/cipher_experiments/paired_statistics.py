"""Explicit saved-prediction postprocessing, never an automatic evaluation step.

Intervals concern rows conditional on a trained model/panel, not seeds or keys.
The caller owns the full 18-cell family and must report incomplete cells; skipping
D self-comparison does not redistribute alpha. Ten-thousand tail quantiles have
Monte Carlo error and are not a cryptographic proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .prediction_evidence import PredictionEvidence, _array, _constant, _identity


@dataclass(frozen=True)
class ConstantBaseline:
    """TRAIN-frozen bit/probability, optionally conditional on public q bits."""

    bit: int
    probability: float
    x_sha256: str
    y_sha256: str
    rows: int
    split: str
    reference_bits: NDArray[np.uint8] | None = None

    def __post_init__(self) -> None:
        _constant(self.bit, self.probability)
        _identity(self.x_sha256)
        _identity(self.y_sha256)
        if type(self.rows) is not int or not 0 < self.rows <= 1_000_000:
            raise ValueError("invalid baseline rows")
        if type(self.split) is not str or self.split not in ("train", "dev", "test"):
            raise ValueError("invalid baseline split")
        if self.reference_bits is not None:
            _array(self.reference_bits, np.dtype(np.uint8), self.rows)
            owned = self.reference_bits.copy(order="C")
            owned.flags.writeable = False
            object.__setattr__(self, "reference_bits", owned)


def paired_interval(
    candidate: NDArray[np.float64], comparator: NDArray[np.float64], *, block: int,
    comparator_id: int, metric_id: int,
) -> dict[str, float | int | str]:
    """Always first minus second; fixed paired IID bootstrap, never a sign switch."""
    for cell, count in ((block, 3), (comparator_id, 3), (metric_id, 2)):
        if type(cell) is not int or not 0 <= cell < count:
            raise ValueError("invalid built-in cell index")
    for value in (candidate, comparator):
        if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float64)
                or value.ndim != 1 or not value.size or not np.isfinite(value).all()):
            raise ValueError("contributions must be nonempty finite float64 [N]")
    if candidate.shape != comparator.shape:
        raise ValueError("paired contributions must have identical shapes")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = candidate - comparator
        point = float(difference.mean())
    if not np.isfinite(difference).all() or not np.isfinite(point):
        raise ValueError("nonfinite paired difference/mean")
    seed = 196801 + 100 * block + 10 * comparator_id + metric_id
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(10000, dtype=np.float64)
    for start in range(0, 10000, 64):
        size = min(64, 10000 - start)
        indices = rng.integers(0, len(difference), size=(size, len(difference)))
        means[start:start + size] = difference[indices].mean(axis=1)
    if not np.isfinite(means).all():
        raise ValueError("nonfinite bootstrap means")
    alpha = .05 / 18
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2], method="linear")
    return {"point": point, "lower": float(lower), "upper": float(upper),
            "resamples": 10000, "seed": seed, "method": "percentile", "cell_alpha": alpha}


def _model_contributions(evidence: PredictionEvidence) -> tuple[
    NDArray[np.float64], NDArray[np.float64],
]:
    _array(evidence.logits, np.dtype(np.float32), evidence.rows)
    _array(evidence.labels, np.dtype(np.uint8), evidence.rows)
    logits = evidence.logits.astype(np.float64)
    labels = evidence.labels.astype(np.float64)
    return ((evidence.logits > 0) == evidence.labels).astype(np.float64), (
        np.logaddexp(0., logits) - labels * logits)


def paired_comparison(
    candidate: PredictionEvidence, comparator: PredictionEvidence | ConstantBaseline, *,
    block: int, comparator_id: int,
) -> dict[str, Any]:
    """ACC=candidate correct−baseline correct; BCE=baseline loss−candidate loss.

    Different methods need not share config/checkpoint identity. Identical selected
    model identities are rejected as a self-comparison; the caller records that
    cell as skipped, without reclaiming its Bonferroni allocation.
    """
    for field in ("x_sha256", "y_sha256", "rows", "split"):
        if getattr(candidate, field) != getattr(comparator, field):
            raise ValueError(f"paired {field} identity mismatch")
    candidate_correct, candidate_loss = _model_contributions(candidate)
    if isinstance(comparator, PredictionEvidence):
        if not np.array_equal(candidate.labels, comparator.labels):
            raise ValueError("paired labels/order mismatch")
        if (candidate.config_sha256 == comparator.config_sha256
                and candidate.checkpoint_sha256 == comparator.checkpoint_sha256):
            raise ValueError("skip D self-comparison; alpha remains .05/18")
        comparator_correct, comparator_loss = _model_contributions(comparator)
    elif isinstance(comparator, ConstantBaseline):
        _constant(comparator.bit, comparator.probability)
        ref = comparator.reference_bits
        if ref is None:
            predicted = np.full(candidate.rows, comparator.bit, dtype=np.uint8)
            reference = np.zeros(candidate.rows, dtype=np.uint8)
        else:
            _array(ref, np.dtype(np.uint8), candidate.rows)
            reference = ref
            predicted = ref ^ np.uint8(comparator.bit)
        comparator_correct = (predicted == candidate.labels).astype(np.float64)
        # Log-space complement avoids rounding 1-p to 1 for tiny manual p.
        log_p = np.log(comparator.probability)
        log_not_p = np.log1p(-comparator.probability)
        residual = candidate.labels ^ reference
        comparator_loss = -np.where(residual == 1, log_p, log_not_p)
    else:
        raise TypeError("unsupported comparator")
    return {"environment": {"numpy": np.__version__, "bit_generator": "PCG64"},
            "accuracy": paired_interval(candidate_correct, comparator_correct, block=block,
                                        comparator_id=comparator_id, metric_id=0),
            "bce": paired_interval(comparator_loss, candidate_loss, block=block,
                                   comparator_id=comparator_id, metric_id=1)}

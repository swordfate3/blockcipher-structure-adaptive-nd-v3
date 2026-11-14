"""Bounded public integer prediction and full-N paired statistics. No oracle."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cipher_lab.speck_study.parity import array_rows, parity_tile, tiles


def _scores(
    points: NDArray[np.uint64],
    terms: tuple[tuple[int, int], ...],
    *,
    check: Callable[[], None] = lambda: None,
) -> NDArray[np.int64]:
    n = array_rows(points, np.dtype(np.uint64))
    if len(terms) > 400 or any(type(m) is not int or type(s) is not int for m, s in terms):
        raise ValueError("invalid terms")
    if sum(abs(s) for _, s in terms) >= 2**63:
        raise ValueError("score overflow")
    score = np.zeros(n, dtype=np.int64)
    for mask, weight in terms:
        for part in tiles(n, check):
            score[part] += weight * parity_tile(points[part], mask).astype(np.int64)
    check()
    return score


def predict_both(
    points: NDArray[np.uint64],
    terms: tuple[tuple[int, int], ...],
    *,
    check: Callable[[], None] = lambda: None,
) -> tuple[NDArray[np.uint8], int, NDArray[np.uint8]]:
    score = _scores(points, terms, check=check)
    return (
        (score < 0).astype(np.uint8),
        int(np.count_nonzero(score == 0)),
        (score <= 0).astype(np.uint8),
    )


def predict(
    points: NDArray[np.uint64],
    terms: tuple[tuple[int, int], ...],
    *,
    check: Callable[[], None] = lambda: None,
) -> tuple[NDArray[np.uint8], int]:
    main, ties, _ = predict_both(points, terms, check=check)
    return main, ties


@dataclass
class Evaluation:
    count: int = 0
    correct: int = 0
    baseline_correct: int = 0
    ties: int = 0
    official_correct: int = 0
    minus: int = 0
    zero: int = 0
    plus: int = 0

    def add(
        self,
        predictions: NDArray[np.uint8],
        labels: NDArray[np.uint8],
        ties: int,
        *,
        official_predictions: NDArray[np.uint8] | None = None,
    ) -> None:
        n = array_rows(predictions, np.dtype(np.uint8))
        if (
            array_rows(labels, np.dtype(np.uint8)) != n
            or np.any(predictions > 1)
            or np.any(labels > 1)
            or type(ties) is not int
            or not 0 <= ties <= n
        ):
            raise ValueError("invalid evaluation rows")
        if official_predictions is not None and (
            array_rows(official_predictions, np.dtype(np.uint8)) != n
            or np.any(official_predictions > 1)
        ):
            raise ValueError("invalid official rows")
        ok = predictions == labels
        base = labels == 0
        diff = ok.astype(np.int8) - base.astype(np.int8)
        self.count += n
        self.correct += int(np.count_nonzero(ok))
        self.baseline_correct += int(np.count_nonzero(base))
        self.ties += ties
        self.minus += int(np.count_nonzero(diff == -1))
        self.zero += int(np.count_nonzero(diff == 0))
        self.plus += int(np.count_nonzero(diff == 1))
        if official_predictions is not None:
            self.official_correct += int(np.count_nonzero(official_predictions == labels))

    def snapshot(self) -> dict[str, Any]:
        return dict(
            count=self.count,
            correct=self.correct,
            baseline_correct=self.baseline_correct,
            ties=self.ties,
            official_tie1_correct=self.official_correct,
            d_histogram={"-1": self.minus, "0": self.zero, "1": self.plus},
        )

    def finish(self, expected: int) -> dict[str, Any]:
        if self.count != expected or expected <= 0:
            raise ValueError("incomplete evaluation")
        delta = (self.plus - self.minus) / expected
        return dict(
            self.snapshot(),
            accuracy=self.correct / expected,
            baseline_accuracy=self.baseline_correct / expected,
            delta=delta,
            lower=delta - math.sqrt(2 * math.log(100) / expected),
        )

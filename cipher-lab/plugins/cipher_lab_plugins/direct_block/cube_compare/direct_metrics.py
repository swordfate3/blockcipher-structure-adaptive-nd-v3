"""Descriptive absolute-word TEST counts and a TRAIN-only majority baseline."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

import numpy as np

from . import config
from .data import Floats, Labels
from .low_metrics import Score


class DirectResult(TypedDict):
    rows: int
    denominators: dict[str, int]
    scores: dict[str, Score]
    train_majority_bits: list[int]
    label_ones_by_column: list[int]
    label_zeros_by_column: list[int]


def _labels(labels: Labels) -> None:
    if (labels.dtype != np.uint8 or labels.ndim != 2 or labels.shape[1] != 64
            or not len(labels) or (labels > 1).any()):
        raise ValueError('invalid full64 labels')


def train_majority(batches: Iterable[Labels]) -> tuple[int, ...]:
    ones = np.zeros(64, dtype=np.int64)
    rows = 0
    for labels in batches:
        _labels(labels)
        ones += labels.sum(axis=0, dtype=np.int64)
        rows += len(labels)
    if not rows:
        raise ValueError('empty TRAIN labels')
    return tuple(int(bit) for bit in ones * 2 > rows)


class DirectScore:
    def __init__(self, rows: int, majority: tuple[int, ...], rounds: int) -> None:
        if type(rows) is not int or rows <= 0:
            raise ValueError('invalid TEST rows')
        if len(majority) != 64 or any(type(b) is not int or b not in (0, 1) for b in majority):
            raise ValueError('invalid TRAIN majority')
        if type(rounds) is not int or rounds not in config.LOW_ROUNDS:
            raise ValueError('invalid direct64 round')
        self._expected, self._majority = rows, majority
        self._names = ('lstm', 'majority', 'copy') if rounds == 1 else ('lstm', 'majority')
        self._rows = 0
        self._ones = np.zeros(64, dtype=np.int64)
        self._columns = np.zeros((len(self._names), 64), dtype=np.int64)
        self._whole = np.zeros(len(self._names), dtype=np.int64)

    def add(self, inputs: Floats, labels: Labels, logits: Floats) -> None:
        _labels(labels)
        if self._rows + len(labels) > self._expected:
            raise ValueError('too many TEST rows')
        if (inputs.dtype != np.float32 or inputs.shape != labels.shape
                or not ((inputs == 0) | (inputs == 1)).all()):
            raise ValueError('invalid TEST inputs')
        if (logits.dtype != np.float32 or logits.shape != labels.shape
                or not np.isfinite(logits).all()):
            raise ValueError('invalid logits')
        majority = np.broadcast_to(np.asarray(self._majority, dtype=np.bool_), labels.shape)
        predictions = [logits >= 0, majority]
        if 'copy' in self._names:
            copied = majority.copy()
            copied[:, 32:] = inputs[:, :32]
            predictions.append(copied)
        for i, prediction in enumerate(predictions):
            good = prediction == labels
            self._columns[i] += good.sum(axis=0, dtype=np.int64)
            self._whole[i] += np.count_nonzero(good.all(axis=1))
        self._ones += labels.sum(axis=0, dtype=np.int64)
        self._rows += len(labels)

    def result(self) -> DirectResult:
        if self._rows != self._expected:
            raise ValueError('incomplete TEST rows')
        denom = config.direct_denominators(self._rows)
        scores: dict[str, Score] = {}
        for name, cols, words in zip(self._names, self._columns, self._whole, strict=True):
            correct = dict(full64=int(cols.sum()), x32=int(cols[:32].sum()),
                           y32=int(cols[32:].sum()), wholeword64=int(words))
            scores[name] = {'correct': correct, 'correct_by_column': cols.tolist(),
                            'accuracy': {key: n / denom[key] for key, n in correct.items()}}
        return {'rows': self._rows, 'denominators': denom, 'scores': scores,
                'train_majority_bits': list(self._majority),
                'label_ones_by_column': self._ones.tolist(),
                'label_zeros_by_column': (self._rows - self._ones).tolist()}

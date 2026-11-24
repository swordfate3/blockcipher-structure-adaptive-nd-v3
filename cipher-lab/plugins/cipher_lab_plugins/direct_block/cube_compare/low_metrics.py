"""Descriptive low-round metrics on explicit MSB0 full-pair bits."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from . import config
from .data import Floats, Labels, Words

_X = (*range(32), *range(64, 96))
_Y = (*range(32, 64), *range(96, 128))
_REGIONS = {'active8': config.ACTIVE_COLUMNS, 'full128': tuple(range(config.WIDTH)),
            'x64': _X, 'y64': _Y, 'block0': tuple(range(64)), 'block1': tuple(range(64, 128))}
_METRICS = (*_REGIONS, 'wholepair')


class Score(TypedDict):
    correct: dict[str, int]
    correct_by_column: list[int]
    accuracy: dict[str, float]


class Result(TypedDict):
    rows: int
    denominators: dict[str, int]
    active_columns_msb0: list[int]
    train_majority_bits: list[int]
    label_ones_by_column: list[int]
    label_zeros_by_column: list[int]
    scores: dict[str, Score]
    families: list[dict[str, object]]


def _validate_labels(labels: Labels) -> None:
    if (labels.dtype != np.uint8 or labels.ndim != 2 or labels.shape[1] != config.WIDTH
            or not len(labels) or (labels > 1).any()):
        raise ValueError('invalid full labels')


def train_majority(train_batches: Iterable[Labels]) -> tuple[int, ...]:
    ones = np.zeros(config.WIDTH, dtype=np.int64)
    rows = 0
    for labels in train_batches:
        _validate_labels(labels)
        ones += labels.sum(axis=0, dtype=np.int64)
        rows += len(labels)
    if not rows:
        raise ValueError('empty TRAIN labels')
    return tuple(int(bit) for bit in ones * 2 > rows)


class LowScore:
    def __init__(self, rows: int, majority: tuple[int, ...], rounds: int) -> None:
        if type(rows) is not int or rows <= 0 or rows % config.PAIRS_PER_FAMILY:
            raise ValueError('invalid TEST family count')
        if len(majority) != config.WIDTH or any(type(b) is not int or b not in (0, 1)
                                               for b in majority):
            raise ValueError('invalid TRAIN majority')
        if type(rounds) is not int or rounds not in config.LOW_ROUNDS:
            raise ValueError('invalid low round')
        self._expected, self._majority = rows, majority
        self._names = ('lstm', 'majority', 'copy') if rounds == 1 else ('lstm', 'majority')
        self._rows = 0
        self._ones = np.zeros(config.WIDTH, dtype=np.int64)
        self._columns = np.zeros((len(self._names), config.WIDTH), dtype=np.int64)
        self._families: dict[int, int] = {}
        n = rows // config.PAIRS_PER_FAMILY
        self._family_rows = np.zeros(n, dtype=np.int64)
        self._family_scores = np.zeros((n, len(self._names), len(_METRICS)), dtype=np.int64)

    def add(self, ids: Words, inputs: Floats, labels: Labels, logits: Floats) -> None:
        _validate_labels(labels)
        n = len(labels)
        if ids.dtype != np.uint64 or ids.shape != (n,) or self._rows + n > self._expected:
            raise ValueError('invalid TEST rows or ids')
        if (logits.dtype != np.float32 or logits.shape != labels.shape
                or not np.isfinite(logits).all()):
            raise ValueError('invalid logits')
        if (inputs.dtype != np.float32 or inputs.shape != labels.shape
                or not ((inputs == 0) | (inputs == 1)).all()):
            raise ValueError('invalid TEST inputs')
        predictions = [logits >= 0, np.broadcast_to(self._majority, labels.shape)]
        if 'copy' in self._names:
            copied = predictions[1].copy()
            copied[:, _Y] = inputs[:, _X]
            predictions.append(copied)
        good = np.stack([p == labels for p in predictions], axis=1)  # rows, methods, columns
        by_row = np.stack([*(good[:, :, cols].sum(axis=2) for cols in _REGIONS.values()),
                           good.all(axis=2)], axis=2)
        at = np.fromiter((self._families.setdefault(int(base), len(self._families))
                          for base in ids), dtype=np.intp, count=n)
        if len(self._families) > len(self._family_rows):
            raise ValueError('too many TEST families')
        np.add.at(self._family_rows, at, 1)
        if (self._family_rows[at] > config.PAIRS_PER_FAMILY).any():
            raise ValueError('too many rows for TEST family')
        np.add.at(self._family_scores, at, by_row)
        self._columns += good.sum(axis=0, dtype=np.int64)
        self._ones += labels.sum(axis=0, dtype=np.int64)
        self._rows += n

    def _named(self, values: NDArray[np.int64]) -> dict[str, dict[str, int]]:
        return {name: dict(zip(_METRICS, row.tolist(), strict=True))
                for name, row in zip(self._names, values, strict=True)}

    def result(self) -> Result:
        if self._rows != self._expected or (self._family_rows != config.PAIRS_PER_FAMILY).any():
            raise ValueError('incomplete TEST families')
        denom = config.low_denominators(self._rows)
        totals = self._named(self._family_scores.sum(axis=0, dtype=np.int64))
        scores: dict[str, Score] = {
            name: {'correct': totals[name], 'correct_by_column': cols.tolist(),
                   'accuracy': {key: count / denom[key] for key, count in totals[name].items()}}
            for name, cols in zip(self._names, self._columns, strict=True)
        }
        return {
            'rows': self._rows, 'denominators': denom,
            'active_columns_msb0': list(config.ACTIVE_COLUMNS),
            'train_majority_bits': list(self._majority),
            'label_ones_by_column': self._ones.tolist(),
            'label_zeros_by_column': (self._rows - self._ones).tolist(),
            'scores': scores,
            'families': [{'family_id': base, 'rows': int(self._family_rows[i]),
                          'correct': self._named(self._family_scores[i])}
                         for base, i in sorted(self._families.items())],
        }

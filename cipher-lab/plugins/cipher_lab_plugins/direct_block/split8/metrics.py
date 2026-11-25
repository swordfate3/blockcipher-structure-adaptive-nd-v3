"""Streaming descriptive counts for the ordered eight-bit TEST prediction."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np

from cipher_lab.execution import Meter

from ..fit import batch_indices
from . import config
from .data import Bytes, Floats, high_bits


class NetworkScore(TypedDict):
    columns_msb0: list[int]
    bit_correct: int
    bit_denominator: int
    bit_accuracy: float
    correct_by_column: list[int]


class Result(TypedDict):
    rows: int
    bit_correct: int
    bit_denominator: int
    bit_accuracy: float
    correct_by_column: list[int]
    label_ones_by_column: list[int]
    label_zeros_by_column: list[int]
    networks: list[NetworkScore]
    train_majority_bits: list[int]
    majority_bit_correct: int
    majority_bit_denominator: int
    majority_bit_accuracy: float
    eight_all_correct: int
    eight_all_denominator: int


def train_majority(labels: Bytes, batch_size: int, meter: Meter) -> tuple[int, ...]:
    if labels.dtype != np.uint8 or labels.ndim != 1 or not len(labels):
        raise ValueError('invalid packed TRAIN labels')
    ones = np.zeros(8, dtype=np.int64)
    for at in batch_indices(len(labels), batch_size):
        meter.check()
        ones += high_bits(labels[at]).sum(axis=0, dtype=np.int64)
    meter.check()
    return tuple(int(x) for x in 2 * ones > len(labels))


class Score:
    def __init__(self, rows: int, majority: tuple[int, ...]) -> None:
        if type(rows) is not int or rows <= 0:
            raise ValueError('invalid TEST rows')
        if len(majority) != 8 or any(type(x) is not int or x not in (0, 1) for x in majority):
            raise ValueError('invalid TRAIN majority')
        self._expected, self._majority = rows, majority
        self._rows = self._whole = self._majority_correct = 0
        self._correct = np.zeros(8, dtype=np.int64)
        self._ones = np.zeros(8, dtype=np.int64)

    def add(self, labels: Bytes, chunks: Sequence[Floats]) -> None:
        if (labels.dtype != np.uint8 or labels.ndim != 2 or labels.shape[1] != 8
                or not len(labels) or self._rows + len(labels) > self._expected
                or (labels > 1).any()):
            raise ValueError('invalid TEST labels')
        if len(chunks) != len(config.COLS) or any(
            z.dtype != np.float32 or z.shape != (len(labels), 2) or not np.isfinite(z).all()
            for z in chunks
        ):
            raise ValueError('invalid TEST prediction chunks')
        good = np.concatenate(chunks, axis=1) >= 0
        good = good == labels
        self._correct += good.sum(axis=0, dtype=np.int64)
        self._whole += int(np.count_nonzero(good.all(axis=1)))
        self._majority_correct += int(np.count_nonzero(labels == self._majority))
        self._ones += labels.sum(axis=0, dtype=np.int64)
        self._rows += len(labels)

    def result(self) -> Result:
        if self._rows != self._expected:
            raise ValueError('incomplete TEST')
        rows = self._rows
        columns = [int(x) for x in self._correct]
        networks: list[NetworkScore] = []
        for left, right in config.COLS:
            pair = columns[left:right + 1]
            networks.append(NetworkScore(columns_msb0=[left, right], bit_correct=sum(pair),
                                         bit_denominator=2 * rows,
                                         bit_accuracy=sum(pair) / (2 * rows),
                                         correct_by_column=pair))
        bits = 8 * rows
        return Result(rows=rows, bit_correct=sum(columns), bit_denominator=bits,
                      bit_accuracy=sum(columns) / bits, correct_by_column=columns,
                      label_ones_by_column=[int(x) for x in self._ones],
                      label_zeros_by_column=[rows - int(x) for x in self._ones], networks=networks,
                      train_majority_bits=list(self._majority), eight_all_correct=self._whole,
                      eight_all_denominator=rows, majority_bit_correct=self._majority_correct,
                      majority_bit_denominator=bits,
                      majority_bit_accuracy=self._majority_correct / bits)

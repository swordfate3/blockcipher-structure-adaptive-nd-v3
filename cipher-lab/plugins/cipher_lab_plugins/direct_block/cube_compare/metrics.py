"""Same-TEST paired correctness, with whole families as the statistical unit."""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from .config import ACTIVE_COLUMNS, ARMS, PAIRS_PER_FAMILY, WIDTH
from .data import Floats, Labels, Words

_NAMES = ('majority', *ARMS)
_BITS = len(ACTIVE_COLUMNS)
_FAMILY_BITS = PAIRS_PER_FAMILY * _BITS
_Z = 2.241402727604947


def _active(labels: Labels) -> Labels:
    if (labels.dtype != np.uint8 or labels.ndim != 2 or labels.shape[1] != WIDTH
            or not len(labels) or (labels > 1).any()):
        raise ValueError('invalid labels')
    return labels[:, ACTIVE_COLUMNS]


def train_majority(train_batches: Iterable[Labels]) -> tuple[int, ...]:
    """Consume only the supplied TRAIN labels. The fixed tie-break is zero."""
    ones = np.zeros(_BITS, dtype=np.int64)
    rows = 0
    for labels in train_batches:
        ones += _active(labels).sum(axis=0, dtype=np.int64)
        rows += len(labels)
    if not rows:
        raise ValueError('empty TRAIN labels')
    return tuple(int(bit) for bit in ones * 2 > rows)


def paired_interval(left: Sequence[int], right: Sequence[int]) -> dict[str, float]:
    """Paired family-score difference, sample SE, untruncated 97.5% normal CI."""
    n = len(left)
    if (n < 2 or n != len(right) or any(type(v) is not int or not 0 <= v <= _FAMILY_BITS
                                       for scores in (left, right) for v in scores)):
        raise ValueError('invalid paired family scores')
    differences = (np.asarray(left, np.float64) - np.asarray(right, np.float64)) / _FAMILY_BITS
    delta = float(differences.mean())
    se = float(differences.std(ddof=1)) / math.sqrt(n)
    return {'delta': delta, 'se': se, 'lower': delta - _Z * se, 'upper': delta + _Z * se}


class PairedScore:
    def __init__(self, rows: int, majority: tuple[int, ...]) -> None:
        if type(rows) is not int or rows < 2 * PAIRS_PER_FAMILY or rows % PAIRS_PER_FAMILY:
            raise ValueError('invalid TEST family count')
        if len(majority) != _BITS or any(type(b) is not int or b not in (0, 1) for b in majority):
            raise ValueError('invalid TRAIN majority')
        self._expected, self._majority = rows, majority
        self._rows = 0
        self._ones = np.zeros(_BITS, dtype=np.int64)
        self._correct = np.zeros((len(_NAMES), _BITS), dtype=np.int64)
        # family_id -> (rows, majority correct, MLP correct, Shared-M correct)
        self._families: dict[int, tuple[int, int, int, int]] = {}

    def add(self, family_ids: Words, labels: Labels, mlp: Floats, sharedm: Floats) -> None:
        active = _active(labels)
        if family_ids.shape != (len(labels),) or family_ids.dtype != np.uint64:
            raise ValueError('invalid family ids')
        if self._rows + len(labels) > self._expected:
            raise ValueError('too many TEST rows')
        if any(z.dtype != np.float32 or z.shape != labels.shape or not np.isfinite(z).all()
               for z in (mlp, sharedm)):
            raise ValueError('invalid logits')
        good = np.stack((self._majority == active, (mlp[:, ACTIVE_COLUMNS] >= 0) == active,
                         (sharedm[:, ACTIVE_COLUMNS] >= 0) == active))
        by_row = good.sum(axis=2)
        changed: dict[int, tuple[int, int, int, int]] = {}
        for i, base in enumerate(family_ids):
            key = int(base)
            n, a, b, c = changed.get(key, self._families.get(key, (0, 0, 0, 0)))
            if n >= PAIRS_PER_FAMILY:
                raise ValueError('too many rows for TEST family')
            changed[key] = (n + 1, a + int(by_row[0, i]), b + int(by_row[1, i]),
                            c + int(by_row[2, i]))
        self._families.update(changed)
        self._correct += good.sum(axis=1)
        self._ones += active.sum(axis=0, dtype=np.int64)
        self._rows += len(labels)

    def result(self) -> dict[str, object]:
        if self._rows != self._expected or any(f[0] != PAIRS_PER_FAMILY
                                               for f in self._families.values()):
            raise ValueError('incomplete TEST families')
        denom = self._rows * _BITS
        by_bit = dict(zip(_NAMES, self._correct.tolist(), strict=True))
        totals = {name: sum(counts) for name, counts in by_bit.items()}
        majority, mlp = totals['majority'], totals['mlp']
        families = sorted(self._families.items())
        base_scores = [f[1] for _, f in families]
        mlp_scores = [f[2] for _, f in families]
        sharedm_scores = [f[3] for _, f in families]
        return {
            'rows': self._rows, 'denominator_bits': denom,
            'active_columns_msb0': list(ACTIVE_COLUMNS),
            'train_majority_bits': list(self._majority),
            'label_ones': int(self._ones.sum()), 'label_zeros': denom - int(self._ones.sum()),
            'label_ones_by_bit': self._ones.tolist(),
            'label_zeros_by_bit': (self._rows - self._ones).tolist(),
            'correct_by_bit': by_bit, 'correct_bits': totals,
            'accuracy': {name: n / denom for name, n in totals.items()},
            'confidence': .975, 'familywise_alpha': .05,
            'comparisons': {
                'mlp_minus_majority': {'delta': (mlp - majority) / denom},
                'sharedm_minus_majority': paired_interval(sharedm_scores, base_scores),
                'sharedm_minus_mlp': paired_interval(sharedm_scores, mlp_scores),
            },
            'families': [{'family_id': base, 'majority': a, 'mlp': b, 'sharedm': c}
                         for base, (_, a, b, c) in families],
        }

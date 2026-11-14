"""Full-prefix streaming GL arithmetic, exact sums, no beam or oracle ownership.

A failed/interrupted add poisons its accumulator: partial work cannot masquerade
as an empty layer. The owner meters attempts/completions and saves partial state.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from cipher_lab.speck_study.codec import StudyTerm
from cipher_lab.speck_study.parity import array_rows, integer, parity_tile, tiles
from cipher_lab.speck_study_config import SpeckStudyRecipe


def _masks(masks: tuple[int, ...], cap: int, suffix: int = 0) -> None:
    if type(masks) is not tuple or len(masks) > cap:
        raise ValueError('expected bounded tuple of masks')
    for mask in masks:
        integer(mask, 'mask', 0, 2**64-1)
        if mask & suffix:
            raise ValueError('mask outside prefix/domain')
    if len(set(masks)) != len(masks):
        raise ValueError('duplicate masks')


def initial_prefixes(bits: int, depth: int = 7) -> tuple[int, ...]:
    integer(bits, 'bits', 1, 64)
    integer(depth, 'initial depth', 0, min(bits, 7))
    return tuple(i << (64-depth) for i in range(1 << depth))


@dataclass(frozen=True, slots=True)
class IntegerScore:
    mask: int
    signed_sum: int
    count: int

    @property
    def score(self) -> float:
        return self.signed_sum / self.count


@dataclass(frozen=True, slots=True)
class BucketLayer:
    bits: int
    depth: int
    masks: tuple[int, ...]
    sums: tuple[int, ...]
    count: int
    kept: tuple[int, ...]
    engineering_limit: bool

    @property
    def status(self) -> str:
        return 'ENGINEERING_LIMIT' if self.engineering_limit else 'COMPLETE'

    @property
    def candidates(self) -> tuple[IntegerScore, ...]:
        return tuple(IntegerScore(m, s, self.count) for m, s in zip(self.masks, self.sums))

    @property
    def sample_bucket_products(self) -> int:
        return self.count * len(self.masks)


class CoefficientAccumulator:
    """Streaming coefficient integer sums; mask0 is descriptive DC, not a term."""

    def __init__(self, masks: tuple[int, ...], count: int) -> None:
        _masks(masks, 400)
        integer(count, 'count', 1, 10_000_000)
        self.masks = masks
        self.expected_count = count
        self.count = 0
        self.sums = [0] * len(masks)
        self._failed = False

    def add(self, points: NDArray[np.uint64], signs: NDArray[np.int8], *,
            check: Callable[[], None]) -> None:
        if self._failed:
            raise ValueError('accumulator invalid after interrupted/invalid add')
        try:
            n = array_rows(points, np.dtype(np.uint64))
            if array_rows(signs, np.dtype(np.int8)) != n or self.count+n > self.expected_count:
                raise ValueError('row count mismatch/overflow')
            delta = [0] * len(self.masks)
            for part in tiles(n, check):
                if not np.all((signs[part] == 1) | (signs[part] == -1)):
                    raise ValueError('signs must be +/-1')
            for i, mask in enumerate(self.masks):
                for part in tiles(n, check):
                    delta[i] += int(np.sum(signs[part] * parity_tile(points[part], mask),
                                           dtype=np.int64))
            self.sums = [a+b for a, b in zip(self.sums, delta)]
            self.count += n
        except BaseException:
            self._failed = True
            raise

    def finish(self) -> tuple[IntegerScore, ...]:
        if self._failed or self.count != self.expected_count:
            raise ValueError('incomplete or failed coefficient accumulator')
        return tuple(IntegerScore(m, s, self.count) for m, s in zip(self.masks, self.sums))


class BucketAccumulator:
    def __init__(self, bits: int, depth: int, prefixes: tuple[int, ...], count: int) -> None:
        integer(bits, 'bits', 1, 64)
        integer(depth, 'depth', 1, bits)
        integer(count, 'count', 1, 10_000_000)
        _masks(prefixes, 400, (1 << (65-depth))-1)
        self.bits, self.depth = bits, depth
        self.masks = tuple(sorted(m | b for m in prefixes for b in (0, 1 << (64-depth))))
        self.expected_count = count
        self.count = 0
        self.sums = [0] * len(self.masks)
        self._failed = False

    def add(self, points: NDArray[np.uint64], paired: NDArray[np.uint64],
            signs: NDArray[np.int8], paired_signs: NDArray[np.int8], *,
            check: Callable[[], None]) -> None:
        if self._failed:
            raise ValueError('accumulator invalid after interrupted/invalid add')
        try:
            n = array_rows(points, np.dtype(np.uint64))
            if (array_rows(paired, np.dtype(np.uint64)) != n
                    or array_rows(signs, np.dtype(np.int8)) != n
                    or array_rows(paired_signs, np.dtype(np.int8)) != n
                    or self.count + n > self.expected_count):
                raise ValueError('row count mismatch/overflow')
            xor = np.empty(n, dtype=np.uint64)
            products = np.empty(n, dtype=np.int8)
            domain_suffix = np.uint64((1 << (64-self.bits))-1)
            pair_suffix = np.uint64((1 << (64-self.depth))-1)
            for part in tiles(n, check):
                if (not np.all((signs[part] == 1) | (signs[part] == -1))
                        or not np.all((paired_signs[part] == 1) | (paired_signs[part] == -1))):
                    raise ValueError('signs must be +/-1')
                if np.any((points[part] | paired[part]) & domain_suffix):
                    raise ValueError('points outside encoded domain')
                xor[part] = points[part] ^ paired[part]
                if np.any(xor[part] & pair_suffix):
                    raise ValueError('pair must share suffix')
                products[part] = signs[part] * paired_signs[part]
            delta = [0] * len(self.masks)
            for i, mask in enumerate(self.masks):
                for part in tiles(n, check):
                    delta[i] += int(np.sum(products[part] * parity_tile(xor[part], mask),
                                           dtype=np.int64))
            self.sums = [a+b for a, b in zip(self.sums, delta)]
            self.count += n
        except BaseException:
            self._failed = True
            raise

    def finish(self, tau: float = .1, max_prefixes: int = 400) -> BucketLayer:
        if self._failed or self.count != self.expected_count:
            raise ValueError('incomplete or failed bucket accumulator')
        if type(tau) is not float or not 0 < tau <= 1:
            raise ValueError('tau must be finite float in (0,1]')
        integer(max_prefixes, 'max_prefixes', 1, 400)
        threshold = Fraction(str(tau))**2 / 2
        kept = tuple(m for m, s in zip(self.masks, self.sums)
                     if s * threshold.denominator > self.count * threshold.numerator)
        # Keep all evidence even on limit. Caller MUST stop, not advance this set.
        return BucketLayer(self.bits, self.depth, self.masks, tuple(self.sums), self.count,
                           kept, len(kept) > max_prefixes)


def select_terms(kind: str, scores: tuple[IntegerScore, ...],
                 recipe: SpeckStudyRecipe) -> tuple[StudyTerm, ...]:
    """R all nonzero abs>.05; B same filter then at most12, no intercept."""
    if type(recipe) is not SpeckStudyRecipe or kind not in ('R', 'B'):
        raise ValueError('invalid recipe/kind')
    count = recipe.r_coef_size if kind == 'R' else recipe.b_coef_size
    bits = recipe.r_bits if kind == 'R' else recipe.b_bits
    if type(scores) is not tuple or len(scores) > 400:
        raise ValueError('invalid coefficient scores')
    seen = set()
    result = []
    threshold = Fraction(str(recipe.coefficient_threshold))
    for item in scores:
        if type(item) is not IntegerScore:
            raise ValueError('invalid score type')
        integer(item.mask, 'mask', 0, 2**64-1)
        integer(item.signed_sum, 'sum', -count, count)
        if (type(item.count) is not int or item.count != count
                or (item.signed_sum-count) % 2 or item.mask in seen
                or item.mask & ((1 << (64-bits))-1)):
            raise ValueError('invalid coefficient sum/count/mask')
        seen.add(item.mask)
        if item.mask and abs(item.signed_sum)*threshold.denominator > count*threshold.numerator:
            result.append(StudyTerm(item.mask, item.signed_sum))
    if kind == 'R':
        return tuple(sorted(result, key=lambda t: t.mask))
    return tuple(sorted(result, key=lambda t: (-abs(t.signed_sum), t.mask))[:recipe.b_max_terms])

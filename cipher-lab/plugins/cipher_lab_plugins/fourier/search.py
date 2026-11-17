"""Bounded public Fourier arithmetic over uint64 points and int8 +/-1 signs.

No oracle, files, cipher labels beyond the supplied target, or private state.
Integer bucket sums determine sorting before float scores are presented. The
caller owns stage timing and meters; each row chunk checks its shared deadline.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cipher_lab_plugins.fourier.codec import (
    FourierPredictor,
    FourierTerm,
    validate_identity,
    validate_predictor,
)
from cipher_lab_plugins.fourier.config import FourierRecipe

CHUNK_ROWS = 4096
MAX_ROWS = 65536


def _integer(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}: expected bounded integer")
    return value


def _array(value: object, dtype: np.dtype[np.generic], *, empty: bool = False) -> int:
    # Subclasses (e.g. MaskedArray) may override validation/reduction semantics.
    if type(value) is not np.ndarray or value.dtype != dtype or value.ndim != 1:
        raise ValueError("array: expected plain one-dimensional native uint64 points/int8 signs")
    if not (0 if empty else 1) <= value.size <= MAX_ROWS:
        raise ValueError("array: row count outside bounded batch")
    return int(value.size)


def _chunks(n: int, check: Callable[[], None]) -> Iterator[slice]:
    for start in range(0, n, CHUNK_ROWS):
        check()
        yield slice(start, min(n, start + CHUNK_ROWS))


def _parity(points: NDArray[np.uint64], mask: int) -> NDArray[np.int8]:
    folded = points & np.uint64(mask)
    for shift in (32, 16, 8, 4, 2, 1):
        folded ^= folded >> np.uint64(shift)
    return (1 - 2 * (folded & np.uint64(1)).astype(np.int8)).astype(np.int8)


def parity_sign(points: NDArray[np.uint64], mask: int) -> NDArray[np.int8]:
    """chi_mask(P); integer bit63 is external MSB0, bit0 is external column63.

    This standalone primitive has no deadline callback; at most one bounded
    batch, computed in 4096-row pieces. Search/fit/predict check before each piece.
    """
    _integer(mask, "mask", 0, 2**64 - 1)
    n = _array(points, np.dtype(np.uint64), empty=True)
    result = np.empty(n, dtype=np.int8)
    for start in range(0, n, CHUNK_ROWS):
        stop = min(n, start + CHUNK_ROWS)
        result[start:stop] = _parity(points[start:stop], mask)
    return result


def _signs(signs: NDArray[np.int8], n: int) -> None:
    if _array(signs, np.dtype(np.int8)) != n:
        raise ValueError("signs: mismatched row count")


def _masks(masks: tuple[int, ...], *, max_count: int, suffix: int = 0) -> None:
    if type(masks) is not tuple or len(masks) > max_count:
        raise ValueError("masks: expected bounded tuple")
    for mask in masks:
        _integer(mask, "mask", 0, 2**64 - 1)
        if mask & suffix:
            raise ValueError("mask: outside prefix/input domain")
    if len(set(masks)) != len(masks):
        raise ValueError("masks: duplicate mask")


def _tau(tau: float) -> None:
    if type(tau) is not float or not np.isfinite(tau) or not 0 < tau <= 1:
        raise ValueError("tau: expected finite float in (0, 1]")


@dataclass(frozen=True, slots=True)
class BucketScore:
    mask: int
    signed_sum: int
    count: int

    @property
    def score(self) -> float:
        return self.signed_sum / self.count


@dataclass(frozen=True, slots=True)
class BeamLayer:
    """All tested candidates in mask order; masks kept in integer-score rank order."""

    depth: int
    candidates: tuple[BucketScore, ...]
    masks: tuple[int, ...]
    tested: int
    survivors: int
    kept: int
    truncated: int
    sample_bucket_products: int


def advance_beam(
    prefixes: tuple[int, ...], depth: int,
    points: NDArray[np.uint64], paired_points: NDArray[np.uint64],
    signs: NDArray[np.int8], paired_signs: NDArray[np.int8], *,
    tau: float, width: int, check: Callable[[], None],
) -> BeamLayer:
    """Extend highest-(depth-1)-bit prefixes, test strict positive bucket mass.

    Each pair must share the low 64-depth suffix. Prefix tuple order need not be
    canonical: a previous beam is score-ranked. All returned candidate records
    are canonical by mask; integer sums alone rank survivors, then mask ascending.
    """
    _integer(depth, "depth", 1, 64)
    _integer(width, "width", 1, 32)
    _tau(tau)
    _masks(prefixes, max_count=width, suffix=(1 << (65 - depth)) - 1)
    n = _array(points, np.dtype(np.uint64))
    if _array(paired_points, np.dtype(np.uint64)) != n:
        raise ValueError("paired_points: mismatched row count")
    _signs(signs, n)
    _signs(paired_signs, n)
    # A single public XOR/sign-product panel is shared by every candidate.
    xor = np.empty(n, dtype=np.uint64)
    products = np.empty(n, dtype=np.int8)
    suffix = np.uint64((1 << (64 - depth)) - 1)
    for part in _chunks(n, check):
        if not (np.all((signs[part] == 1) | (signs[part] == -1)) and
                np.all((paired_signs[part] == 1) | (paired_signs[part] == -1))):
            raise ValueError("signs: expected +/-1")
        xor[part] = points[part] ^ paired_points[part]
        if np.any(xor[part] & suffix):
            raise ValueError("pair: unequal suffix at search depth")
        products[part] = signs[part] * paired_signs[part]
    candidate_masks = sorted(mask | bit for mask in prefixes for bit in (0, 1 << (64 - depth)))
    candidates = []
    for mask in candidate_masks:
        total = 0
        for part in _chunks(n, check):
            total += int(np.sum(products[part] * _parity(xor[part], mask), dtype=np.int64))
        candidates.append(BucketScore(mask, total, n))
    survivors = [item for item in candidates if item.score > tau * tau / 2]
    ranked = sorted(survivors, key=lambda item: (-item.signed_sum, item.mask))
    masks = tuple(item.mask for item in ranked[:width])
    return BeamLayer(depth, tuple(candidates), masks, len(candidates), len(survivors),
                     len(masks), max(0, len(survivors) - width), len(candidates) * n)


def fit_coefficients(
    points: NDArray[np.uint64], signs: NDArray[np.int8], masks: tuple[int, ...], *,
    recipe: FourierRecipe, identity: dict[str, object], check: Callable[[], None],
) -> FourierPredictor | None:
    """Fit signed sample means only, never an optimizer or DEV-dependent update.

    Returned coefficient_rows counts intercept plus each distinct nonzero input
    candidate panel, including candidates subsequently dropped by the threshold.
    Zero mask is accepted as an input candidate but is estimated only once, as
    intercept, never as a discovery. None means no nonzero terms, not a zero model.
    """
    if type(recipe) is not FourierRecipe:
        raise ValueError("recipe: expected FourierRecipe")
    public_identity = validate_identity(identity, recipe)
    suffix = (1 << (64 - recipe.input_bits)) - 1
    _masks(masks, max_count=recipe.beam_width, suffix=suffix)
    n = _array(points, np.dtype(np.uint64))
    _signs(signs, n)
    if n != recipe.coef_size:
        raise ValueError("coef panel: wrong sample count")
    intercept_sum = 0
    for part in _chunks(n, check):
        if np.any(points[part] & np.uint64(suffix)):
            raise ValueError("points: outside synthetic input domain")
        if not np.all((signs[part] == 1) | (signs[part] == -1)):
            raise ValueError("signs: expected +/-1")
        intercept_sum += int(np.sum(signs[part], dtype=np.int64))
    terms = []
    nonzero = sorted(mask for mask in masks if mask)
    for mask in nonzero:
        total = 0
        for part in _chunks(n, check):
            total += int(np.sum(signs[part] * _parity(points[part], mask), dtype=np.int64))
        coefficient = total / n
        if total and abs(coefficient) >= recipe.tau / 2:
            terms.append(FourierTerm(mask, total, coefficient))
    if not terms:
        return None
    model = FourierPredictor(recipe, public_identity, n, intercept_sum, intercept_sum / n,
                             int(intercept_sum < 0), tuple(terms), n * (1 + len(nonzero)))
    validate_predictor(model)
    return model


def predict(
    predictor: FourierPredictor, points: NDArray[np.uint64], *, check: Callable[[], None],
) -> NDArray[np.uint8]:
    """Frozen sign reconstruction; score<0 -> absolute target bit1, tie -> bit0.

    The caller meters len(points) prediction rows; this function neither mutates
    the model nor queries labels. Arithmetic shares the model's exact denominator,
    so integer numerators avoid floating summation changing zero-score tie rules.
    """
    validate_predictor(predictor)
    n = _array(points, np.dtype(np.uint64), empty=True)
    suffix = np.uint64((1 << (64 - predictor.recipe.input_bits)) - 1)
    result = np.empty(n, dtype=np.uint8)
    for part in _chunks(n, check):
        if np.any(points[part] & suffix):
            raise ValueError("points: outside synthetic input domain")
        score = np.full(points[part].size, predictor.intercept_sum, dtype=np.int64)
        for term in predictor.terms:
            check()
            # int64 first: int8 signs times a Python integer can overflow int8.
            score += term.signed_sum * _parity(points[part], term.mask).astype(np.int64)
        result[part] = (score < 0).astype(np.uint8)
    return result

"""Exact public Boolean span compression, never a new plan or training input.

The production reducer chooses LOWEST coordinates; the verifier independently
reconstructs source ANFs and reduces selected original columns by HIGHEST
coordinates. Both verification checks use the budgeted source expansion cache,
not a second full expansion. Only small literal-expansion tests cross-check the
expander independently. Callers own plan validation and wall/CPU/RSS supervision;
checkpoint must raise EngineeringLimit for cooperative engineering limits.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .simon_anf import Budget, EngineeringLimit, Polynomial
from .simon_frontier import PublicFeaturePlan

_MAX_COORDINATES = 200_000
_CHECK_INTERVAL = 4096


@dataclass(frozen=True)
class CompressionResult:
    """Progress is not a certified rank unless both full checks have passed.

    Witness bit j names original column basis_indices[j]. Prefixes count fully
    finished columns only. The independent prefix advances during reconstruction
    AFTER the entire selected basis has independently passed highest-pivot
    elimination. Thus every counted reconstruction uses an independent basis.
    """

    status: str
    reason: str | None
    candidate_count: int
    basis_indices: tuple[int, ...]
    witnesses: tuple[int, ...]
    exact_rank: int | None
    expanded_prefix: int
    eliminated_prefix: int
    independently_verified_prefix: int
    reported_rank_estimate: int | None
    coordinate_count: int
    work: Mapping[str, int]

    def __post_init__(self) -> None:
        # A caller must not mutate an old result by continuing the shared ledger.
        object.__setattr__(self, 'work', MappingProxyType(dict(self.work)))

    def to_dict(self) -> dict[str, object]:
        return {
            'status': self.status, 'reason': self.reason,
            'candidate_count': self.candidate_count,
            'basis_indices': list(self.basis_indices),
            'witnesses': [hex(witness) for witness in self.witnesses],
            'exact_rank': self.exact_rank,
            'expanded_prefix': self.expanded_prefix,
            'eliminated_prefix': self.eliminated_prefix,
            'independently_verified_prefix': self.independently_verified_prefix,
            'reported_rank_estimate': self.reported_rank_estimate,
            'coordinate_count': self.coordinate_count, 'work': dict(self.work),
        }


class _Clock:
    """One callback per <=4096 loop steps, shared across all production phases."""

    def __init__(self, checkpoint: Callable[[], None]) -> None:
        self.checkpoint = checkpoint
        self.pending = 0

    def check(self) -> None:
        self.checkpoint()
        self.pending = 0

    def step(self) -> None:
        self.pending += 1
        if self.pending >= _CHECK_INTERVAL:
            self.check()


def _and(left: Polynomial, right: Polynomial, budget: Budget, clock: _Clock) -> Polynomial:
    """Same reservations/order as anf_and, with interruptible Cartesian loops."""
    pairs = len(left) * len(right)
    budget.charge('pairwork', pairs)
    budget.charge('term_work', pairs)
    terms: set[int] = set()
    for a in left:
        clock.step()  # Empty right operands must not create unchecked outer loops.
        for b in right:
            clock.step()
            mask = a | b
            if mask in terms:
                terms.remove(mask)
            else:
                terms.add(mask)
    return frozenset(terms)


def _xor(left: Polynomial, right: Polynomial, budget: Budget, clock: _Clock) -> Polynomial:
    """Same term reservation as anf_xor; cancellation never refunds work."""
    budget.charge('term_work', len(left) + len(right))
    terms: set[int] = set()
    for mask in left:
        clock.step()
        terms.add(mask)
    for mask in right:
        clock.step()
        if mask in terms:
            terms.remove(mask)
        else:
            terms.add(mask)
    return frozenset(terms)


@dataclass
class _VerifiedProgress:
    prefix: int = 0


def _verify(polynomials: tuple[Polynomial, ...], basis_indices: tuple[int, ...],
            witnesses: tuple[int, ...], budget: Budget, clock: _Clock,
            progress: _VerifiedProgress) -> bool:
    """No production pivots, row conversion, or reduction helper is reused."""
    clock.check()
    if (type(polynomials) is not tuple or type(basis_indices) is not tuple
            or type(witnesses) is not tuple or len(witnesses) != len(polynomials)
            or len(basis_indices) > len(polynomials)):
        return False
    selected: set[int] = set()
    for index in basis_indices:
        clock.step()
        if (type(index) is not int or not 0 <= index < len(polynomials)
                or index in selected):
            return False
        selected.add(index)
    for witness in witnesses:
        clock.step()
        # Reject negatives BEFORE any bit-clearing/bit-scanning loop.
        if (type(witness) is not int or witness < 0
                or witness.bit_length() > len(basis_indices)):
            return False
    for polynomial in polynomials:
        clock.step()
        if type(polynomial) is not frozenset:
            return False
        for mask in polynomial:
            clock.step()
            if type(mask) is not int or not 0 <= mask < 1 << 64:
                return False

    # Independently build coordinates, bounded BEFORE every insertion/shift.
    # Coordinate order here has no relationship to production's first appearance.
    coordinates: dict[int, int] = {}
    for index in basis_indices:
        clock.step()
        for mask in sorted(polynomials[index]):
            clock.step()
            if mask not in coordinates:
                if len(coordinates) >= _MAX_COORDINATES:
                    raise EngineeringLimit('verification distinct coordinate limit')
                coordinates[mask] = len(coordinates)
    high_pivots: dict[int, int] = {}
    for index in basis_indices:
        clock.step()
        row = 0
        for mask in polynomials[index]:
            clock.step()
            row |= 1 << coordinates[mask]
        while row:
            clock.step()
            pivot = row.bit_length() - 1
            if pivot not in high_pivots:
                high_pivots[pivot] = row
                break
            budget.charge('reduction_xors', 1)
            row ^= high_pivots[pivot]
        if not row:
            return False  # Includes a selected zero polynomial.
    clock.check()

    # Literal ANF reconstruction, not row/pivot/witness algebra from production.
    for polynomial, witness in zip(polynomials, witnesses, strict=True):
        clock.step()
        recovered: Polynomial = frozenset()
        remaining = witness
        while remaining:
            clock.step()
            bit = remaining & -remaining
            index = bit.bit_length() - 1
            budget.charge('reduction_xors', 1)
            recovered = _xor(recovered, polynomials[basis_indices[index]], budget, clock)
            remaining ^= bit
        if recovered != polynomial:
            return False
        progress.prefix += 1
    clock.check()
    return True


def verify_span(polynomials: tuple[frozenset[int], ...], basis_indices: tuple[int, ...],
                witnesses: tuple[int, ...], budget: Budget, *,
                checkpoint: Callable[[], None]) -> bool:
    """Return False for malformed/false certificates; propagate resource stops.

    Inputs are exact public source ANFs, in original column order. This verifies
    their span and independence, not their correspondence to an external plan.
    A caller must retain that correspondence; compression uses its own ordered
    source cache. No secret input, production reducer or hash proves this check.
    """
    return _verify(polynomials, basis_indices, witnesses, budget, _Clock(checkpoint),
                   _VerifiedProgress())


def compress_span(plan: PublicFeaturePlan, budget: Budget, *, checkpoint: Callable[[], None],
                  coordinate_limit: int = _MAX_COORDINATES) -> CompressionResult:
    """Compress an already-validated public plan with one cumulative ledger.

    COMPLETE requires the full source cache and both independent checks. Source
    term references are reserved before retaining ANY copied leaf. No truncated
    plan or exact partial rank is returned on failure. EngineeringLimit (including
    a checkpoint wall/CPU stop) and MemoryError are engineering limits; other
    ordinary exceptions are structural failures, not apparent quota exhaustion.
    KeyboardInterrupt/SystemExit deliberately propagate to the supervisor.
    """
    clock = _Clock(checkpoint)
    sources: list[Polynomial] = []
    basis: list[int] = []
    witnesses: list[int] = []
    coordinates: dict[int, int] = {}
    pivots: dict[int, tuple[int, int]] = {}
    progress = _VerifiedProgress()
    status = 'STRUCTURAL_FAILURE'
    reason: str | None = None
    exact_rank: int | None = None
    candidate_count = len(plan.features) if type(plan) is PublicFeaturePlan else 0
    try:
        clock.check()
        if type(plan) is not PublicFeaturePlan:
            raise ValueError('expected validated PublicFeaturePlan')
        if (type(coordinate_limit) is not int
                or not 0 <= coordinate_limit <= _MAX_COORDINATES):
            raise ValueError('coordinate_limit must be in 0..200000')

        term_refs = 0
        for leaf in plan.leaves:
            clock.step()
            term_refs += len(leaf)
        budget.charge('term_work', term_refs)
        leaves: list[Polynomial] = []
        for leaf in plan.leaves:
            clock.step()
            copied: set[int] = set()
            for mask in leaf:
                clock.step()
                copied.add(mask)
            leaves.append(frozenset(copied))

        for feature in plan.features:
            clock.check()
            polynomial: Polynomial = frozenset({0})  # Empty product is ONE.
            for index in feature:
                clock.step()
                polynomial = _and(polynomial, leaves[index], budget, clock)
            sources.append(polynomial)  # Ordered, complete original column cache.

            # Do not construct an unbounded union of all columns' coordinates.
            # Each column is already term-budget bounded; sorted order is fixed.
            for mask in sorted(polynomial):
                clock.step()
                if mask not in coordinates:
                    if len(coordinates) >= coordinate_limit:
                        raise EngineeringLimit('distinct coordinate limit')
                    coordinates[mask] = len(coordinates)
            row = 0
            for mask in polynomial:
                clock.step()
                row |= 1 << coordinates[mask]
            witness = 0
            while row:
                clock.step()
                pivot = (row & -row).bit_length() - 1
                if pivot not in pivots:
                    new = 1 << len(basis)
                    # All current witness bits name EARLIER basis columns;
                    # initialize this residual's new provenance bit (no row
                    # reduction and no additional witness XOR is performed).
                    combination = witness | new
                    pivots[pivot] = (row, combination)
                    basis.append(len(sources) - 1)
                    witness = new
                    break
                other, combination = pivots[pivot]
                budget.charge('reduction_xors', 1)
                row ^= other
                budget.charge('reduction_xors', 1)
                witness ^= combination
            witnesses.append(witness)
        clock.check()
        if not _verify(tuple(sources), tuple(basis), tuple(witnesses), budget, clock, progress):
            raise ValueError('independent span certificate verification failed')
        exact_rank = len(basis)
        status = ('COMPLETE_COMPRESSED' if exact_rank < candidate_count
                  else 'COMPLETE_FULL_RANK')
    except (EngineeringLimit, MemoryError) as exc:
        status = 'ENGINEERING_LIMIT'
        reason = f'{type(exc).__name__}: {exc}'
    except Exception as exc:
        status = 'STRUCTURAL_FAILURE'
        reason = f'{type(exc).__name__}: {exc}'
    return CompressionResult(
        status=status, reason=reason, candidate_count=candidate_count,
        basis_indices=tuple(basis), witnesses=tuple(witnesses), exact_rank=exact_rank,
        expanded_prefix=len(sources), eliminated_prefix=len(witnesses),
        independently_verified_prefix=progress.prefix,
        reported_rank_estimate=len(basis), coordinate_count=len(coordinates),
        work=budget.to_dict(),
    )

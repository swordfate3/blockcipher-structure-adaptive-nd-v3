"""Bounded Boolean algebra for public SIMON target cones; no key assignment API.

Mask bits 0..63 are raw MSB-first columns. Bits 64..191 are *unassigned*
formal master-key variables for the first four rounds only. No cipher import.
"""
from __future__ import annotations

from dataclasses import dataclass

Polynomial = frozenset[int]
PUBLIC_MASK = (1 << 64) - 1


class EngineeringLimit(RuntimeError):
    """No partial polynomial/plan may escape a failed resource reservation."""


@dataclass
class Budget:
    """One cumulative ledger, including rejected attempts; never auto-increases."""

    term_limit: int = 1_000_000
    pair_limit: int = 10_000_000
    reduction_limit: int = 2_000_000
    term_work: int = 0
    pairwork: int = 0
    reduction_xors: int = 0

    def __post_init__(self) -> None:
        for value, ceiling in ((self.term_limit, 1_000_000), (self.pair_limit, 10_000_000),
                               (self.reduction_limit, 2_000_000)):
            if type(value) is not int or not 0 <= value <= ceiling:
                raise ValueError('invalid symbolic quota')
        if any(v != 0 for v in (self.term_work, self.pairwork, self.reduction_xors)):
            raise ValueError('budget must start empty')

    def charge(self, resource: str, count: int) -> None:
        names = {'term_work': 'term_limit', 'pairwork': 'pair_limit',
                 'reduction_xors': 'reduction_limit'}
        if resource not in names or type(count) is not int or count < 0:
            raise ValueError('invalid charge')
        used = getattr(self, resource) + count
        setattr(self, resource, used)
        limit = getattr(self, names[resource])
        if used > limit:
            raise EngineeringLimit(f'{resource}: attempted {used} > {limit}')

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in (
            'term_work', 'pairwork', 'reduction_xors', 'term_limit', 'pair_limit',
            'reduction_limit')}


def _validate(p: Polynomial, width: int = 192) -> None:
    if type(p) is not frozenset:
        raise TypeError('ANF must be frozenset')
    if any(type(m) is not int or not 0 <= m < 1 << width for m in p):
        raise ValueError('invalid ANF mask')


def anf_xor(left: Polynomial, right: Polynomial, budget: Budget) -> Polynomial:
    _validate(left)
    _validate(right)
    budget.charge('term_work', len(left) + len(right))
    return left ^ right


def anf_and(left: Polynomial, right: Polynomial, budget: Budget) -> Polynomial:
    _validate(left)
    _validate(right)
    pairs = len(left) * len(right)
    budget.charge('pairwork', pairs)
    budget.charge('term_work', pairs)
    product: set[int] = set()
    for a in left:
        for b in right:
            term = a | b  # Boolean idempotence, then XOR parity, never OR of terms.
            if term in product:
                product.remove(term)
            else:
                product.add(term)
    return frozenset(product)


def rotate_public(p: Polynomial, shift: int, budget: Budget) -> Polynomial:
    _validate(p, 64)
    if type(shift) is not int or not 0 <= shift < 32:
        raise ValueError('shift must be in 0..31')
    budget.charge('term_work', len(p))
    result = set()
    for mask in p:
        rotated = 0
        while mask:
            bit = (mask & -mask).bit_length() - 1
            rotated |= 1 << (32 * (bit // 32) + (bit % 32 + shift) % 32)
            mask &= mask - 1
        result.add(rotated)
    return frozenset(result)


class TargetCone:
    """Memoized MSB-indexed X_r cone; X_-1 is the public low word Y_0.

    For r<=4 the first four round keys ARE the formal master-key words. There
    is no invented higher-round key schedule and no concrete zero-key oracle.
    """

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.memo: dict[tuple[int, int], Polynomial] = {}

    def bit(self, rounds: int, position: int) -> Polynomial:
        if type(rounds) is not int or not -1 <= rounds <= 4 or type(position) is not int:
            raise ValueError('formal cone supports -1..4 only')
        j = position % 32
        key = (rounds, j)
        if key not in self.memo:
            if rounds <= 0:
                self.budget.charge('term_work', 1)
                result = frozenset({1 << (j + (32 if rounds == -1 else 0))})
            else:
                nonlinear = anf_and(self.bit(rounds - 1, j + 1),
                                    self.bit(rounds - 1, j + 8), self.budget)
                result = anf_xor(self.bit(rounds - 2, j), nonlinear, self.budget)
                result = anf_xor(result, self.bit(rounds - 1, j + 2), self.budget)
                result = anf_xor(result, frozenset({1 << (64 + 32*(rounds-1) + 31-j)}),
                                 self.budget)
            self.memo[key] = result
        return self.memo[key]


@dataclass(frozen=True)
class CoefficientBasis:
    polynomials: tuple[Polynomial, ...]
    groups: tuple[tuple[int, Polynomial], ...]
    witnesses: tuple[int, ...]


def coefficient_basis(terms: Polynomial, budget: Budget) -> CoefficientBasis:
    """Deterministic original-generator subset plus exact reconstruction witnesses.

    Ascending formal-key mask, then least-monomial pivot; no fitted/sample rank.
    The witness for every h_q certifies F(P,K)=XOR_q K^q XOR_i w_qi b_i(P).
    """
    _validate(terms)
    budget.charge('term_work', len(terms))
    grouped: dict[int, set[int]] = {}
    for term in sorted(terms):
        grouped.setdefault(term >> 64, set()).add(term & PUBLIC_MASK)
    groups = tuple((k, frozenset(grouped[k])) for k in sorted(grouped))
    monomials = sorted({m for _, p in groups for m in p})
    coordinates = {m: i for i, m in enumerate(monomials)}
    pivots: dict[int, tuple[int, int]] = {}
    basis: list[Polynomial] = []
    witnesses = []
    for _, polynomial in groups:
        row = sum(1 << coordinates[m] for m in polynomial)
        witness = 0
        while row:
            pivot = (row & -row).bit_length() - 1
            if pivot not in pivots:
                new = 1 << len(basis)
                pivots[pivot] = (row, witness ^ new)
                basis.append(polynomial)
                witness = new
                break
            budget.charge('reduction_xors', 1)
            other, combination = pivots[pivot]
            row ^= other
            witness ^= combination
        witnesses.append(witness)
    return CoefficientBasis(tuple(basis), groups, tuple(witnesses))


def verify_certificate(terms: Polynomial, basis: CoefficientBasis, budget: Budget) -> bool:
    """Reconstruct every symbolic coefficient, without evaluating formal keys."""
    _validate(terms)
    reconstructed: set[int] = set()
    if len(basis.groups) != len(basis.witnesses):
        return False
    for (key, polynomial), witness in zip(basis.groups, basis.witnesses, strict=True):
        if not 0 <= witness < 1 << len(basis.polynomials):
            return False
        recovered: Polynomial = frozenset()
        for i, leaf in enumerate(basis.polynomials):
            if witness >> i & 1:
                budget.charge('reduction_xors', 1)
                recovered = anf_xor(recovered, leaf, budget)
        if recovered != polynomial:
            return False
        budget.charge('term_work', len(polynomial))
        reconstructed.update((key << 64) | m for m in polynomial)
    return frozenset(reconstructed) == terms

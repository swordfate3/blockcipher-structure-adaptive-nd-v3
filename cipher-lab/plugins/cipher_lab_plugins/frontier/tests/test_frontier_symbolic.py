"""Toy-only public algebra tests. Never run the full frontier builder here."""
from __future__ import annotations

import itertools

import pytest

from cipher_lab_plugins.frontier.simon_anf import (
    Budget,
    EngineeringLimit,
    TargetCone,
    anf_and,
    anf_xor,
    coefficient_basis,
    rotate_public,
    verify_certificate,
)


def test_boolean_ring_not_set_union() -> None:
    budget = Budget()
    a, b = frozenset({1}), frozenset({2})
    assert anf_and(a, a, budget) == a
    p = anf_xor(a, b, budget)
    assert anf_and(p, p, budget) == p
    assert anf_and(frozenset({0, 1}), a, budget) == frozenset()
    assert anf_xor(p, p, budget) == frozenset()


def test_attempt_is_charged_before_failure_and_no_truncation() -> None:
    budget = Budget(pair_limit=3)
    with pytest.raises(EngineeringLimit, match='pairwork'):
        anf_and(frozenset({1, 2}), frozenset({4, 8}), budget)
    assert budget.pairwork == 4
    with pytest.raises(EngineeringLimit):
        anf_and(frozenset({1}), frozenset({2}), budget)
    assert budget.pairwork == 5


def test_exact_coefficient_span_and_reconstruction_certificate() -> None:
    # h0=x+y, h1=x, h2=y: rank two, not three. No formal key assignment.
    terms = frozenset({1, 2, (1 << 64) | 1, (1 << 65) | 2})
    budget = Budget()
    basis = coefficient_basis(terms, budget)
    assert len(basis.polynomials) == 2
    assert verify_certificate(terms, basis, budget)
    for witness, (_, polynomial) in zip(basis.witnesses, basis.groups, strict=True):
        reconstructed: set[int] = set()
        for i, p in enumerate(basis.polynomials):
            if witness >> i & 1:
                reconstructed.symmetric_difference_update(p)
        assert frozenset(reconstructed) == polynomial
    assert budget.reduction_xors > 0


def test_rotation_msb_wrap_is_within_each_word() -> None:
    assert rotate_public(frozenset({1 << 31, 1 << 63}), 1, Budget()) == frozenset({1, 1 << 32})
    cone = TargetCone(Budget())
    # The first round MSB has raw y0, x1*x8, x2 and unassigned formal k[31].
    assert cone.bit(1, 0) == frozenset({1 << 32, (1 << 1) | (1 << 8),
                                             1 << 2, 1 << (64 + 31)})
    assert cone.bit(0, 32) == cone.bit(0, 0)
    with pytest.raises(ValueError):
        cone.bit(5, 0)


@pytest.mark.parametrize('bad', [True, -1, 1 << 192])
def test_symbolic_mask_rejects_invalid(bad: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        anf_xor(frozenset({bad}), frozenset(), Budget())


def test_toy_span_exhaustive_public_only() -> None:
    terms = frozenset({1, 2, (1 << 64) | 1, (1 << 65) | 2})
    basis = coefficient_basis(terms, Budget())
    for bits in itertools.product((0, 1), repeat=2):
        assignment = bits[0] | bits[1] << 1
        for witness, (_, p) in zip(basis.witnesses, basis.groups, strict=True):
            direct = sum((m & assignment) == m for m in p) % 2
            recovered = sum(sum((m & assignment) == m for m in leaf) % 2
                            for i, leaf in enumerate(basis.polynomials) if witness >> i & 1) % 2
            assert direct == recovered

"""Public-only toys: no fixture, cipher, RNG, Torch or training imports."""
from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from itertools import combinations, product

import pytest

from cipher_lab_plugins.frontier import span_compression as span_compression
from cipher_lab_plugins.frontier.simon_anf import Budget, EngineeringLimit
from cipher_lab_plugins.frontier.simon_frontier import PublicFeaturePlan
from cipher_lab_plugins.frontier.span_compression import compress_span, verify_span

_STATIC_POPEN = subprocess.Popen


def noop() -> None:
    pass


def make_plan(leaves: tuple[tuple[int, ...], ...],
              features: tuple[tuple[int, ...], ...]) -> PublicFeaturePlan:
    return PublicFeaturePlan(5, 0, 'simon-r5-factored-v1', leaves, features)


def toy() -> PublicFeaturePlan:
    # Canonical order: 1, x, x(x+y), x+y, y.
    return make_plan(((1,), (1, 2), (2,)), ((), (0,), (0, 1), (1,), (2,)))


def literal_expand(plan: PublicFeaturePlan) -> tuple[frozenset[int], ...]:
    """Independent literal Cartesian expansion, never production Boolean helpers."""
    columns = []
    for feature in plan.features:
        counts: dict[int, int] = {}
        for terms in product(*(plan.leaves[i] for i in feature)):
            mask = 0
            for term in terms:
                mask |= term
            counts[mask] = counts.get(mask, 0) + 1
        columns.append(frozenset(mask for mask, count in counts.items() if count % 2))
    return tuple(columns)


def assert_reconstruction(polynomials: tuple[frozenset[int], ...],
                          basis: tuple[int, ...], witnesses: tuple[int, ...]) -> None:
    for expected, witness in zip(polynomials, witnesses, strict=True):
        terms: set[int] = set()
        for bit, index in enumerate(basis):
            if witness & (1 << bit):
                terms.symmetric_difference_update(polynomials[index])
        assert terms == set(expected)


def test_boolean_span_exact_witnesses_and_literal_expansion() -> None:
    plan = toy()
    columns = literal_expand(plan)
    assert columns == (frozenset({0}), frozenset({1}), frozenset({1, 3}),
                       frozenset({1, 2}), frozenset({2}))
    budget = Budget()
    result = compress_span(plan, budget, checkpoint=noop)
    assert result.status == 'COMPLETE_COMPRESSED'
    assert result.candidate_count == 5
    assert result.exact_rank == result.reported_rank_estimate == 4
    assert result.basis_indices == (0, 1, 2, 3)
    assert result.witnesses == (1, 2, 4, 8, 10)
    assert result.coordinate_count == 4
    assert (result.expanded_prefix, result.eliminated_prefix,
            result.independently_verified_prefix) == (5, 5, 5)
    assert_reconstruction(columns, result.basis_indices, result.witnesses)
    assert verify_span(columns, result.basis_indices, result.witnesses,
                       Budget(), checkpoint=noop)
    serialized = result.to_dict()
    assert serialized['witnesses'] == ['0x1', '0x2', '0x4', '0x8', '0xa']
    assert serialized['work'] == budget.to_dict()
    # Source 4 + production products 7 + independent spanning terms 10.
    assert (budget.term_work, budget.pairwork, budget.reduction_xors) == (21, 7, 12)


def test_zero_function_and_boolean_square_cancellation() -> None:
    # x(1+x)=0; (x+y)(1+x+y)=0 because every Boolean function squares to itself.
    plan = make_plan(((0, 1), (0, 1, 2), (1,), (1, 2)),
                     ((), (0,), (0, 2), (1,), (1, 3), (2,), (3,)))
    columns = literal_expand(plan)
    assert columns[2] == columns[4] == frozenset()
    result = compress_span(plan, Budget(), checkpoint=noop)
    assert result.status == 'COMPLETE_COMPRESSED'
    assert result.exact_rank == 3
    assert 2 not in result.basis_indices and 4 not in result.basis_indices
    assert result.witnesses[2] == result.witnesses[4] == 0
    assert_reconstruction(columns, result.basis_indices, result.witnesses)


def test_full_rank_and_constant_counted() -> None:
    result = compress_span(make_plan(((1,), (2,)), ((), (0,), (1,))),
                           Budget(), checkpoint=noop)
    assert result.status == 'COMPLETE_FULL_RANK'
    assert result.exact_rank == 3
    constant = compress_span(make_plan((), ((),)), Budget(), checkpoint=noop)
    assert constant.status == 'COMPLETE_FULL_RANK'
    assert constant.exact_rank == 1
    assert verify_span((frozenset(),), (), (0,), Budget(), checkpoint=noop)


def test_spanning_alone_cannot_certify_dependent_basis() -> None:
    columns = (frozenset({1}), frozenset({2}), frozenset({1, 2}))
    assert not verify_span(columns, (0, 1, 2), (1, 2, 4), Budget(), checkpoint=noop)
    assert verify_span(columns, (0, 1), (1, 2, 3), Budget(), checkpoint=noop)
    assert not verify_span((frozenset(),), (0,), (1,), Budget(), checkpoint=noop)


def test_false_independent_verification_is_structural_not_limit(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate an internal certificate mismatch without changing production code.
    monkeypatch.setattr(span_compression, '_verify', lambda *args: False)
    result = compress_span(toy(), Budget(), checkpoint=noop)
    assert result.status == 'STRUCTURAL_FAILURE'
    assert result.exact_rank is None and result.reported_rank_estimate == 4
    assert result.expanded_prefix == result.eliminated_prefix == 5
    assert result.independently_verified_prefix == 0
    assert 'certificate verification failed' in str(result.reason)


@pytest.mark.parametrize(('basis', 'witnesses'), [
    ((0, 0), (1, 2, 3)), ((0, 3), (1, 2, 3)), ((-1, 1), (1, 2, 3)),
    ((0, 1), (-1, 2, 3)), ((0, 1), (1 << 10000, 2, 3)),
    ((0, 1), (1, 2)), ((0, 1), (1, 2, 1)),
    ((True, 1), (1, 2, 3)), ((0, 1), (True, 2, 3)),
])
def test_certificate_tampering_rejected(basis: tuple[int, ...],
                                       witnesses: tuple[int, ...]) -> None:
    assert not verify_span((frozenset({1}), frozenset({2}), frozenset({1, 2})),
                           basis, witnesses, Budget(), checkpoint=noop)


def test_public_mask_validation_in_verifier() -> None:
    for mask in (-1, 1 << 64):
        assert not verify_span((frozenset({mask}),), (0,), (1,),
                               Budget(), checkpoint=noop)


@pytest.mark.parametrize(('budget', 'resource', 'attempt'), [
    (Budget(term_limit=0), 'term_work', 4),
    (Budget(pair_limit=0), 'pairwork', 1),
])
def test_source_and_multiplication_attempts_are_charged(
        budget: Budget, resource: str, attempt: int) -> None:
    result = compress_span(toy(), budget, checkpoint=noop)
    assert result.status == 'ENGINEERING_LIMIT'
    assert result.exact_rank is None
    assert result.work[resource] == attempt
    assert getattr(budget, resource) == attempt
    assert result.reported_rank_estimate is not None


def test_multiplication_pair_reserved_before_term_rejection() -> None:
    # Source=2 terms; first 1*2 multiplication reserves pairs and then fails terms.
    budget = Budget(term_limit=2)
    result = compress_span(make_plan(((1, 2),), ((0,),)), budget, checkpoint=noop)
    assert result.status == 'ENGINEERING_LIMIT'
    assert (budget.pairwork, budget.term_work) == (2, 4)
    assert result.expanded_prefix == result.eliminated_prefix == 0


def test_reduction_attempts_and_partial_witness_progress() -> None:
    # x(x+y) becomes a new basis column only after reducing by x.
    budget = Budget(reduction_limit=1)
    result = compress_span(toy(), budget, checkpoint=noop)
    assert result.status == 'ENGINEERING_LIMIT'
    assert budget.reduction_xors == 2  # row XOR then separately charged witness XOR
    assert (result.expanded_prefix, result.eliminated_prefix) == (3, 2)
    assert result.basis_indices == (0, 1)
    assert result.witnesses == (1, 2)
    assert result.exact_rank is None


def test_coordinate_limit_checks_before_insert_and_shift() -> None:
    result = compress_span(toy(), Budget(), checkpoint=noop, coordinate_limit=1)
    assert result.status == 'ENGINEERING_LIMIT'
    assert result.coordinate_count == 1
    assert (result.expanded_prefix, result.eliminated_prefix) == (2, 1)
    assert result.exact_rank is None


def test_zero_coordinate_limit_and_invalid_coordinate_ceiling() -> None:
    stopped = compress_span(make_plan((), ((),)), Budget(), checkpoint=noop, coordinate_limit=0)
    assert stopped.status == 'ENGINEERING_LIMIT' and stopped.coordinate_count == 0
    invalid = compress_span(toy(), Budget(), checkpoint=noop, coordinate_limit=200_001)
    assert invalid.status == 'STRUCTURAL_FAILURE'
    assert invalid.exact_rank is None


def test_independent_verification_continues_shared_ledger() -> None:
    # One x: leaf retention 1 + 1*1 multiplication 1 + spanning XOR 1.
    budget = Budget(term_limit=2)
    result = compress_span(make_plan(((1,),), ((0,),)), budget, checkpoint=noop)
    assert result.status == 'ENGINEERING_LIMIT'
    assert (result.expanded_prefix, result.eliminated_prefix,
            result.independently_verified_prefix) == (1, 1, 0)
    assert result.exact_rank is None and result.reported_rank_estimate == 1
    assert budget.term_work == 3 and budget.reduction_xors == 1


def test_verifier_reserves_reconstruction_and_independence_xors() -> None:
    columns = (frozenset({1}), frozenset({1, 2}))
    budget = Budget()
    assert verify_span(columns, (0, 1), (1, 2), budget, checkpoint=noop)
    assert budget.term_work == 3
    assert budget.reduction_xors == 2  # highest pivots differ: no independence XOR
    # Highest pivots coincide here: independent verifier must pay one further XOR.
    columns = (frozenset({1, 2}), frozenset({2}))
    budget = Budget(reduction_limit=2)
    with pytest.raises(EngineeringLimit, match='reduction_xors'):
        verify_span(columns, (0, 1), (1, 2), budget, checkpoint=noop)
    assert budget.reduction_xors == 3


@pytest.mark.parametrize(('exception', 'status'), [
    (EngineeringLimit('wall budget'), 'ENGINEERING_LIMIT'),
    (RuntimeError('broken checkpoint'), 'STRUCTURAL_FAILURE'),
    (MemoryError('allocation'), 'ENGINEERING_LIMIT'),
])
def test_checkpoint_failures_classified(exception: Exception, status: str) -> None:
    def stop() -> None:
        raise exception
    result = compress_span(toy(), Budget(), checkpoint=stop)
    assert result.status == status
    assert result.exact_rank is None
    assert result.expanded_prefix == result.eliminated_prefix == 0
    assert result.independently_verified_prefix == 0
    assert str(exception) in str(result.reason)


def test_checkpoint_inside_large_cartesian_product() -> None:
    # 70x70 public terms: synthetic only, max source mask <2**64.
    plan = make_plan((tuple(range(1, 71)), tuple(range(128, 198))), ((0, 1),))
    budget = Budget()
    checkpoints_after_reservation = 0

    def stop_during_product() -> None:
        nonlocal checkpoints_after_reservation
        if budget.pairwork >= 4970:  # 1*70 plus 70*70, reserved before loop
            checkpoints_after_reservation += 1
            raise EngineeringLimit('inner product checkpoint')

    result = compress_span(plan, budget, checkpoint=stop_during_product)
    assert result.status == 'ENGINEERING_LIMIT'
    assert checkpoints_after_reservation == 1
    assert result.expanded_prefix == 0  # callback was not deferred to feature boundary
    assert budget.pairwork == 4970
    assert budget.term_work == 5110


def test_verification_checkpoint_propagates_not_false() -> None:
    def stop() -> None:
        raise EngineeringLimit('verifier wall')
    with pytest.raises(EngineeringLimit, match='verifier wall'):
        verify_span((frozenset({1}),), (0,), (1,), Budget(), checkpoint=stop)


def test_result_work_snapshot_and_hex_no_decimal_integer_limit() -> None:
    result = compress_span(toy(), Budget(), checkpoint=noop)
    before = dict(result.work)
    # A frozen result must not expose a mutable budget ledger.
    with pytest.raises(TypeError):
        result.work['term_work'] = 9  # type: ignore[index]
    assert dict(result.work) == before
    big = replace(result, witnesses=(1 << 15000,))
    encoded = json.dumps(big.to_dict())
    assert '0x1' in encoded


def test_no_scientific_modules_imported() -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[4]
    script = """
import sys
from cipher_lab_plugins.frontier.span_compression import compress_span
forbidden = ('torch', 'numpy', 'cipher_lab.ciphers', 'cipher_lab.models',
             'cipher_lab.training', 'cipher_lab.data')
assert not any(name == prefix or name.startswith(prefix + '.')
               for name in sys.modules for prefix in forbidden)
"""
    # This static-only subprocess is deliberately exempt from the scientific guard.
    result = _STATIC_POPEN([sys.executable, '-c', script], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join((str(root/'src'), str(root/'plugins'))),
                 PYTHONDONTWRITEBYTECODE='1'))
    _, stderr = result.communicate()
    assert result.returncode == 0, stderr


def test_partial_independent_prefix_does_not_certify_overall_rank() -> None:
    budget = Budget(term_limit=20)
    result = compress_span(toy(), budget, checkpoint=noop)
    assert result.status == 'ENGINEERING_LIMIT'
    assert result.expanded_prefix == result.eliminated_prefix == 5
    assert result.independently_verified_prefix == 4
    assert result.exact_rank is None and result.reported_rank_estimate == 4
    assert budget.term_work == 21  # final reconstruction attempt retained


def test_public_literal_family_cross_checks_all_witnesses() -> None:
    # All <=3-factor subsets of four public leaves, including zero products.
    leaves = ((0, 1), (0, 1, 2), (1,), (1, 2))
    features = tuple(sorted(feature for arity in range(4)
                            for feature in combinations(range(len(leaves)), arity)))
    plan = make_plan(leaves, features)
    result = compress_span(plan, Budget(), checkpoint=noop)
    assert result.status == 'COMPLETE_COMPRESSED'
    assert result.exact_rank == 4
    columns = literal_expand(plan)
    assert_reconstruction(columns, result.basis_indices, result.witnesses)
    assert verify_span(columns, result.basis_indices, result.witnesses,
                       Budget(), checkpoint=noop)

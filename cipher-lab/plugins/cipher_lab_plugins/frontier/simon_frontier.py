"""Versioned, bounded public plans and exact frontier inclusion construction.

No Torch/NumPy, cipher, actual key, labels, sample rank or training dependency.
Hashes bind canonical syntax, not a claim of mathematical provenance for an
arbitrary externally authored plan. Only build_frontier_plans supplies the
separate containment certificates used for scientific admission.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from itertools import product
from typing import cast

from .simon_anf import (
    Budget,
    EngineeringLimit,
    Polynomial,
    TargetCone,
    anf_and,
    anf_xor,
    coefficient_basis,
    rotate_public,
    verify_certificate,
)

MAX_PLAN_BYTES = 16 * 1024 * 1024
MAX_DIMENSION = 8192
MAX_TERM_REFS = 100_000
_REPRESENTATIONS = {4: 'simon-r4-residual89-v1', 5: 'simon-r5-factored-v1',
                    6: 'simon-r6-factored-v1'}


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def _sha(value: object) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


@dataclass(frozen=True)
class FeatureIdentity:
    rounds: int
    target_bit: int
    representation: str
    dimension: int
    order_sha256: str
    plan_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            'rounds', 'target_bit', 'representation', 'dimension',
            'order_sha256', 'plan_sha256')}


@dataclass(frozen=True)
class PublicFeaturePlan:
    rounds: int
    target_bit: int
    representation: str
    leaves: tuple[tuple[int, ...], ...]
    features: tuple[tuple[int, ...], ...]
    _identity: FeatureIdentity = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (type(self.rounds) is not int or self.rounds not in _REPRESENTATIONS
                or type(self.target_bit) is not int or self.target_bit != 0
                or type(self.representation) is not str
                or self.representation != _REPRESENTATIONS[self.rounds]):
            raise ValueError('invalid public plan round/target/representation')
        if (type(self.leaves) is not tuple or type(self.features) is not tuple
                or not 1 <= len(self.features) <= MAX_DIMENSION
                or len(self.leaves) > MAX_TERM_REFS):
            raise ValueError('plan dimension/leaf quota')
        term_refs = 0
        previous: tuple[int, ...] | None = None
        for leaf in self.leaves:
            if type(leaf) is not tuple or not leaf or leaf == (0,):
                raise ValueError('leaf must be a nonzero nonconstant polynomial')
            term_refs += len(leaf)
            if term_refs > MAX_TERM_REFS:
                raise ValueError('leaf term reference quota')
            if any(type(m) is not int or not 0 <= m < 1 << 64 for m in leaf):
                raise ValueError('public leaves require uint64 masks, no formal key')
            if any(a >= b for a, b in zip(leaf, leaf[1:])):
                raise ValueError('leaf masks must be strictly ordered')
            if previous is not None and previous >= leaf:
                raise ValueError('leaves must be unique and lexicographically ordered')
            previous = leaf
        used: set[int] = set()
        previous = None
        for feature in self.features:
            if type(feature) is not tuple or len(feature) > 3:
                raise ValueError('feature AND arity must be 0..3')
            if any(type(i) is not int or not 0 <= i < len(self.leaves) for i in feature):
                raise ValueError('invalid feature leaf index')
            if any(a >= b for a, b in zip(feature, feature[1:])):
                raise ValueError('duplicate/unordered feature factors')
            if previous is not None and previous >= feature:
                raise ValueError('features must be unique and lexicographically ordered')
            previous = feature
            used.update(feature)
        if used != set(range(len(self.leaves))):
            raise ValueError('unused public leaf')
        if self.rounds == 4 and (len(self.features) != 89
                                or any(len(f) > 1 for f in self.features)):
            raise ValueError('r4 requires residual89 linear coordinates')
        body = self._body()
        # Unique encoding of syntax; self-referential identity is excluded.
        order = _sha({'leaves': self.leaves, 'features': self.features})
        identity = FeatureIdentity(self.rounds, self.target_bit, self.representation,
                                   len(self.features), order, _sha(body))
        object.__setattr__(self, '_identity', identity)
        if len(_json({**body, 'identity': identity.to_dict()})) > MAX_PLAN_BYTES:
            raise ValueError('plan byte quota')

    def _body(self) -> dict[str, object]:
        return {'schema': 1, 'rounds': self.rounds, 'target_bit': self.target_bit,
                'representation': self.representation, 'leaves': self.leaves,
                'features': self.features}

    @property
    def dimension(self) -> int:
        return len(self.features)

    @property
    def identity(self) -> FeatureIdentity:
        return self._identity

    @property
    def term_references(self) -> int:
        return sum(map(len, self.leaves))


def encode_plan(plan: PublicFeaturePlan) -> bytes:
    if type(plan) is not PublicFeaturePlan:
        raise ValueError('expected validated PublicFeaturePlan')
    return _json({**plan._body(), 'identity': plan.identity.to_dict()})


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _integer(text: str) -> int:
    if len(text) > 20:
        raise ValueError('oversized JSON integer')
    return int(text)


def _noninteger(text: str) -> object:
    raise ValueError('plan contains a noninteger number')


def decode_plan(content: bytes) -> PublicFeaturePlan:
    if type(content) is not bytes or not 0 < len(content) <= MAX_PLAN_BYTES:
        raise ValueError('invalid plan bytes/size')
    try:
        value = json.loads(content, object_pairs_hook=_pairs, parse_int=_integer,
                           parse_float=_noninteger, parse_constant=_noninteger)
        if type(value) is not dict or set(value) != {
            'schema', 'rounds', 'target_bit', 'representation', 'leaves', 'features', 'identity'
        }:
            raise ValueError('invalid plan fields')
        if type(value['schema']) is not int or value['schema'] != 1:
            raise ValueError('invalid plan schema')
        leaves, features = value['leaves'], value['features']
        if (type(leaves) is not list or type(features) is not list
                or len(leaves) > MAX_TERM_REFS or len(features) > MAX_DIMENSION
                or any(type(row) is not list for row in leaves + features)):
            raise ValueError('invalid plan arrays/quotas')
        plan = PublicFeaturePlan(value['rounds'], value['target_bit'], value['representation'],
                                 tuple(tuple(row) for row in leaves),
                                 tuple(tuple(row) for row in features))
        # Canonical byte comparison also rejects bool-as-int identity, whitespace,
        # noncanonical escaping/order, unknown identity fields and digest mismatch.
        if encode_plan(plan) != content:
            raise ValueError('noncanonical plan or identity mismatch')
        return plan
    except (TypeError, KeyError, RecursionError, UnicodeError, OverflowError) as exc:
        raise ValueError('malformed public plan') from exc


def factored_candidates(groups: tuple[tuple[int | None, ...], ...],
                        budget: Budget) -> tuple[tuple[int, ...], ...]:
    """Complete r5 containing candidates; None is 1, squares become one leaf.

    This is a union of *coordinates*, not a polynomial sum: duplicate columns
    merge, while cancellation of terms inside each ANF is always XOR.
    """
    if len(groups) != 4:
        raise ValueError('need r4pos1/8/2 direct and r3pos0 direct groups')
    result: set[tuple[int, ...]] = set()
    for a, b in product(groups[0], groups[1]):
        budget.charge('pairwork', 1)
        result.add(tuple(sorted({i for i in (a, b) if i is not None})))
    for group in groups[2:]:
        for i in group:
            budget.charge('term_work', 1)
            result.add(() if i is None else (i,))
    result.add(())
    return tuple(sorted(result))


def _plan(rounds: int, basis: tuple[Polynomial, ...]) -> PublicFeaturePlan:
    leaves = tuple(sorted(tuple(sorted(p)) for p in basis if p != frozenset({0})))
    indices = {leaf: i for i, leaf in enumerate(leaves)}
    features = tuple(sorted(() if p == frozenset({0}) else (indices[tuple(sorted(p))],)
                            for p in basis))
    return PublicFeaturePlan(rounds, 0, _REPRESENTATIONS[rounds], leaves, features)


def _rotated_groups(spec: tuple[tuple[tuple[Polynomial, ...], int], ...], budget: Budget
                    ) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int | None, ...], ...]]:
    polynomials = tuple(tuple(rotate_public(p, shift, budget) for p in basis)
                        for basis, shift in spec)
    leaves = tuple(sorted({tuple(sorted(p)) for group in polynomials for p in group
                           if p != frozenset({0})}))
    indices = {p: i for i, p in enumerate(leaves)}
    groups = tuple(tuple(None if p == frozenset({0}) else indices[tuple(sorted(p))]
                         for p in group) for group in polynomials)
    return leaves, groups


def r6_formula_certificate(budget: Budget) -> dict[str, object]:
    """Exact Boolean identity over abstract public-span *functions*, no states.

    U_j denotes an arbitrary function in rotated direct-r4 space, V_j in
    rotated direct-r3. d/e/f/h are unassigned coefficient constants (NOT keys).
    X6=(U2 U9+U3+V1+d)(U9 U16+U10+V8+e)+U3 U10+U4+V2+U0+f+h.
    Shared U9 squares away and U3 U10 cancels. Expanding symbolic group names
    certifies degree <=3 before any leaf product expansion.
    """
    names = ('U2', 'U9', 'U16', 'U3', 'U10', 'U4', 'U0', 'V1', 'V8', 'V2',
             'd', 'e', 'f', 'h')
    symbols = {name: frozenset({1 << i}) for i, name in enumerate(names)}
    def mul(a: Polynomial, b: Polynomial) -> Polynomial:
        return anf_and(a, b, budget)
    def add(*args: Polynomial) -> Polynomial:
        out: Polynomial = frozenset()
        for arg in args:
            out = anf_xor(out, arg, budget)
        return out
    s = symbols
    left = add(mul(s['U2'], s['U9']), s['U3'], s['V1'], s['d'])
    right = add(mul(s['U9'], s['U16']), s['U10'], s['V8'], s['e'])
    original = add(mul(left, right), mul(s['U3'], s['U10']), s['U4'], s['V2'],
                   s['U0'], s['f'], s['h'])
    # Independent expected expansion (16 product terms minus shared cancellation,
    # plus five linear terms), used as a complete identity, not sampled rows.
    factors = (('U2', 'U9', 'U16'), ('U2', 'U9', 'U10'), ('U2', 'U9', 'V8'),
               ('U2', 'U9', 'e'), ('U3', 'U9', 'U16'), ('U3', 'V8'), ('U3', 'e'),
               ('V1', 'U9', 'U16'), ('V1', 'U10'), ('V1', 'V8'), ('V1', 'e'),
               ('d', 'U9', 'U16'), ('d', 'U10'), ('d', 'V8'), ('d', 'e'),
               ('U4',), ('V2',), ('U0',), ('f',), ('h',))
    expected = frozenset(sum(1 << names.index(name) for name in group) for group in factors)
    if original != expected:
        raise RuntimeError('r6 shared Boolean identity mismatch')
    return {'equivalent': True, 'maximum_factors': max(m.bit_count() for m in original),
            'shared_square_reduced': True, 'shared_u3_u10_cancelled': True,
            'variables': names, 'monomials': tuple(sorted(original)),
            'sha256': _sha({'variables': names, 'monomials': sorted(original)}),
            'meaning': 'unassigned containing-function coefficients, not actual states/keys'}


def build_frontier_plans() -> tuple[tuple[PublicFeaturePlan, ...], dict[str, object]]:
    """One shared bounded construction. External supervisor accounts encode/decode too.

    No retry or higher quotas. No cached full build in tests. A dimension mismatch
    is a structural failure, not an invitation to alter the basis to match history.
    """
    wall, cpu = time.monotonic(), time.process_time()
    budget = Budget()
    plans: list[PublicFeaturePlan] = []
    rounds: dict[str, object] = {}
    audit: dict[str, object] = {'rounds': rounds, 'science_calls': 0,
                              'budget_scope': 'all rounds, no reset'}
    try:
        cone = TargetCone(budget)
        terms4 = cone.bit(4, 0)
        terms3 = cone.bit(3, 0)
        residual = frozenset(t for t in terms4 if t >> 64)
        b4 = coefficient_basis(terms4, budget)
        br = coefficient_basis(residual, budget)
        b3 = coefficient_basis(terms3, budget)
        for terms, basis in ((terms4, b4), (residual, br), (terms3, b3)):
            if not verify_certificate(terms, basis, budget):
                raise RuntimeError('symbolic coefficient reconstruction failed')
        dimensions = (len(b4.polynomials), len(br.polynomials), len(b3.polynomials))
        if dimensions != (90, 89, 14):
            raise RuntimeError(f'structural dimensions mismatch: {dimensions}')
        r4 = _plan(4, br.polynomials)
        plans.append(r4)
        certificates = [
            {'groups': [(k, sorted(p)) for k, p in b.groups],
             'basis': [sorted(p) for p in b.polynomials], 'witnesses': b.witnesses}
            for b in (b4, br, b3)]
        rounds['4'] = {'status': 'READY', 'identity': r4.identity.to_dict(),
                       'direct_dimension': dimensions[0], 'residual_dimension': dimensions[1],
                       'r3_direct_dimension': dimensions[2], 'symbolic_terms': len(terms4),
                       'residual_terms': len(residual), 'r3_terms': len(terms3),
                       'coefficient_groups': [len(b.groups) for b in (b4, br, b3)],
                       'certificate': 'every formal key coefficient exactly reconstructed',
                       'certificate_sha256': _sha(certificates),
                       'certificates_direct4_residual4_direct3': certificates,
                       'term_references': r4.term_references}
        if time.monotonic() - wall > 300 or time.process_time() - cpu > 300:
            raise EngineeringLimit('shared construction time')
        leaves, groups = _rotated_groups(((b4.polynomials, 1), (b4.polynomials, 8),
                                          (b4.polynomials, 2), (b3.polynomials, 0)), budget)
        features = factored_candidates(groups, budget)
        info5: dict[str, object] = {
            'candidate_dimension': len(features), 'leaf_count': len(leaves),
            'term_references': sum(map(len, leaves)),
            'unique_public_monomials': len({m for leaf in leaves for m in leaf}),
            'group_sizes': tuple(map(len, groups)), 'groups': groups,
            'candidate_order_sha256': _sha({'leaves': leaves, 'features': features}),
            'certificate': 'r5=R4[1]*R4[8] XOR R4[2] XOR R3[0] XOR constant; '
                           'complete rotated direct spans; constants and squares normalized; '
                           'reference is contained, hence residual is contained',
            'rank_claim': 'syntactic containing candidates, not exact function-space rank'}
        rounds['5'] = info5
        if len(features) > MAX_DIMENSION or sum(map(len, leaves)) > MAX_TERM_REFS:
            info5.update(status='ENGINEERING_LIMIT', reason='dimension/term quota; no truncation')
        else:
            r5 = PublicFeaturePlan(5, 0, _REPRESENTATIONS[5], leaves, features)
            plans.append(r5)
            info5.update(status='READY', identity=r5.identity.to_dict())
        certificate6 = r6_formula_certificate(budget)
        # A disjoint-box subset of cubic U2*U9*U16 candidates alone exceeds
        # input quota. Count its Cartesian box analytically, without enumerating
        # even that subset. This is a syntactic candidate lower bound, not rank.
        leaves6, groups6 = _rotated_groups(tuple((b4.polynomials, shift)
                                                for shift in (2, 9, 16)), budget)
        exclusive = []
        for i, group in enumerate(groups6):
            other = set().union(*(set(g) for j, g in enumerate(groups6) if j != i))
            exclusive.append(tuple(sorted(cast(set[int], set(group) - other - {None}))))
        lower_bound = 1
        for group in exclusive:
            lower_bound *= len(group)
        rounds['6'] = {
            'status': 'ENGINEERING_LIMIT', 'certificate': certificate6,
            'cubic_exclusive_group_sizes': tuple(map(len, exclusive)),
            'candidate_lower_bound': lower_bound, 'input_dimension_limit': MAX_DIMENSION,
            'raw_historical_candidate_count': 2337180,
            'historical_count_is_not_recomputed_or_used_as_rank': True,
            'shared_leaf_count_for_cubic_evidence': len(leaves6),
            'reason': 'complete raw syntax exceeds 8192; no certified exact compression '
                      'within this implementation/budget; no plan emitted, no truncation',
            'lower_bound_is_exact_function_rank': False,
            'exclusive_group_sha256': _sha(exclusive)}
        if lower_bound <= MAX_DIMENSION:
            raise RuntimeError('r6 excess-candidate certificate did not establish bound')
    except EngineeringLimit as exc:
        audit['status'] = 'ENGINEERING_LIMIT'
        audit['reason'] = str(exc)
        for r in (4, 5, 6):
            rounds.setdefault(str(r), {'status': 'ENGINEERING_LIMIT', 'reason': str(exc)})
    except RuntimeError as exc:
        audit['status'] = 'STRUCTURAL_FAILURE'
        audit['reason'] = str(exc)
        plans.clear()
    audit.setdefault('status', 'COMPLETE')
    audit.update(work=budget.to_dict(), construction_wall=time.monotonic() - wall,
                 construction_cpu=time.process_time() - cpu)
    return tuple(plans), audit

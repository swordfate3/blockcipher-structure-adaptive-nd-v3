"""Small public plans only; full construction is a separately budgeted audit."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest

from cipher_lab_plugins.frontier.simon_anf import Budget
from cipher_lab_plugins.frontier.simon_frontier import (
    MAX_PLAN_BYTES,
    PublicFeaturePlan,
    decode_plan,
    encode_plan,
    factored_candidates,
    r6_formula_certificate,
)

if TYPE_CHECKING:
    from torch import Tensor


def test_stdlib_plan_import_never_attempts_torch_or_numpy() -> None:
    code = '''
import sys
class NoScientificImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy'}:
            raise AssertionError('scientific import attempted: ' + fullname)
sys.meta_path.insert(0, NoScientificImports())
from cipher_lab_plugins.frontier.simon_frontier import PublicFeaturePlan, encode_plan, decode_plan
p = PublicFeaturePlan(5, 0, 'simon-r5-factored-v1', ((1,),), ((), (0,)))
assert decode_plan(encode_plan(p)) == p
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            timeout=20, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_lazy_exports_keep_original_objects_and_static_types() -> None:
    import cipher_lab.representations as package
    from cipher_lab.representations import public_backstep, public_features, validate_public
    from cipher_lab.representations.input import public_backstep as original_backstep
    from cipher_lab.representations.tasks import public_features as original_features
    from cipher_lab.representations.tasks import validate_public as original_validate

    backstep: Callable[[Tensor, int], Tensor] = public_backstep
    features: Callable[[str], tuple[float, ...]] = public_features
    validate: Callable[[Tensor], None] = validate_public
    assert backstep is original_backstep
    assert features is original_features
    assert validate is original_validate
    assert package.__all__ == ['public_backstep', 'public_features', 'validate_public']
    with pytest.raises(AttributeError):
        getattr(package, 'not_an_export')


def toy_plan() -> PublicFeaturePlan:
    return PublicFeaturePlan(5, 0, 'simon-r5-factored-v1', ((1,), (2, 4)),
                             ((), (0,), (0, 1), (1,)))


def test_canonical_roundtrip_and_identity() -> None:
    plan = toy_plan()
    content = encode_plan(plan)
    assert decode_plan(content) == plan
    assert encode_plan(decode_plan(content)) == content
    identity = plan.identity
    assert identity.dimension == 4
    assert len(identity.order_sha256) == 64
    assert identity.to_dict()['plan_sha256'] == identity.plan_sha256
    body = json.loads(content)
    body.pop('identity')
    material = json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
    assert identity.plan_sha256 == hashlib.sha256(material).hexdigest()
    with pytest.raises(FrozenInstanceError):
        plan.rounds = 6  # type: ignore[misc]


@pytest.mark.parametrize(('field', 'value'), [
    ('rounds', True), ('rounds', 3), ('target_bit', 1), ('target_bit', False),
    ('representation', 'unknown'), ('leaves', [[1 << 128]]),
    ('leaves', [[-1]]), ('leaves', [[True]]), ('leaves', [[2, 1]]),
    ('leaves', [[1, 1]]), ('leaves', [[1], [1]]), ('leaves', [[0]]),
    ('leaves', [[]]), ('features', [[2]]), ('features', [[0, 0]]),
    ('features', [[1, 0]]), ('features', [[True]]), ('features', [[], []]),
    ('features', [[1], [0]]), ('features', []), ('features', [[0]] * 8193),
    ('features', [[0, 1, 2, 3]]), ('schema', 2), ('extra', 0),
])
def test_decoder_strict_schema_types_bounds_order(field: str, value: object) -> None:
    body = json.loads(encode_plan(toy_plan()))
    body[field] = value
    with pytest.raises(ValueError):
        decode_plan(json.dumps(body, sort_keys=True, separators=(',', ':')).encode())


def test_decode_rejects_noncanonical_and_hostile_json() -> None:
    content = encode_plan(toy_plan())
    for bad in (b' ' + content, content + b'\n', content.replace(b'"schema":1',
                b'"schema":1,"schema":1'), b'{"x":NaN}', b'{"x":Infinity}',
                b'[' * 2000, b'\xff', b'0', b'{}', b'x' * (MAX_PLAN_BYTES + 1)):
        with pytest.raises(ValueError):
            decode_plan(bad)
    body = json.loads(content)
    body['identity']['dimension'] = True
    with pytest.raises(ValueError):
        decode_plan(json.dumps(body, sort_keys=True, separators=(',', ':')).encode())
    body = json.loads(content)
    body['identity']['plan_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        decode_plan(json.dumps(body, sort_keys=True, separators=(',', ':')).encode())


def test_constructor_is_validated_before_adapter() -> None:
    with pytest.raises(ValueError):
        PublicFeaturePlan(5, 0, 'simon-r5-factored-v1', ((1,),), ((1,),))
    with pytest.raises(ValueError):
        PublicFeaturePlan(5, 0, 'simon-r5-factored-v1', ((1,), (2,)), ((0,),))
    # Every leaf is visited at most once per row; quota includes all leaf terms.
    many = tuple(range(1, 100002))
    with pytest.raises(ValueError, match='term'):
        PublicFeaturePlan(5, 0, 'simon-r5-factored-v1', (many,), ((0,),))


def test_factored_direct_spaces_boolean_normalization_not_residual13() -> None:
    # Constant=None; identical factors square away, duplicate candidates merge.
    groups = ((None, 0, 1), (None, 1, 2), (None, 3), (None, 4))
    result = factored_candidates(groups, Budget())
    assert result == ((), (0,), (0, 1), (0, 2), (1,), (1, 2), (2,), (3,), (4,))
    assert len(result) == 9


def test_six_round_shared_boolean_formula_has_exact_certificate() -> None:
    certificate = r6_formula_certificate(Budget())
    assert certificate['equivalent'] is True
    assert certificate['maximum_factors'] == 3
    assert certificate['shared_square_reduced'] is True
    assert certificate['shared_u3_u10_cancelled'] is True

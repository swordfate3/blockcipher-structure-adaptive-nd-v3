from __future__ import annotations

import json
from dataclasses import replace

import pytest
from speck_study_fakes import science_guard  # noqa: F401

from cipher_lab.speck_study.codec import (
    StudyPredictor,
    StudyTerm,
    decode_predictor,
    encode_predictor,
)
from cipher_lab.speck_study_config import SpeckStudyRecipe


def model(kind: str = 'B') -> StudyPredictor:
    r = SpeckStudyRecipe.synthetic_recipe(r_coef_size=40, b_coef_size=40)
    return StudyPredictor(kind=kind, task=('absolute-bit-prediction' if kind == 'B'
                          else 'teacher-reproduction'), source_sha256='a'*64, recipe=r,
                          terms=(StudyTerm(1 << 62, -20), StudyTerm(1 << 63, 20)),
                          count=40)


def test_strict_roundtrip_external_identity_and_recipe() -> None:
    m = model()
    data = encode_predictor(m)
    assert decode_predictor(data, expected_identity=m.identity, expected_recipe=m.recipe) == m
    assert m.predict_bit(0) == 0 and m.score(0) == 0
    assert m.predict_bit(1 << 63) == 1
    assert m.predict_bit(1 << 62) == 0
    with pytest.raises(ValueError):
        decode_predictor(data, expected_identity={**m.identity, 'source_sha256': 'b'*64},
                         expected_recipe=m.recipe)
    with pytest.raises(ValueError):
        decode_predictor(data, expected_identity=m.identity, expected_recipe=SpeckStudyRecipe())


@pytest.mark.parametrize('change', ['duplicate', 'nan', 'extra', 'bool', 'count', 'sum',
                                   'parity', 'zero', 'domain', 'duplicate-mask', 'oversize',
                                   'intercept', 'tie', 'threshold', 'over400', 'b-over12'])
def test_codec_rejects_malformed_and_scientific_contract(change: str) -> None:
    m = model('R' if change == 'over400' else 'B')
    raw = encode_predictor(m)
    obj = json.loads(raw)
    if change == 'duplicate':
        raw = b'{"count":40,'+raw[1:]
    elif change == 'nan':
        raw = raw.replace(b'"intercept":0', b'"intercept":NaN')
    elif change == 'oversize':
        raw = b' '*262145
    else:
        if change == 'extra':
            obj['weights'] = []
        elif change == 'bool':
            obj['count'] = True
        elif change == 'count':
            obj['count'] = 42
        elif change == 'sum':
            obj['terms'][0]['signed_sum'] = 42
        elif change == 'parity':
            obj['terms'][0]['signed_sum'] = 19
        elif change == 'zero':
            obj['terms'][0]['mask'] = 0
        elif change == 'domain':
            obj['terms'][0]['mask'] = 1
        elif change == 'duplicate-mask':
            obj['terms'][1]['mask'] = obj['terms'][0]['mask']
        elif change == 'intercept':
            obj['intercept'] = 1
        elif change == 'tie':
            obj['tie_bit'] = 1
        elif change == 'threshold':
            obj['terms'][0]['signed_sum'] = 2
        else:
            obj['terms'] = [{'mask': (i+1) << 32, 'signed_sum': 20}
                            for i in range(401 if change == 'over400' else 13)]
        raw = json.dumps(obj).encode()
    with pytest.raises(ValueError):
        decode_predictor(raw, expected_identity=m.identity, expected_recipe=m.recipe)


def test_invalid_model_cannot_be_encoded() -> None:
    m = model()
    with pytest.raises(ValueError):
        encode_predictor(replace(m, kind='R'))
    with pytest.raises(ValueError):
        encode_predictor(replace(m, terms=(StudyTerm(0, 20),)))
    with pytest.raises(ValueError):
        m.predict_bit(1)


def test_stdlib_closure_child_owns_guards() -> None:
    import os
    import subprocess
    import sys

    # No imported science in this fresh process, so do not import the torch guard.
    code = '''
import importlib.abc, os, random, secrets, sys
class Trap(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy', 'psutil'):
            raise AssertionError('scientific import')
        if fullname.startswith(('cipher_lab.ciphers', 'cipher_lab.training')):
            raise AssertionError('cipher/training import')
def blocked(*a, **k):
    raise AssertionError('private entropy')
for mod in (os, random, secrets):
    for name in ('urandom', '_urandom', 'SystemRandom', 'token_bytes', 'randbits'):
        if hasattr(mod, name):
            setattr(mod, name, blocked)
sys.meta_path.insert(0, Trap())
from cipher_lab.speck_study_config import SpeckStudyRecipe, estimate_study
from cipher_lab.speck_study.codec import (
    StudyPredictor, StudyTerm, encode_predictor, decode_predictor,
)
r = SpeckStudyRecipe()
m = StudyPredictor('B', 'absolute-bit-prediction', 'a'*64, r,
                   (StudyTerm(1<<63, 100000),), r.b_coef_size)
assert decode_predictor(encode_predictor(m), expected_identity=m.identity, expected_recipe=r) == m
assert estimate_study(r)['real_blocks'] == 541000512
print('stdlib-only guarded child OK')
'''
    reply = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True,
                           timeout=30, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert reply.returncode == 0, reply.stderr
    assert reply.stdout.strip() == 'stdlib-only guarded child OK'


def test_r_preserves_all_above_threshold_no_top12() -> None:
    r = SpeckStudyRecipe.synthetic_recipe(r_coef_size=40)
    terms = tuple(StudyTerm(i << 32, 20 if i % 2 else -20) for i in range(1, 401))
    m = StudyPredictor('R', 'teacher-reproduction', 'a'*64, r, terms, 40)
    assert decode_predictor(encode_predictor(m), expected_identity=m.identity,
                            expected_recipe=r).terms == terms


def test_nested_duplicate_and_identity_kind_bool_fail_closed() -> None:
    m = model()
    data = encode_predictor(m)
    for raw in (data.replace(b'"signed_sum":-20', b'"signed_sum":-20,"signed_sum":20'),
                data.replace(b'"mask":4611686018427387904', b'"mask":true'),
                data.replace(b'"kind":"B"', b'"kind":"R"'),
                b'\xff', b'['*10000):
        with pytest.raises(ValueError):
            decode_predictor(raw, expected_identity=m.identity, expected_recipe=m.recipe)

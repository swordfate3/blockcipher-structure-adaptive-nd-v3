"""Strict standard-library model format tests; intentionally no NumPy import."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from typing import cast

import pytest
from cipher_lab_plugins.fourier.codec import (
    FourierPredictor,
    FourierTerm,
    decode_predictor,
    encode_predictor,
)
from cipher_lab_plugins.fourier.config import FourierRecipe


def model() -> FourierPredictor:
    r = FourierRecipe(synthetic=True, rounds=0, pairs_per_layer=16, dev_size=16, test_size=16,
                      input_bits=2, coef_size=4)
    ident: dict[str, object] = {
        "source_sha256": "a" * 64, "backend": "demo", "synthetic": True,
        "task": {"cipher": "synthetic", "rounds": 0, "target_bit": 0, "input_bits": 2},
        "label_mode": "target_bit",
    }
    return FourierPredictor(recipe=r, identity=ident, count=4, intercept_sum=2,
                            intercept=.5, baseline=0,
                            terms=(FourierTerm(1 << 62, 2, .5), FourierTerm(1 << 63, -2, -.5)),
                            coefficient_rows=12)


def test_round_trip_canonical_and_detached_identity() -> None:
    m = model()
    content = encode_predictor(m)
    assert len(content) <= 65536
    recovered = decode_predictor(content, expected_identity=m.identity)
    assert recovered == m and encode_predictor(recovered) == content
    m.identity["backend"] = "simon"
    assert recovered.identity["backend"] == "demo"


@pytest.mark.parametrize("content", [b"{}", b"[]", b"null", b"not json", b"\xff",
                                       b" " * 65537, b'{"schema":1,"schema":2}',
                                       b'{"x": NaN}', b'{"x": Infinity}'])
def test_bad_bytes(content: bytes) -> None:
    with pytest.raises(ValueError):
        decode_predictor(content, expected_identity=model().identity)


@pytest.mark.parametrize("field,value", [
    ("schema", "other"), ("count", True), ("count", 0), ("count", 3),
    ("intercept", True), ("intercept", float("nan")), ("intercept", 1.1),
    ("intercept", .25), ("intercept_sum", 1), ("intercept_sum", 5),
    ("baseline", 1), ("coefficient_rows", 4), ("secret", "forbidden"),
])
def test_corrupt_top_level(field: str, value: object) -> None:
    m = model()
    obj = json.loads(encode_predictor(m))
    obj[field] = value
    with pytest.raises(ValueError):
        decode_predictor(json.dumps(obj).encode(), expected_identity=m.identity)


@pytest.mark.parametrize("mutation", ["zero", "negative", "overflow", "bool", "duplicate",
                                      "order", "coef", "sum", "extra", "many", "empty",
                                      "below", "domain", "countbool"])
def test_corrupt_terms(mutation: str) -> None:
    m = model()
    obj = json.loads(encode_predictor(m))
    terms = obj["terms"]
    if mutation in {"zero", "negative", "overflow", "bool", "domain"}:
        terms[0]["mask"] = {"zero": 0, "negative": -1, "overflow": 2**64,
                            "bool": True, "domain": 1}[mutation]
    if mutation == "duplicate":
        terms[1] = terms[0]
    if mutation == "order":
        terms.reverse()
    if mutation == "coef":
        terms[0]["coefficient"] = -.5
    if mutation == "sum":
        terms[0]["signed_sum"] = 1
    if mutation == "extra":
        terms[0]["key"] = "not allowed"
    if mutation == "many":
        obj["terms"] = terms * 17
    if mutation == "empty":
        obj["terms"] = []
    if mutation == "below":
        terms[0]["coefficient"] = 0
        terms[0]["signed_sum"] = 0
    if mutation == "countbool":
        terms[0]["signed_sum"] = True
    with pytest.raises(ValueError):
        decode_predictor(json.dumps(obj).encode(), expected_identity=m.identity)


@pytest.mark.parametrize("field,value", [("source_sha256", "b" * 64), ("backend", "simon"),
                                         ("label_mode", "pseudorandom_function"),
                                         ("synthetic", False), ("key", "forbidden"),
                                         ("task", {"cipher": "SIMON"})])
def test_identity_mismatch(field: str, value: object) -> None:
    m = model()
    with pytest.raises(ValueError):
        decode_predictor(encode_predictor(m), expected_identity=m.identity | {field: value})
    obj = json.loads(encode_predictor(m))
    obj["identity"][field] = value
    with pytest.raises(ValueError):
        decode_predictor(json.dumps(obj).encode(), expected_identity=m.identity)


def test_encode_also_validates_and_bool_identity_not_equal_integer() -> None:
    m = model()
    with pytest.raises(ValueError):
        encode_predictor(replace(m, count=True))
    ident = dict(m.identity)
    task = cast("dict[str, object]", ident["task"])
    ident["task"] = task | {"rounds": False}
    with pytest.raises(ValueError):
        decode_predictor(encode_predictor(m), expected_identity=ident)


def test_formal_identity_and_null_mode() -> None:
    r = FourierRecipe()
    ident: dict[str, object] = {
        "source_sha256": "b" * 64, "backend": "simon", "synthetic": False,
        "task": {"cipher": "SIMON", "rounds": 6, "input_bits": 64, "target_bit": 0},
        "label_mode": "pseudorandom_function",
    }
    m = FourierPredictor(r, ident, 65536, 0, 0.0, 0,
                         (FourierTerm(1, 65536, 1.0),), 131072)
    assert decode_predictor(encode_predictor(m), expected_identity=ident) == m
    with pytest.raises(ValueError):
        encode_predictor(replace(m, identity=ident | {"backend": "demo"}))


def test_codec_and_config_imports_stay_standard_library_only() -> None:
    script = '''
import sys
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch'} or fullname.endswith('.search'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Guard())
from cipher_lab_plugins.fourier.codec import encode_predictor, decode_predictor
from cipher_lab_plugins.fourier.config import FourierRecipe
print(FourierRecipe().method)
'''
    proc = subprocess.run([sys.executable, "-c", script], env=os.environ.copy(),
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "bit-fourier-search-v1"


def test_nested_duplicate_keys_rejected() -> None:
    m = model()
    content = encode_predictor(m)
    corrupted = content.replace(b'"rounds":0', b'"rounds":0,"rounds":0', 1)
    with pytest.raises(ValueError):
        decode_predictor(corrupted, expected_identity=m.identity)


def test_search_import_and_tiny_math_forbid_scientific_dependencies() -> None:
    script = '''
import sys
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'secrets', 'psutil'} or fullname.startswith((
            'cipher_lab.ciphers', 'cipher_lab.data', 'cipher_lab.models', 'cipher_lab.training'
        )):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Guard())
import numpy as np
from cipher_lab_plugins.fourier.search import parity_sign
assert parity_sign(np.array([0, 1], dtype=np.uint64), 1).tolist() == [1, -1]
print('public-tiny-only')
'''
    proc = subprocess.run([sys.executable, "-c", script], env=os.environ.copy(),
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "public-tiny-only"

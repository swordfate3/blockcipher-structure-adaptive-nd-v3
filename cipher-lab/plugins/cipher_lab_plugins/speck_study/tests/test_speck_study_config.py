from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from cipher_lab_plugins.speck_study.config import SpeckStudyRecipe, estimate_study
from cipher_lab_plugins.speck_study.tests.speck_study_fakes import science_guard  # noqa: F401


def test_formal_full_recipe_budget_and_pair_cost() -> None:
    r = SpeckStudyRecipe()
    assert r.method == 'speck-fourier-paper-prediction-v1'
    assert r.pairs_per_layer == 10_000_000
    e = estimate_study(r)
    assert e['r_search_teacher_rows'] == 1_140_000_000
    assert e['teacher_rows'] == 1_161_000_000
    assert e['r_classification_real_blocks'] == 20_000_000
    assert e['b_real_blocks'] == 521_000_000
    assert e['real_blocks'] == 541_000_512
    assert e['search_sample_bucket_products'] == 656_000_000_000
    assert r.policy_real_blocks == 542_000_000
    assert SpeckStudyRecipe.parse(json.dumps(r.to_dict())) == r
    with pytest.raises(FrozenInstanceError):
        r.rounds = 9  # type: ignore[misc]


@pytest.mark.parametrize('field,value', [
    ('pairs_per_layer', 1), ('max_prefixes', 401), ('tau', .2),
    ('rounds', 7), ('r_coef_size', True), ('r_teacher_accuracy_min', float('nan')),
    ('public_r_search_seed', 0), ('chunk_rows', 65536), ('intercept', 1),
])
def test_formal_refuses_scientific_or_invalid_changes(field: str, value: object) -> None:
    data = SpeckStudyRecipe().to_dict()
    data[field] = value
    with pytest.raises(ValueError):
        SpeckStudyRecipe.parse(data)


def test_recipe_strict_complete_unknown_duplicate_bool_nan() -> None:
    r = SpeckStudyRecipe()
    data = r.to_dict()
    for change in ('missing', 'unknown', 'bool'):
        bad = dict(data)
        if change == 'missing':
            del bad['b_test_size']
        elif change == 'unknown':
            bad['beam_width'] = 32
        else:
            bad['synthetic'] = 0
        with pytest.raises(ValueError):
            SpeckStudyRecipe.parse(bad)
    text = json.dumps(data)
    for bad_text in ('{"rounds":8,' + text[1:], text.replace('0.1', 'NaN', 1)):
        with pytest.raises(ValueError):
            SpeckStudyRecipe.parse(bad_text)


def test_synthetic_has_explicit_separate_identity() -> None:
    r = SpeckStudyRecipe.synthetic_recipe(pairs_per_layer=16, r_bits=8, b_bits=8,
                                          initial_depth=2, r_coef_size=16)
    assert r.synthetic and r.rounds == 0
    assert r.method != SpeckStudyRecipe().method
    assert SpeckStudyRecipe.parse(r.to_dict()) == r
    with pytest.raises(ValueError):
        replace(r, synthetic=False)
    with pytest.raises(ValueError):
        replace(r, r_bits=True)


def test_nested_batch_types_are_exact_not_numeric_equivalence() -> None:
    data = SpeckStudyRecipe().to_dict()
    data['batch_fallback_rows'] = [50000.0, 25000, 12500]
    with pytest.raises(ValueError):
        SpeckStudyRecipe.parse(data)


def test_all_recipe_fields_reject_bool_replacing_numeric_or_text() -> None:
    base = SpeckStudyRecipe().to_dict()
    for key in base:
        if key == 'synthetic':
            continue
        bad = {**base, key: True}
        with pytest.raises(ValueError, match='.'):
            SpeckStudyRecipe.parse(bad)

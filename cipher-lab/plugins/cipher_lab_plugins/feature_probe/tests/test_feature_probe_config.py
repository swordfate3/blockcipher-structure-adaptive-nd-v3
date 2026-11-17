from __future__ import annotations

import json
from pathlib import Path

import pytest

from cipher_lab_plugins.feature_probe.tests.feature_probe_fakes import poison, raw_recipe

__all__ = ['poison']


def test_strict_roundtrip(tmp_path: Path) -> None:
    from cipher_lab_plugins.feature_probe.cli import load_experiment
    from cipher_lab_plugins.feature_probe.config import parse_feature_probe
    raw = raw_recipe()
    recipe = parse_feature_probe(raw)
    assert recipe.to_dict() == raw
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(raw))
    assert load_experiment(path) == recipe


@pytest.mark.parametrize('field,value', [('key_seed', 4), ('arms', ['A']), ('rounds', 4),
    ('rounds', True), ('target_bit', 1), ('reference', 'bad'), ('representation', 'bad'),
    ('output_mode', 'residual'), ('schema', 2), ('label_mode', 'residual')])
def test_reject(field: str, value: object) -> None:
    from cipher_lab_plugins.feature_probe.config import parse_feature_probe
    raw = raw_recipe()
    raw[field] = value
    with pytest.raises(ValueError):
        parse_feature_probe(raw)

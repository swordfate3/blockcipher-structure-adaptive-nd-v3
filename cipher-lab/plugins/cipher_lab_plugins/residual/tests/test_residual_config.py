"""Residual tests: artificial inputs only; production science is poisoned first."""
from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch


@pytest.fixture(autouse=True)
def poison(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> Any:
        raise AssertionError('FORBIDDEN production science')
    # Torch is imported before patching secrets (Torch imports may use it).
    import secrets

    import cipher_lab.execution as execution
    monkeypatch.setattr(secrets, 'token_bytes', forbidden)
    monkeypatch.setattr(secrets, 'randbits', forbidden)
    monkeypatch.setattr(execution, 'production_rss_bytes', forbidden)
    monkeypatch.setattr(torch.optim.Adam, '__init__', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    # Manual output only: unit regressions must not consume real fit/forward budget.
    from cipher_lab.models import BitMLP
    def artificial_forward(model: BitMLP, inputs: torch.Tensor) -> torch.Tensor:
        bias = model.state_dict()['layers.4.bias'].detach()
        return cast(torch.Tensor, bias.reshape(1, 1).expand(len(inputs), 1).clone())
    monkeypatch.setattr(BitMLP, 'forward', artificial_forward)
    for family in ('simon', 'simeck', 'present', 'skinny'):
        module = importlib.import_module('cipher_lab.ciphers.' + family)
        for name in ('encrypt_many', 'encrypt_batch', 'encrypt'):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)


def raw_recipe() -> dict[str, Any]:
    return {'schema': 1, 'method': 'bit-residual-paired-v1',
            'reference': 'simon64-zero-master-v1', 'model': {'hidden_width': 2},
            'training': {'train': 2, 'dev': 2, 'test': 2, 'batch_size': 2, 'epochs': 2,
                         'learning_rate': .01, 'init_seed': 5, 'sample_seed': 7,
                         'shuffle_seed': 11}, 'rounds': 6, 'target_bit': 0,
            'label_mode': 'cipher'}


def residual_recipe() -> Any:
    from cipher_lab_plugins.residual.config import parse_residual_prediction
    return parse_residual_prediction(raw_recipe())


def test_strict_recipe_and_loader(tmp_path: Path) -> None:
    import json

    from cipher_lab.training_config import parse_prediction
    from cipher_lab_plugins.residual.cli import load_experiment
    from cipher_lab_plugins.residual.config import parse_residual_prediction
    raw = raw_recipe()
    recipe = parse_residual_prediction(raw)
    assert recipe.to_dict() == raw
    assert replace(recipe, rounds=44).rounds == 44
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(raw))
    assert load_experiment(path) == recipe
    with pytest.raises(ValueError):
        parse_prediction(raw)


@pytest.mark.parametrize('field,value', [('key_seed', 4), ('private_seed', 2),
    ('reference', 'zero-round-keys'), ('rounds', 45), ('rounds', True),
    ('target_bit', 64), ('label_mode', 'residual'), ('output_mode', 'residual')])
def test_recipe_rejects_unknown_and_invalid(field: str, value: object) -> None:
    from cipher_lab_plugins.residual.config import parse_residual_prediction
    raw = raw_recipe()
    raw[field] = value
    with pytest.raises(ValueError):
        parse_residual_prediction(raw)

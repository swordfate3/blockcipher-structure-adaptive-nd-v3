"""Artificial engineering only: poison science before fixtures create any owners."""
from __future__ import annotations

import importlib
from typing import Any, cast

import pytest
import torch
from torch import Tensor


@pytest.fixture(autouse=True)
def poison(monkeypatch: pytest.MonkeyPatch) -> None:
    import secrets

    import cipher_lab.execution as execution
    from cipher_lab.models import BitMLP, SharedM

    def forbidden(*args: object, **kwargs: object) -> Any:
        raise AssertionError('FORBIDDEN production science')

    def manual(model: Any, inputs: Tensor, *args: object, **kwargs: object) -> Tensor:
        state = model.state_dict()
        bias = state.get('layers.4.bias', torch.zeros(1))
        return cast(Tensor, bias.detach().reshape(1, 1).expand(len(inputs), 1).clone())

    for name in ('token_bytes', 'randbits', 'randbelow'):
        monkeypatch.setattr(secrets, name, forbidden)
    monkeypatch.setattr(execution, 'production_rss_bytes', forbidden)
    monkeypatch.setattr(torch.optim.Adam, '__init__', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)
    monkeypatch.setattr(torch.autograd, 'backward', forbidden)
    monkeypatch.setattr(torch.nn.Linear, 'forward', forbidden)
    monkeypatch.setattr(BitMLP, 'forward', manual)
    monkeypatch.setattr(SharedM, 'forward', manual)
    try:
        from cipher_lab_plugins.feature_probe.model import FeatureBitMLP
    except ModuleNotFoundError:
        pass  # Initial RED precedes the new module.
    else:
        monkeypatch.setattr(FeatureBitMLP, 'forward', manual)
    for family in ('simon', 'simeck', 'present', 'skinny'):
        module = importlib.import_module('cipher_lab.ciphers.' + family)
        for name in ('encrypt_many', 'encrypt_batch', 'encrypt'):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)


def raw_recipe() -> dict[str, Any]:
    return {'schema': 1, 'method': 'bit-public-feature-probe-v1',
            'reference': 'simon64-zero-master-v1', 'representation': 'simon-r3-residual13-v1',
            'model': {'hidden_width': 2},
            'training': {'train': 2, 'dev': 2, 'test': 2, 'batch_size': 2, 'epochs': 2,
                         'learning_rate': .01, 'init_seed': 31, 'sample_seed': 196905,
                         'shuffle_seed': 296905}, 'rounds': 3, 'target_bit': 0,
            'label_mode': 'cipher'}


def recipe() -> Any:
    from cipher_lab_plugins.feature_probe.config import parse_feature_probe
    return parse_feature_probe(raw_recipe())

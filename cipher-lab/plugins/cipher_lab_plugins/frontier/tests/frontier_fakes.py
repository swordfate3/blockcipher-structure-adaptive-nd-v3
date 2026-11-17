"""Artificial ledger only; no scientific forwards, updates or subprocesses."""
from __future__ import annotations

import hashlib
import importlib
import secrets
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'frontier'
# Audited public math bytes only; the loader verifies before any decode.
AUDITED_PLAN_SHA256 = {
    4: '44a837ee116facc009b942bbcdde27488fc7e52aa9f9d94be3ebc4a5130a207c',
    5: 'c0693d6dfa8d44a33e06df421eae993f690b040aece0aecb79e8477ce5493d81',
}


@pytest.fixture(autouse=True)
def poison(monkeypatch: pytest.MonkeyPatch) -> None:
    import cipher_lab.execution as execution
    from cipher_lab.models import BitMLP, SharedM
    from cipher_lab_plugins.feature_probe.model import FeatureBitMLP

    def forbidden(*args: object, **kwargs: object) -> Any:
        raise AssertionError('FORBIDDEN production science/subprocess')

    def manual(model: Any, inputs: Tensor, *args: object, **kwargs: object) -> Tensor:
        bias = model.state_dict().get('layers.4.bias', torch.zeros(1))
        return cast(Tensor, bias.detach().reshape(1, 1).expand(len(inputs), 1).clone())

    for name in ('token_bytes', 'randbits', 'randbelow'):
        monkeypatch.setattr(secrets, name, forbidden)
    monkeypatch.setattr(execution, 'production_rss_bytes', forbidden)
    for owner, name in ((torch.optim.Adam, '__init__'), (torch.optim.Adam, 'step'),
                        (torch.Tensor, 'backward'), (torch.autograd, 'backward'),
                        (torch.nn.Linear, 'forward'), (torch.nn.functional, 'linear'),
                        (subprocess, 'Popen')):
        monkeypatch.setattr(owner, name, forbidden)
    for cls in (BitMLP, FeatureBitMLP, SharedM):
        monkeypatch.setattr(cls, 'forward', manual)
    try:
        from cipher_lab_plugins.frontier.model import FrontierBitMLP
    except ModuleNotFoundError:
        pass
    else:
        monkeypatch.setattr(FrontierBitMLP, 'forward', manual)
    for family in ('simon', 'simeck', 'present', 'skinny'):
        module = importlib.import_module('cipher_lab.ciphers.' + family)
        for name in ('encrypt_many', 'encrypt_batch', 'encrypt'):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)


def plan_path(rounds: int = 4) -> Path:
    if rounds not in AUDITED_PLAN_SHA256:
        raise ValueError(f'no audited fixture plan for r{rounds}')
    path = FIXTURES / f'r{rounds}-public-plan.json'
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = AUDITED_PLAN_SHA256[rounds]
    if actual != expected:
        raise AssertionError(f'{path.name} sha256 {actual} != audited {expected}')
    return path


def plan(rounds: int = 4) -> Any:
    from cipher_lab_plugins.frontier.simon_frontier import decode_plan
    return decode_plan(plan_path(rounds).read_bytes())


def raw_recipe() -> dict[str, Any]:
    return {'schema': 1, 'method': 'bit-public-feature-frontier-v1',
            'reference': 'simon64-zero-master-v1', 'identity': plan().identity.to_dict(),
            'model': {'hidden_width': 2},
            'training': {'train': 2, 'dev': 2, 'test': 2, 'batch_size': 2, 'epochs': 2,
                         'learning_rate': .001, 'init_seed': 31, 'sample_seed': 196909,
                         'shuffle_seed': 296909}, 'rounds': 4, 'target_bit': 0,
            'label_mode': 'cipher'}


def recipe() -> Any:
    from cipher_lab_plugins.frontier.config import parse_frontier
    return parse_frontier(raw_recipe())

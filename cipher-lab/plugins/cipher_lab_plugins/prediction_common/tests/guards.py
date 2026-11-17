"""Test-only fail-closed poison, never imported by product execution."""
from __future__ import annotations

import importlib
from typing import Any

import pytest


@pytest.fixture
def science_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    import secrets

    import torch

    import cipher_lab.execution as execution
    from cipher_lab.models import BitMLP, FeatureBitMLP, FrontierBitMLP, SharedM
    from cipher_lab_plugins.feature_probe.model import FeatureBitMLP as FeatureWrapper
    from cipher_lab_plugins.frontier.model import FrontierBitMLP as FrontierWrapper

    def forbidden(*args: object, **kwargs: object) -> Any:
        raise AssertionError('UNAUTHORIZED_SCIENCE: test must bind artificial outputs')

    for name in ('token_bytes', 'randbits', 'randbelow'):
        monkeypatch.setattr(secrets, name, forbidden)
    monkeypatch.setattr(execution, 'production_rss_bytes', forbidden)
    for owner, name in ((torch.optim.Adam, '__init__'), (torch.optim.Adam, 'step'),
                        (torch.optim.Adam, 'zero_grad'), (torch.Tensor, 'backward'),
                        (torch.autograd, 'backward'), (torch.nn.Linear, 'forward'),
                        (torch.nn.functional, 'linear'), (torch.nn.Conv1d, 'forward'),
                        (torch.nn.Conv2d, 'forward'), (torch.nn.Conv3d, 'forward'),
                        (torch.nn.GroupNorm, 'forward'), (torch.nn.functional, 'conv1d'),
                        (torch.nn.functional, 'conv2d'), (torch.nn.functional, 'conv3d'),
                        (torch.nn.functional, 'group_norm')):
        monkeypatch.setattr(owner, name, forbidden)
    for cls in (BitMLP, SharedM, FeatureBitMLP, FrontierBitMLP, FeatureWrapper, FrontierWrapper):
        monkeypatch.setattr(cls, 'forward', forbidden)
    for family in ('simon', 'simeck', 'present', 'skinny'):
        module = importlib.import_module('cipher_lab.ciphers.' + family)
        for name in ('encrypt_many', 'encrypt_batch', 'encrypt'):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)

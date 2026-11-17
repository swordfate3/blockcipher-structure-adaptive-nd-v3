"""Pure cipher test poison only; independent of study/teacher/plugin helpers."""
from __future__ import annotations

import importlib
import os
import random
import secrets
from collections.abc import Iterator
from typing import NoReturn

import pytest

# Torch imports may initialize NumPy's lazy RNG: feed deterministic bytes only.
with pytest.MonkeyPatch.context() as bootstrap:
    bootstrap.setattr(os, 'urandom', lambda size: bytes(size))
    bootstrap.setattr(secrets, 'randbits', lambda bits: 0)
    importlib.import_module('numpy.random')
    import numpy as np
    import torch

__all__ = ['torch', 'science_guard', 'forbidden']


def forbidden(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError('pure cipher test forbids scientific execution')


@pytest.fixture(autouse=True)
def science_guard(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Permit public arithmetic/manual tensors, never real encryption or science."""
    for module in (os, random, secrets):
        for name in ('urandom', '_urandom', 'SystemRandom', 'token_bytes', 'token_hex',
                     'token_urlsafe', 'randbits', 'randbelow', 'choice'):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    for name in ('default_rng', 'seed', 'rand', 'randn', 'randint', 'random',
                 'random_sample', 'bytes', 'RandomState', 'Generator'):
        monkeypatch.setattr(np.random, name, forbidden)
    for name in ('rand', 'randn', 'randint', 'manual_seed', 'seed', 'initial_seed'):
        monkeypatch.setattr(torch, name, forbidden)
    monkeypatch.setattr(torch.nn.Linear, '__init__', forbidden)
    monkeypatch.setattr(torch.nn.Module, '__call__', forbidden)
    monkeypatch.setattr(torch.nn.Module, '_call_impl', forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)
    for cls in (torch.optim.Optimizer, torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        monkeypatch.setattr(cls, '__init__', forbidden)
        monkeypatch.setattr(cls, 'step', forbidden)
    for name in ('init', '_lazy_init', 'is_available', 'device_count', 'set_device',
                 'current_device', 'get_device_properties', 'memory_allocated'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    for module_name, entry in (('speck', 'encrypt_scalar'), ('speck_tensor', 'encrypt_tensor')):
        monkeypatch.setattr(importlib.import_module('cipher_lab.ciphers.' + module_name),
                            entry, forbidden)
    try:
        import resource
    except ImportError:
        pass
    else:
        monkeypatch.setattr(resource, 'getrusage', forbidden)
    yield

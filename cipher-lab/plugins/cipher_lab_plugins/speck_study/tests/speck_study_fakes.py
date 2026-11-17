"""Explicit Task1-only guards. Every child must call install_guards itself."""
from __future__ import annotations

import importlib
import os
import random
import secrets
import sys
from collections.abc import Iterator
from typing import NoReturn

import pytest

# Collection imports Torch only after deterministic NumPy lazy-RNG initialization.
# Restore these temporary import shims immediately; runtime fixture installs traps.
with pytest.MonkeyPatch.context() as _bootstrap:
    _bootstrap.setattr(os, 'urandom', lambda size: bytes(size))
    _bootstrap.setattr(secrets, 'randbits', lambda bits: 0)
    importlib.import_module('numpy.random')
    import numpy as np
    import torch

__all__ = ['np', 'torch', 'science_guard', 'install_guards', 'forbidden']


def forbidden(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError('Task1 forbids scientific execution')


def install_guards(patch: pytest.MonkeyPatch) -> None:
    # NumPy's lazy random initialization is fed deterministic fake entropy first.
    patch.setattr(os, 'urandom', lambda size: bytes(size))
    patch.setattr(secrets, 'randbits', lambda bits: 0)
    importlib.import_module('numpy.random')

    for module in (os, random, secrets):
        for name in ('urandom', '_urandom', 'SystemRandom', 'token_bytes', 'token_hex',
                     'token_urlsafe', 'randbits', 'randbelow', 'choice'):
            if hasattr(module, name):
                patch.setattr(module, name, forbidden)
    for name in ('default_rng', 'seed', 'rand', 'randn', 'randint', 'random',
                 'random_sample', 'bytes', 'RandomState', 'Generator'):
        patch.setattr(np.random, name, forbidden)
    for name in ('rand', 'randn', 'randint', 'manual_seed', 'seed', 'initial_seed'):
        patch.setattr(torch, name, forbidden)
    patch.setattr(torch.nn.Linear, '__init__', forbidden)
    patch.setattr(torch.nn.Module, '__call__', forbidden)
    patch.setattr(torch.nn.Module, '_call_impl', forbidden)
    patch.setattr(torch.Tensor, 'backward', forbidden)
    for cls in (torch.optim.Optimizer, torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        patch.setattr(cls, '__init__', forbidden)
        patch.setattr(cls, 'step', forbidden)
    for name in ('init', '_lazy_init', 'is_available', 'device_count', 'set_device',
                 'current_device', 'get_device_properties', 'memory_allocated'):
        patch.setattr(torch.cuda, name, forbidden)
    for name in ('encrypt_scalar', 'encrypt_tensor'):
        for module_name in ('cipher_lab.ciphers.speck', 'cipher_lab.ciphers.speck_tensor'):
            module = importlib.import_module(module_name)
            if hasattr(module, name):
                patch.setattr(module, name, forbidden)
    for name, module in tuple(sys.modules.items()):
        if name.startswith('cipher_lab.ciphers.'):
            for attr in ('encrypt', 'encrypt_many', 'encrypt_batch'):
                if hasattr(module, attr):
                    patch.setattr(module, attr, forbidden)
    try:
        import resource
    except ImportError:  # Windows synthetic import has no resource module.
        pass
    else:
        patch.setattr(resource, 'getrusage', forbidden)


@pytest.fixture(autouse=True)
def science_guard(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    install_guards(monkeypatch)
    yield

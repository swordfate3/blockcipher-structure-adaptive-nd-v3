"""Narrow guard for the in-memory checkpoint import-boundary child (tests only)."""
from __future__ import annotations

import importlib
import importlib.abc
import io
import os
import random
import secrets
import sys
from typing import Any, NoReturn

import pytest


def forbidden(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError('UNAUTHORIZED_SCIENCE: checkpoint child permits only artificial format IO')


class TeacherImportBlocker(importlib.abc.MetaPathFinder):
    """Reject actual imports without claiming the module was already imported.

    Explicit test fakes in sys.modules bypass find_spec normally. Do not insert
    None sentinels: production preimport membership checks must stay meaningful.
    """

    def find_spec(self, fullname: str, path: object = None,
                  target: object = None) -> None:
        if fullname.split('.')[0] in {'tensorflow', 'keras', 'h5py'}:
            forbidden()


class _NoCollectionOrMachine(TeacherImportBlocker):
    bootstrap = True

    def find_spec(self, fullname: str, path: object = None,
                  target: object = None) -> None:
        super().find_spec(fullname, path, target)
        if fullname.split('.')[0] in {'psutil', 'pynvml'}:
            if self.bootstrap:
                # Torch optionally imports NVML; keep it unavailable rather than
                # importing a machine API during the import-only bootstrap.
                raise ModuleNotFoundError('optional machine dependency blocked', name=fullname)
            forbidden()
        if fullname.startswith(('cipher_lab.ciphers', 'cipher_lab.training.',
                                'cipher_lab_plugins.speck_study.teacher')):
            forbidden()


def install_checkpoint_guard() -> None:
    """Import-only fake-entropy bootstrap, then poison, without cipher/training.

    This helper is invoked once in a disposable child; all patches last until
    that child exits. No product execution entry or scientific permission flag.
    """
    # Install collection/teacher rejection before even importing Torch. No
    # production package is imported merely to poison its cipher/teacher API.
    blocker = _NoCollectionOrMachine()
    sys.meta_path.insert(0, blocker)
    with pytest.MonkeyPatch.context() as bootstrap:
        bootstrap.setattr(os, 'urandom', lambda size: bytes(size))
        bootstrap.setattr(secrets, 'randbits', lambda bits: 0)
        torch = importlib.import_module('torch')
        np = importlib.import_module('numpy')
        importlib.import_module('numpy.random')
    blocker.bootstrap = False
    patch = pytest.MonkeyPatch()
    for module, names in (
        (os, ('urandom',)), (random, ('SystemRandom', '_urandom')),
        (secrets, ('token_bytes', 'token_hex', 'token_urlsafe', 'randbits', 'randbelow', 'choice')),
        (torch.nn.Module, ('__call__', '_call_impl')),
        (torch.Tensor, ('backward',)), (torch.autograd, ('backward', 'grad')),
        (torch.cuda, ('init', '_lazy_init', 'is_available', 'device_count', 'set_device',
                      'current_device', 'get_device_properties', 'memory_allocated')),
        (torch.nn.functional, ('linear', 'conv1d', 'conv2d', 'conv3d', 'group_norm')),
    ):
        for name in names:
            patch.setattr(module, name, forbidden)
    for optimizer in (torch.optim.Optimizer, torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        for name in ('__init__', 'step', 'zero_grad'):
            patch.setattr(optimizer, name, forbidden)
    import cipher_lab.execution as execution
    patch.setattr(execution, 'production_rss_bytes', forbidden)
    try:
        import resource
    except ImportError:
        pass
    else:
        patch.setattr(resource, 'getrusage', forbidden)
    # Pure CPU model construction/serialization is allowed; no actual saved
    # weights/NPZ, filename or file-backed stream may reach either decoder.
    for owner in (torch, np):
        load = owner.load

        def memory_load(source: object, *args: object,
                        _load: Any = load, **kwargs: object) -> Any:
            if not isinstance(source, io.BytesIO):
                forbidden()
            return _load(source, *args, **kwargs)

        patch.setattr(owner, 'load', memory_load)

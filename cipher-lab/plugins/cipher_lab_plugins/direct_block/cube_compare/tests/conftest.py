"""Task1 tests are independently fail-closed even without the core pytest plugin."""
from __future__ import annotations

import ctypes
import importlib
import itertools
import os
import random
import secrets
import socket
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, NoReturn
from uuid import UUID

import numpy as np
import pytest

# NumPy's unused legacy global RandomState can be initialized on first import.
# Do this with fake entropy even when the outer pytest safety plugin is disabled.
with pytest.MonkeyPatch.context() as bootstrap:
    bootstrap.setattr(secrets, 'randbits', lambda bits: 0)
    bootstrap.setattr(os, 'urandom', lambda size: bytes(size))
    bootstrap.setattr(random, '_urandom', lambda size: bytes(size))
    importlib.import_module('numpy.random')
    torch = importlib.import_module('torch')

from cipher_lab import artifacts, execution
from cipher_lab.ciphers import simon
from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter
from cipher_lab.representations.input import InputRepresentation


def forbidden(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError('Task1 scientific or external operation forbidden')


@pytest.fixture
def cube_entropy_probe(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Install a harmless entropy stand-in before testing that the guard replaces it."""
    calls: list[int] = []

    def record(size: int) -> bytes:
        calls.append(size)
        return bytes(size)

    monkeypatch.setattr(os, 'urandom', record)
    if hasattr(os, 'getrandom'):
        monkeypatch.setattr(os, 'getrandom', record)
    monkeypatch.setattr(random, '_urandom', record)
    return calls


@pytest.fixture(autouse=True)
def cube_manual_only(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest,
                     tmp_path_factory: pytest.TempPathFactory) -> None:
    if 'cube_entropy_probe' in request.fixturenames:
        request.getfixturevalue('cube_entropy_probe')
    counter = itertools.count(1)
    monkeypatch.setattr(artifacts, 'uuid4', lambda: UUID(int=next(counter)))
    for owner, names in (
        (os, ('urandom', 'getrandom', 'system', 'posix_spawn', 'posix_spawnp')),
        (secrets.SystemRandom, ('getrandbits', 'randbytes', 'random')),
        (random, ('_urandom', 'SystemRandom')),
        (secrets, ('SystemRandom', 'randbelow', 'randbits', 'choice',
                   'token_bytes', 'token_hex', 'token_urlsafe')),
        (socket, ('create_connection', 'socket', 'getaddrinfo')),
        (subprocess, ('Popen', 'run')),
        (ctypes, ('WinDLL',)),
        (simon, ('encrypt_many', 'encrypt_batch', '_round_words')),
        (execution, ('production_rss_bytes',)),
        (torch.cuda, ('is_available', 'device_count', 'get_device_properties', 'mem_get_info',
                      'init', '_lazy_init', 'synchronize', 'max_memory_allocated',
                      'max_memory_reserved')),
        (torch.Tensor, ('backward',)), (torch.autograd, ('grad', 'backward')),
        (torch.optim.Adam, ('__init__', 'step', 'zero_grad')),
        (InputRepresentation, ('forward',)),
        (torch.nn.functional, ('linear', 'conv1d', 'conv2d', 'conv3d', 'group_norm')),
    ):
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, forbidden)
    for layer in (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                  torch.nn.GroupNorm):
        monkeypatch.setattr(layer, 'forward', forbidden)
    # Prohibit opening any saved archives here, including outside the repo.
    for loader in (torch, np):
        original = loader.load

        def in_memory(source: object, *args: object,
                      _load: Callable[..., Any] = original, **kwargs: object) -> Any:
            if type(source) is not BytesIO:
                forbidden()
            return _load(source, *args, **kwargs)

        monkeypatch.setattr(loader, 'load', in_memory)

    opened = Path.open
    base = tmp_path_factory.getbasetemp().resolve()

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.as_posix().startswith('/proc/'):
            forbidden()
        if path.suffix in ('.npz', '.pt') or path.name in ('run.json', 'events.jsonl'):
            if not path.resolve().is_relative_to(base):
                forbidden()
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded_open)


@pytest.fixture
def cube_meter() -> Meter:
    return Meter(Policy(2, 120, 120, '2099-01-01T00:00:00Z', real_blocks=10000,
                        null_words=0, rss_stop_bytes=2**31, output_bytes=2**26,
                        optimizer_updates=10000, forward_rows=10000),
                 Clocks(lambda: 0., lambda: 0., lambda: datetime(2026, 1, 1, tzinfo=UTC)),
                 lambda: 1)

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
from types import SimpleNamespace
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


_RNN_NUMERICAL = ('lstm', 'gru', 'rnn_tanh', 'rnn_relu', 'lstm_cell', 'gru_cell',
                  'rnn_tanh_cell', 'rnn_relu_cell')
_RNN_DISPATCH = ('RNN_TANH', 'RNN_RELU')
_RNN_LAYERS = (torch.nn.LSTM, torch.nn.RNN, torch.nn.GRU,
               torch.nn.LSTMCell, torch.nn.RNNCell, torch.nn.GRUCell)


@pytest.fixture
def cube_rnn_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], dict[str, Callable[..., object]]]:
    """Install safe lower callables and capture the real handles before the guard."""
    calls: list[str] = []

    def record(*args: object, **kwargs: object) -> NoReturn:
        calls.append('reached harmless recurrent stub')
        raise AssertionError('recurrent guard missing')

    for layer in _RNN_LAYERS:
        monkeypatch.setattr(layer, 'forward', record)
    functions = SimpleNamespace(**dict.fromkeys(_RNN_NUMERICAL, record))
    monkeypatch.setattr(torch._C, '_VariableFunctions', functions)
    monkeypatch.setattr(torch._VF, 'vf', functions)
    for name in _RNN_NUMERICAL:
        monkeypatch.setattr(torch, name, record)
        monkeypatch.setattr(torch._VF, name, record)
    packet = torch.ops.aten.lstm
    handles = {'packet': packet, 'input': packet.input, 'data': packet.data,
               'bound-call': packet.input.__call__}
    monkeypatch.setattr(packet, '_op', record)
    for op in (packet.input, packet.data):
        monkeypatch.setattr(op, '_op', record)
        if hasattr(op, '_cpp_dispatch_handle'):
            monkeypatch.setattr(op, '_cpp_dispatch_handle', record)
    for name in _RNN_DISPATCH:
        monkeypatch.setitem(torch.nn.modules.rnn._rnn_impls, name, record)
    return calls, handles


def _block_recurrent_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    for layer in _RNN_LAYERS:
        monkeypatch.setattr(layer, 'forward', forbidden)
    functions = torch._C._VariableFunctions

    class GuardedVariableFunctions:
        # The native object's methods are read-only. Delegate non-recurrent
        # access unchanged; this proxy exists only inside the owner fixture.
        def __getattr__(self, name: str) -> Any:
            if name in _RNN_NUMERICAL:
                return forbidden
            return getattr(functions, name)

    guarded = GuardedVariableFunctions()
    monkeypatch.setattr(torch._C, '_VariableFunctions', guarded)
    monkeypatch.setattr(torch._VF, 'vf', guarded)
    for name in _RNN_NUMERICAL:
        for owner in (torch, torch._VF):
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, forbidden)
        packet = getattr(torch.ops.aten, name)
        # Protect the call targets on the actual objects before replacing the
        # namespace lookup, including packets/overloads cached before setup.
        overloads = [getattr(packet, key) for key in packet.overloads()]
        monkeypatch.setattr(packet, '_op', forbidden)
        for op in overloads:
            monkeypatch.setattr(op, '_op', forbidden)
            if hasattr(op, '_cpp_dispatch_handle'):
                monkeypatch.setattr(op, '_cpp_dispatch_handle', forbidden)
        monkeypatch.setattr(torch.ops.aten, name, forbidden)
    # Older Torch versions also cache VF functions in this dispatch table.
    for name in _RNN_DISPATCH:
        monkeypatch.setitem(torch.nn.modules.rnn._rnn_impls, name, forbidden)


@pytest.fixture(autouse=True)
def cube_manual_only(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest,
                     tmp_path_factory: pytest.TempPathFactory) -> None:
    if 'cube_entropy_probe' in request.fixturenames:
        request.getfixturevalue('cube_entropy_probe')
    if 'cube_rnn_probe' in request.fixturenames:
        request.getfixturevalue('cube_rnn_probe')
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
    _block_recurrent_calls(monkeypatch)
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

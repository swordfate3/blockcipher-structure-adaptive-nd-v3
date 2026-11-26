"""Owner-local artificial-only guard; independent of plugin_isolation."""
from __future__ import annotations

import ctypes
import importlib
import itertools
import os
import random
import secrets
import socket
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NoReturn

import pytest

# Torch can import NumPy's legacy global RNG. Bootstrap using fake bytes only.
with pytest.MonkeyPatch.context() as bootstrap:
    bootstrap.setattr(os, 'urandom', lambda size: bytes(size))
    bootstrap.setattr(random, '_urandom', lambda size: bytes(size))
    bootstrap.setattr(secrets, 'randbits', lambda bits: 0)
    if hasattr(os, 'getrandom'):
        bootstrap.setattr(os, 'getrandom', lambda size, *args: bytes(size))
    torch = importlib.import_module('torch')
    adam = importlib.import_module('torch.optim.adam')
    np = importlib.import_module('numpy')
    importlib.import_module('numpy.random')

from cipher_lab import artifacts, execution
from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter

_GUARD_ERROR = 'PREFIX_BRIDGE_MANUAL_ONLY'
_PREFIX = 'cipher_lab.ciphers.present_prefix'
_CIPHERS = {
    'present': ('encrypt_many',),
    'present_prefix': ('encrypt_prefixes', '_encrypt_into', '_round_keys', '_p_layer_table'),
    'present_s2': ('invert_second_output', '_invert_into'),
    'simon': ('encrypt_many', 'encrypt_batch', '_round_words'),
    'simon32': ('encrypt_scalar', 'encrypt_batch'),
    'simeck': ('encrypt_many',),
    'skinny': ('encrypt_many',),
    'speck': ('encrypt_scalar', 'enc_one_round', 'expand_key'),
    'speck_tensor': ('encrypt_tensor', 'enc_one_round', 'expand_key'),
}
_OPS = (
    'linear', 'addmm', 'matmul', 'mm', 'bmm',
    'conv1d', 'conv2d', 'conv3d', 'convolution', '_convolution',
    'group_norm', 'native_group_norm', 'layer_norm', 'native_layer_norm', 'batch_norm',
    'relu', 'relu_', 'gelu', 'silu', 'sigmoid', 'sigmoid_', 'tanh', 'leaky_relu',
    'softmax', '_softmax', 'log_softmax', '_log_softmax', 'topk',
    'binary_cross_entropy_with_logits', '_fused_adam_',
)


def forbidden(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError(_GUARD_ERROR)


def _poison_aliases(mp: pytest.MonkeyPatch, originals: dict[int, object]) -> None:
    # Patch `from ... import ...` aliases bound in loaded project modules.
    for name, module in tuple(sys.modules.items()):
        if isinstance(module, ModuleType) and name.startswith(
            ('cipher_lab.', 'cipher_lab_plugins.')
        ):
            for attr, value in tuple(vars(module).items()):
                if id(value) in originals:
                    mp.setattr(module, attr, forbidden)


def _numerical_guard(mp: pytest.MonkeyPatch, originals: dict[int, object]) -> None:
    for layer in (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                  torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.BatchNorm1d,
                  torch.nn.ReLU, torch.nn.GELU, torch.nn.Sigmoid, torch.nn.SiLU,
                  torch.nn.Softmax, torch.nn.Tanh, torch.nn.LeakyReLU):
        old = layer.forward
        originals[id(old)] = old
        mp.setattr(layer, 'forward', forbidden)
    functions = torch._C._VariableFunctions

    class GuardedVariableFunctions:
        def __getattr__(self, name: str) -> Any:
            return forbidden if name in _OPS else getattr(functions, name)

    guarded = GuardedVariableFunctions()
    mp.setattr(torch._C, '_VariableFunctions', guarded)
    vf = torch._VF
    mp.setattr(vf, 'vf', guarded)
    vf_attrs = vars(vf)
    for name in _OPS:
        if hasattr(functions, name):
            native = getattr(functions, name)
            originals[id(native)] = native
        # Dynamic _VF names are already poisoned by the proxy. Only replace
        # explicit module attrs, preserving exactly what was defined on undo.
        if name in vf_attrs:
            original = vf_attrs[name]
            originals[id(original)] = original
            mp.setitem(vf_attrs, name, forbidden)
        for namespace in (torch, torch.nn.functional, torch._C._nn, torch.Tensor):
            if hasattr(namespace, name):
                original = getattr(namespace, name)
                originals[id(original)] = original
                mp.setattr(namespace, name, forbidden)
        if not hasattr(torch.ops.aten, name):
            continue
        packet = getattr(torch.ops.aten, name)
        overloads = [getattr(packet, key) for key in packet.overloads()]
        # A cached packet/overload/bound __call__ still reads these targets.
        for operator in (packet, *overloads):
            for attr in ('_op', '_op_dk', '_cpp_dispatch_handle'):
                if hasattr(operator, attr):
                    mp.setattr(operator, attr, forbidden)
        for overload in overloads:
            # A pre-bound redispatch reads these fields, not the call targets above.
            # Use the dict so we don't materialize the native cached _handle.
            attrs = vars(overload)
            mp.setitem(attrs, '_handle', SimpleNamespace(redispatch_boxed=forbidden))
            if '_pyobj_dispatcher' in attrs:
                mp.setitem(attrs, '_pyobj_dispatcher', None)
        mp.setattr(torch.ops.aten, name, forbidden)


@pytest.fixture
def bridge_prefix_with_stub(
    bridge_manual_only: tuple[pytest.MonkeyPatch, Callable[..., Any]],
) -> Callable[[Callable[..., None]], Callable[..., Any]]:
    """Expose the public validator only after substituting its private kernel."""
    guard, public = bridge_manual_only
    prefix = importlib.import_module(_PREFIX)

    def bind(stub: Callable[..., None]) -> Callable[..., Any]:
        # The guard already patched this name; undo both writes on the same stack.
        guard.setattr(prefix, '_encrypt_into', stub)
        return public

    return bind


@pytest.fixture
def bridge_guard_probe(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], tuple[object, ...]]:
    """Cache real *dispatch wrappers* with harmless call targets before the guard."""
    from cipher_lab.ciphers import present, present_s2

    calls: list[str] = []

    def harmless(*args: object, **kwargs: object) -> NoReturn:
        calls.append('unblocked stub')
        raise AssertionError('guard did not run')

    packet = torch.ops.aten.linear
    overload = packet.default
    for dispatcher in (packet, overload):
        for name in ('_op', '_op_dk', '_cpp_dispatch_handle'):
            if hasattr(dispatcher, name):
                monkeypatch.setattr(dispatcher, name, harmless)
    def harmless_redispatch(*args: object, **kwargs: object) -> NoReturn:
        calls.append('unblocked boxed redispatch stub')
        raise AssertionError('redispatch guard did not run')

    # Do not materialize the native cached_property, even when setting the stub.
    monkeypatch.setitem(vars(overload), '_handle',
                        SimpleNamespace(redispatch_boxed=harmless_redispatch))
    monkeypatch.setattr(overload, '_pyobj_dispatcher', None, raising=False)
    redispatch = overload.redispatch

    prefix = importlib.import_module(_PREFIX)
    aliases = ModuleType('cipher_lab_plugins._bridge_test_aliases')

    def stub_for(label: str) -> Callable[..., NoReturn]:
        def record(*args: object, **kwargs: object) -> NoReturn:
            calls.append(label)
            raise AssertionError('guard did not run')
        return record

    for owner, name, alias in (
        (torch.nn.functional, 'linear', 'functional'),
        (torch.nn.Conv1d, 'forward', 'conv'),
        (torch.nn.GroupNorm, 'forward', 'norm'),
        (torch.nn.ReLU, 'forward', 'relu'),
        (present, 'encrypt_many', 'public_cipher'),
        (prefix, 'encrypt_prefixes', 'public_prefix'),
        (prefix, '_encrypt_into', 'private_loop'),
        (present_s2, 'invert_second_output', 'public_inverse'),
        (present_s2, '_invert_into', 'private_inverse'),
    ):
        stub = stub_for(alias)
        monkeypatch.setattr(owner, name, stub)
        setattr(aliases, alias, stub)
    native_stub = stub_for('native_cached')
    fake_vf = SimpleNamespace(linear=native_stub)
    monkeypatch.setattr(torch._C, '_VariableFunctions', fake_vf)
    monkeypatch.setattr(torch._VF, 'vf', fake_vf)
    setattr(aliases, 'native_cached', native_stub)
    monkeypatch.setitem(sys.modules, aliases.__name__, aliases)
    return calls, (packet, overload, overload.__call__, lambda: redispatch(None),
                   lambda: overload.redispatch(None), aliases)


@pytest.fixture
def bridge_vf_lifecycle_probe() -> Iterator[None]:
    """Check the real module's attrs after the guard's own context has exited."""
    module = torch._VF
    assert 'linear' not in vars(module)  # the probe exercises a dynamic name
    assert 'relu' not in vars(module)
    dynamic = object()
    explicit = object()
    fake_vf = SimpleNamespace(linear=dynamic, relu=object())
    with pytest.MonkeyPatch.context() as probe:
        probe.setattr(module, 'vf', fake_vf)
        probe.setitem(vars(module), 'relu', explicit)
        yield
        # Probe attributes/identity only: no actual operator is called.
        try:
            assert module.vf is fake_vf
            assert 'linear' not in vars(module), 'guard left a shadowing explicit _VF.linear'
            assert module.linear is dynamic
            assert vars(module)['relu'] is explicit
        finally:
            # Keep a failing regression from contaminating later tests.
            vars(module).pop('linear', None)


@pytest.fixture(scope='module')
def bridge_prefix_lifecycle_probe() -> Iterator[None]:
    """Compare identity after even the outer function-scoped patch stack unwinds."""
    prefix = importlib.import_module(_PREFIX)
    original = prefix._encrypt_into
    assert original is not forbidden
    yield
    try:
        assert prefix._encrypt_into is original, 'guard left _encrypt_into rebound'
        assert prefix._encrypt_into is not forbidden
    finally:
        # Isolate later tests even on a failing regression.
        vars(prefix)['_encrypt_into'] = original


@pytest.fixture(autouse=True)
def bridge_manual_only(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[pytest.MonkeyPatch, Callable[..., Any]]]:
    for fixture in ('bridge_guard_probe', 'bridge_vf_lifecycle_probe'):
        if fixture in request.fixturenames:
            request.getfixturevalue(fixture)
    public = importlib.import_module(_PREFIX).encrypt_prefixes
    # End the guard before undoing the probes' harmless originals.
    with pytest.MonkeyPatch.context() as guard:
        _install_guard(guard, tmp_path_factory)
        yield guard, public


def _install_guard(monkeypatch: pytest.MonkeyPatch,
                   tmp_path_factory: pytest.TempPathFactory) -> None:
    counter = itertools.count(1)

    def fake_uuid() -> uuid.UUID:
        return uuid.UUID(int=next(counter))

    monkeypatch.setattr(uuid, 'uuid4', fake_uuid)
    monkeypatch.setattr(artifacts, 'uuid4', fake_uuid)
    originals: dict[int, object] = {}
    cube_resources = importlib.import_module(
        'cipher_lab_plugins.direct_block.cube_compare.resources')
    owners = [
        (os, ('urandom', 'getrandom', 'system', 'posix_spawn', 'posix_spawnp')),
        (random, ('_urandom', 'SystemRandom')),
        (secrets.SystemRandom, ('getrandbits', 'randbytes', 'random')),
        (secrets, ('SystemRandom', 'randbelow', 'randbits', 'choice',
                   'token_bytes', 'token_hex', 'token_urlsafe')),
        (socket, ('socket', 'create_connection', 'getaddrinfo')),
        (subprocess, ('Popen', 'run')),
        (ctypes, ('WinDLL', 'CDLL')),
        (execution, ('production_rss_bytes',)),
        (cube_resources, ('working_set_bytes', '_read_windows_working_set', 'device_identity')),
        (torch.Tensor, ('backward', 'cuda', 'seed')),
        (torch.autograd, ('backward', 'grad')),
        (torch.nn.utils, ('clip_grad_norm_',)),
        (torch.optim.Adam, ('__init__', 'step', 'zero_grad')),
        (adam, ('adam', '_single_tensor_adam', '_multi_tensor_adam', '_fused_adam')),
        (torch.optim._functional, ('adam',)),
        (torch.cuda, ('init', '_lazy_init', 'is_available', 'device_count',
                      'get_device_properties', 'mem_get_info', 'synchronize', 'empty_cache',
                      'max_memory_allocated', 'max_memory_reserved')),
        (torch, ('load', 'seed')),
        (np, ('load', 'savez', 'savez_compressed')),
    ]
    for family, attrs in _CIPHERS.items():
        name = f'cipher_lab.ciphers.{family}'
        module = importlib.import_module(name)
        # Fail closed if any required cipher boundary changes.
        for attr in attrs:
            original = getattr(module, attr)
            originals[id(original)] = original
            monkeypatch.setattr(module, attr, forbidden)
    for owner, names in owners:
        for name in names:
            if hasattr(owner, name):
                original = getattr(owner, name)
                originals[id(original)] = original
                monkeypatch.setattr(owner, name, forbidden)
    _numerical_guard(monkeypatch, originals)
    _poison_aliases(monkeypatch, originals)
    original_open = Path.open
    base = tmp_path_factory.getbasetemp().resolve()

    def open_guard(path: Path, *args: Any, **kwargs: Any) -> Any:
        # Task1 fixtures use opaque artificial bytes and run.json, no NPZ/weights load.
        resolved = path.resolve()
        if resolved.is_relative_to('/proc') or path.suffix in ('.npz', '.pth', '.h5'):
            forbidden()
        if path.suffix == '.pt' or path.name in ('run.json', 'events.jsonl'):
            if not resolved.is_relative_to(base):
                forbidden()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_guard)


@pytest.fixture
def bridge_meter() -> Meter:
    return Meter(
        Policy(2, 120, 120, '2026-01-01T01:00:00Z', real_blocks=14, null_words=0,
               optimizer_updates=1_000, forward_rows=10_000, rss_stop_bytes=2**31,
               output_bytes=2**24),
        Clocks(lambda: 0., lambda: 0., lambda: datetime(2026, 1, 1, tzinfo=UTC)),
        lambda: 1,
    )

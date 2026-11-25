"""Independently guarded artificial-only split8 tests, also with -p no:plugin_isolation."""
from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import import_module
from types import ModuleType
from typing import Any

import pytest
from cipher_lab_plugins.direct_block.cube_compare.tests.conftest import (
    cube_manual_only as cube_manual_only,
)
from cipher_lab_plugins.direct_block.cube_compare.tests.conftest import forbidden

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter

NOW = datetime(2026, 1, 1, tzinfo=UTC)
_MLP_OPS = ('linear', 'relu', 'relu_')


def _block_mlp_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    for layer in (torch.nn.Linear, torch.nn.ReLU):
        monkeypatch.setattr(layer, 'forward', forbidden)

    # Chain over the cube guard's read-only native-function proxy so the
    # recurrent poison is preserved. Patch the direct public/native aliases too.
    functions = torch._C._VariableFunctions

    class GuardedVariableFunctions:
        def __getattr__(self, name: str) -> Any:
            if name in _MLP_OPS:
                return forbidden
            return getattr(functions, name)

    guarded = GuardedVariableFunctions()
    monkeypatch.setattr(torch._C, '_VariableFunctions', guarded)
    monkeypatch.setattr(torch._VF, 'vf', guarded)
    for name in _MLP_OPS:
        for owner in (torch, torch.nn.functional, torch._C._nn, torch._VF, torch.Tensor):
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, forbidden)
        packet = getattr(torch.ops.aten, name)
        overloads = [getattr(packet, key) for key in packet.overloads()]
        # Patching the call targets on the objects also covers already-cached
        # packets, overloads and their bound __call__, not just fresh lookup.
        monkeypatch.setattr(packet, '_op', forbidden)
        for op in overloads:
            for target in ('_op', '_cpp_dispatch_handle', '_op_dk'):
                if hasattr(op, target):
                    monkeypatch.setattr(op, target, forbidden)
        monkeypatch.setattr(torch.ops.aten, name, forbidden)


@pytest.fixture(autouse=True)
def split8_manual_only(monkeypatch: pytest.MonkeyPatch, cube_manual_only: None,
                       request: pytest.FixtureRequest) -> None:
    if 'split8_mlp_probe' in request.fixturenames:
        request.getfixturevalue('split8_mlp_probe')
    _block_mlp_calls(monkeypatch)

    # Guard the defining modules and aliases already bound at collection.
    from cipher_lab_plugins.direct_block.cube_compare import preflight, resources

    for name, attrs in (
        ('cipher_lab.ciphers.simon32', ('encrypt_scalar', 'encrypt_batch')),
        ('cipher_lab_plugins.direct_block.split8.kat', ('production',)),
    ):
        module = import_module(name)
        for attr in attrs:
            original = getattr(module, attr)
            for key, loaded in tuple(sys.modules.items()):
                if isinstance(loaded, ModuleType) and key.startswith(
                    ('cipher_lab.', 'cipher_lab_plugins.')
                ):
                    for alias, value in tuple(vars(loaded).items()):
                        if value is original:
                            monkeypatch.setattr(loaded, alias, forbidden)
    for module, defined in (
        (preflight, ('gpu_gate', 'gpu_peaks', 'public_cipher_checks')),
        (resources, ('working_set_bytes', '_read_windows_working_set', 'device_identity')),
    ):
        for attr in defined:
            monkeypatch.setattr(module, attr, forbidden)


@pytest.fixture
def split8_meter() -> Meter:
    return Meter(Policy(2, 120, 120, '2026-01-01T01:00:00Z', real_blocks=2_000_000,
                        null_words=0, optimizer_updates=1_000_000, forward_rows=200_000_000,
                        rss_stop_bytes=2**31, output_bytes=2**26), split8_clocks(), lambda: 1)


def split8_clocks(wall: Callable[[], float] = lambda: 0.) -> Clocks:
    return Clocks(wall, lambda: 0., lambda: NOW)

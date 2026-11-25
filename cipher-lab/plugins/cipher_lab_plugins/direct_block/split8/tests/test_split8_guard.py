"""Exercise only harmless lower dispatch targets installed before the split8 guard."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch


@pytest.fixture
def split8_mlp_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], dict[str, Callable[..., object]]]:
    calls: list[str] = []
    handles: dict[str, Callable[..., object]] = {}

    def stub(*args: object, **kwargs: object) -> None:
        calls.append('lower stub reached')

    # The native _VariableFunctions attributes are read-only. Use a proxy with
    # harmless MLP bindings so even a missing guard cannot enter a real kernel.
    original = torch._C._VariableFunctions

    class FakeFunctions:
        linear = relu = relu_ = staticmethod(stub)

        def __getattr__(self, name: str) -> Any:
            return getattr(original, name)

    functions = FakeFunctions()
    monkeypatch.setattr(torch._C, '_VariableFunctions', functions)
    monkeypatch.setattr(torch._VF, 'vf', functions)
    monkeypatch.setattr(torch._C._nn, 'linear', stub)
    for name in ('relu', 'relu_'):
        for owner in (torch, torch._VF, torch.Tensor):
            monkeypatch.setattr(owner, name, stub)
    for name in ('linear', 'relu', 'relu_'):
        packet = getattr(torch.ops.aten, name)
        monkeypatch.setattr(packet, '_op', stub)
        handles[f'{name}.packet'] = packet
        handles[f'{name}.packet.bound'] = packet.__call__
        for variant in packet.overloads():
            op = getattr(packet, variant)
            handles[f'{name}.{variant}'] = op
            handles[f'{name}.{variant}.bound'] = op.__call__
            monkeypatch.setattr(op, '_op', stub)
            for low in ('_cpp_dispatch_handle', '_op_dk'):
                if hasattr(op, low):
                    monkeypatch.setattr(op, low, stub)
    return calls, handles


def test_split8_guard_rejects_lower_mlp_and_cached_handles(
    split8_mlp_probe: tuple[list[str], dict[str, Callable[..., object]]],
) -> None:
    calls, cached = split8_mlp_probe
    # All targets were replaced by harmless stubs before the guard, or reach
    # such a stub through the cached object. Exercise each, even under RED.
    entrypoints = {'C._nn.linear': torch._C._nn.linear, **cached}
    for name in ('relu', 'relu_'):
        for label, module in (('torch', torch), ('VF', torch._VF), ('Tensor', torch.Tensor),
                              ('C._VariableFunctions', torch._C._VariableFunctions),
                              ('VF.vf', getattr(torch._VF, 'vf'))):
            entrypoints[f'{label}.{name}'] = getattr(module, name)
    for name in ('linear', 'relu', 'relu_'):
        entrypoints[f'aten.{name}'] = getattr(torch.ops.aten, name)
    for label, op in cached.items():
        if label.endswith(('.bound', '.packet')):
            continue
        for low in ('_cpp_dispatch_handle', '_op_dk'):
            if hasattr(op, low):
                entrypoints[f'{label}.{low}'] = getattr(op, low)

    escaped: list[str] = []
    for name, call in entrypoints.items():
        try:
            call()
        except AssertionError as error:
            assert str(error) == 'Task1 scientific or external operation forbidden'
        else:
            escaped.append(name)
    assert not escaped and not calls, (escaped, len(calls))

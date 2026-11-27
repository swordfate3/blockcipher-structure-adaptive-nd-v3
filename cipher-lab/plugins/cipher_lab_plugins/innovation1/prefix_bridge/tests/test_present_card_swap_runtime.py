from __future__ import annotations

import sys
from dataclasses import replace
from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from cipher_lab.execution import Meter


@pytest.fixture
def swap_config(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # Stand-in for the separately owned assembly contract when it is absent.
    from cipher_lab_plugins.innovation1 import present_card_swap as package

    name = f'{package.__name__}.config'
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
    config = ModuleType(name)
    monkeypatch.setattr(config, 'SLOT', 'original', raising=False)
    monkeypatch.setattr(config, 'SLOTS', ('original', 'swap-50'), raising=False)
    monkeypatch.setattr(config, 'SWAP_SEED', 20261128, raising=False)
    monkeypatch.setattr(config, 'METHOD', 'innovation1-present-card-swap-v1', raising=False)
    monkeypatch.setattr(config, 'recipe', lambda: {'method': config.METHOD}, raising=False)
    monkeypatch.setitem(sys.modules, name, config)
    monkeypatch.setattr(package, 'config', config, raising=False)
    return config


def test_swap_all_makes_owned_contiguous_copy() -> None:
    from cipher_lab_plugins.innovation1.present_card_swap.views import swap_all

    original = np.arange(4 * 128, dtype=np.float32).reshape(4, 1, 2, 64)[::2]
    original.flags.writeable = False
    actual = swap_all(original)
    assert actual.flags.owndata and actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, original[:, :, ::-1])
    actual[:] = -1
    assert np.all(original >= 0)


@pytest.mark.parametrize('slot,prior_steps', [('original', 0), ('swap-50', 3000)])
def test_local_step_drives_one_pure_update(monkeypatch: pytest.MonkeyPatch,
                                          bridge_meter: Meter,
                                          slot: str, prior_steps: int) -> None:
    from cipher_lab_plugins.innovation1.models import TorchRuntime
    from cipher_lab_plugins.innovation1.present_card_fresh import config as pure
    from cipher_lab_plugins.innovation1.present_card_swap.runtime import FitRuntime, Runtime

    monkeypatch.setattr(bridge_meter, '_policy', replace(bridge_meter._policy,
                                                       optimizer_updates=6001))
    bridge_meter._attempted['optimizer_updates'] = prior_steps
    bridge_meter._completed['optimizer_updates'] = prior_steps

    def start(self: Runtime) -> None:
        self._state = 'TRAINING'
        monkeypatch.setitem(self.models, slot, None)
        monkeypatch.setitem(self.optimizers, slot, None)

    inputs = np.arange(5000 * 128, dtype=np.float32).reshape(5000, 1, 2, 64)
    inputs.flags.writeable = False
    labels = (np.arange(5000, dtype=np.uint16) % 2).astype(np.uint8).reshape(5000, 1)
    transfers: list[object] = []
    updates: list[tuple[object, float]] = []

    def device(self: Runtime, value: Any) -> Any:
        transfers.append(value)
        return value

    def update(self: object, actual_slot: str, task: Any, x: Any, y: Any,
               lr: float, meter: Meter) -> float:
        assert actual_slot == slot and task.rounds == 7 and meter is bridge_meter
        assert y is labels
        updates.append((x, lr))
        return .5

    monkeypatch.setattr(Runtime, 'start', start)
    monkeypatch.setattr(Runtime, '_on_device', device)
    monkeypatch.setattr(TorchRuntime, 'update', update)
    fit = FitRuntime(slot, {}, True, source={'toy/a.py': 'f'*64}, run_id='toy',
                     origin=prior_steps)
    with pytest.raises(ValueError, match='UPDATE_SEQUENCE'):
        fit.batch_update(inputs, labels, bridge_meter)
    bridge_meter.reserve('optimizer_updates', 2)
    with pytest.raises(ValueError, match='UPDATE_SEQUENCE'):
        fit.batch_update(inputs, labels, bridge_meter)
    assert transfers == [] and updates == []
    bridge_meter._attempted['optimizer_updates'] = prior_steps

    for step in range(2):
        bridge_meter.reserve('optimizer_updates', 1)
        assert fit.batch_update(inputs, labels, bridge_meter) == .5
        assert len(updates) == step + 1 and len(transfers) == 2 * (step + 1)
        actual, lr = updates[-1]
        assert lr == pure.learning_rate(step)
        if slot == 'original':
            assert actual is inputs
        else:
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261128, step])))
            mask = rng.random(len(inputs)) < .5
            assert 0 < mask.sum() < len(inputs)
            assert isinstance(actual, np.ndarray) and actual.flags.c_contiguous
            assert not np.shares_memory(actual, inputs)
            expected = inputs.copy()
            expected[mask] = inputs[mask, :, ::-1]
            np.testing.assert_array_equal(actual, expected)
        bridge_meter.complete('optimizer_updates', 1)

    np.testing.assert_array_equal(inputs.ravel(), np.arange(inputs.size, dtype=np.float32))
    monkeypatch.setattr(Runtime, 'predict_inputs', lambda self, x: x)
    assert fit.predict(inputs) is inputs


@pytest.mark.parametrize('fault', ['', 'method', 'slot', 'run_id', 'recipe', 'source', 'step',
                                  'epoch', 'geometry', 'initial_sha256', 'frozen', 'buffer'])
def test_checkpoint_binds_swap_card_and_restores_on_cpu(
    monkeypatch: pytest.MonkeyPatch, swap_config: ModuleType, fault: str,
) -> None:
    import io

    import torch
    from cipher_lab_plugins.innovation1.checkpoint import encode
    from cipher_lab_plugins.innovation1.present_card_fresh import config as pure
    from cipher_lab_plugins.innovation1.present_card_fresh.layout import STRUCTURE_SHA256
    from cipher_lab_plugins.innovation1.present_card_swap.runtime import FitRuntime, Runtime
    from torch import nn

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.]))
            self.frozen = nn.Parameter(torch.tensor([1.]), requires_grad=False)
            self.register_buffer('indices', torch.tensor([2]))

    toy = Toy()
    initial = 'c'*64
    geometry = {'public': 'a'*64, 'skinny_column': 'b'*64,
                'shared.stems.local.indices': 'd'*64, 'shared.stems.word.indices': 'e'*64,
                'shared.stems.nibble.indices': STRUCTURE_SHA256}
    point = dict(epoch=2, step=200, bce=.5)
    arm: dict[str, object] = dict(selected=point, dev=[point], initial_sha256=initial,
                                  geometry=geometry, parameters=3)
    source = {'toy/a.py': 'f'*64}
    origin = 3000

    def fresh(self: Runtime) -> Any:
        self._initial, self._parameters, self.geometry = initial, 3, geometry
        return toy

    def start(self: Runtime) -> None:
        self.models[self.slot] = self._fresh_cpu()
        self._state = 'TRAINING'

    monkeypatch.setattr(Runtime, '_fresh_cpu', fresh)
    monkeypatch.setattr(Runtime, 'start', start)
    writer = FitRuntime('swap-50', arm, True, source=source, run_id='toy', origin=origin)
    metadata = writer._metadata(point)
    assert metadata == dict(schema=1, method='innovation1-present-card-swap-v1', slot='swap-50',
                            round=7, recipe=swap_config.recipe(), source=source, run_id='toy',
                            epoch=2, step=200, bce=.5, initial_sha256=initial, geometry=geometry)
    assert writer.dump()
    with pytest.raises(ValueError, match='INVALID_DEV_STEP'):
        writer._metadata(dict(point, step=3200))
    changed = dict(method=pure.METHOD, slot='original', run_id='another', recipe=pure.recipe(),
                   source={'toy/a.py': '9'*64}, step=3200, epoch=3,
                   geometry=dict(geometry, public='0'*64), initial_sha256='0'*64)
    if fault in changed:
        metadata[fault] = changed[fault]
    state = dict(toy.state_dict(), weight=torch.tensor([.25]))
    if fault == 'frozen':
        state['frozen'] = torch.tensor([3.])
    if fault == 'buffer':
        state['indices'] = torch.tensor([3])
    raw = encode(metadata, state)

    def load(stream: io.BytesIO, *, map_location: str, weights_only: bool) -> dict[str, Any]:
        assert stream.getvalue() == raw and map_location == 'cpu' and weights_only
        return {'metadata': metadata, 'state': state}

    monkeypatch.setattr(torch, 'load', load)
    reader = FitRuntime('swap-50', arm, False, source=source, run_id='toy', origin=origin)
    if fault:
        with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
            reader.restore(raw)
        assert reader._runtime.models == reader._runtime.optimizers == {}
    else:
        reader.restore(raw)
        assert reader.geometry == geometry and reader.initial_sha == initial
        assert reader._runtime.models['swap-50'] is toy and not reader._runtime.optimizers
        assert not toy.training and toy.weight.detach().tolist() == [.25]

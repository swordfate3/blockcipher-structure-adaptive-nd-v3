from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from cipher_lab.execution import Meter


@pytest.mark.parametrize('slot,origin', [('swap-50', 0), ('swap-50-mask', 2300)])
def test_training_views(monkeypatch: pytest.MonkeyPatch, bridge_meter: Meter,
                        slot: str, origin: int) -> None:
    from cipher_lab_plugins.innovation1.models import TorchRuntime
    from cipher_lab_plugins.innovation1.prefix_bridge.runtime import Runtime as Bridge
    from cipher_lab_plugins.innovation1.present_card_fresh import config as pure
    from cipher_lab_plugins.innovation1.present_card_mask import masking
    from cipher_lab_plugins.innovation1.present_card_swap.views import swap_half
    from cipher_lab_plugins.innovation1.present_card_swap_mask import runtime

    inputs = np.arange(5000 * 128, dtype=np.float32).reshape(5000, 1, 2, 64)
    inputs.flags.writeable = False
    labels = np.zeros((len(inputs), 1), dtype=np.uint8)
    transfers: list[Any] = []
    masks: list[tuple[Any, int]] = []
    rates: list[float] = []

    def start(self: Bridge) -> None:
        self._state = 'TRAINING'
        monkeypatch.setitem(self.models, slot, None)
        monkeypatch.setitem(self.optimizers, slot, None)

    def transfer(self: Bridge, value: Any) -> Any:
        transfers.append(value)
        return value

    def prepare(value: Any, step: int) -> Any:
        masks.append((value, step))
        return value, None, None

    def update(self: Any, actual_slot: str, task: Any, x: Any, y: Any,
               lr: float, meter: Meter) -> float:
        assert actual_slot == slot and task.rounds == 7 and meter is bridge_meter
        assert x is transfers[-2] and y is labels
        if slot.endswith('-mask'):
            assert x is masks[-1][0]
            self._batch_aux = .25
        rates.append(lr)
        return .5

    monkeypatch.setattr(Bridge, 'start', start)
    monkeypatch.setattr(Bridge, '_on_device', transfer)
    monkeypatch.setattr(TorchRuntime, 'update', update)
    monkeypatch.setattr(masking, 'prepare', prepare)
    monkeypatch.setattr(bridge_meter, '_policy', replace(bridge_meter._policy,
                                                       optimizer_updates=6000))
    bridge_meter._attempted['optimizer_updates'] = origin
    bridge_meter._completed['optimizer_updates'] = origin
    arm: dict[str, object] = {}
    fit = runtime.FitRuntime(slot, arm, True, source={'toy/a.py': 'f'*64},
                             run_id='toy', origin=origin)
    with pytest.raises(ValueError, match='UPDATE_SEQUENCE'):
        fit.batch_update(inputs, labels, bridge_meter)
    assert not transfers and not masks
    for step in range(2):
        bridge_meter.reserve('optimizer_updates', 1)
        assert fit.batch_update(inputs, labels, bridge_meter) == .5
        np.testing.assert_array_equal(transfers[-2], swap_half(inputs, seed=20261128, step=step))
        bridge_meter.complete('optimizer_updates', 1)
    assert rates == [pure.learning_rate(step) for step in range(2)]
    if slot.endswith('-mask'):
        assert [step for _, step in masks] == [1, 2]
        assert arm['auxiliary_train'] == {'rows': 10000, 'targets': 160000, 'bce': .25}
    else:
        assert not masks and 'auxiliary_train' not in arm
    monkeypatch.setattr(Bridge, 'predict_inputs', lambda self, x: x)
    assert fit.predict(inputs) is inputs


@pytest.mark.parametrize('fault', ['', 'method', 'slot', 'source', 'run_id', 'recipe', 'step',
                                  'initial_sha256', 'aux_state'])
def test_full_checkpoint_identity(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    import torch
    from cipher_lab_plugins.innovation1.checkpoint import encode
    from cipher_lab_plugins.innovation1.prefix_bridge.runtime import Runtime as Bridge
    from cipher_lab_plugins.innovation1.present_card_fresh import model as pure_model
    from cipher_lab_plugins.innovation1.present_card_fresh.layout import STRUCTURE_SHA256
    from cipher_lab_plugins.innovation1.present_card_mask import model as masked_model
    from cipher_lab_plugins.innovation1.present_card_mask import runtime as masked_runtime
    from cipher_lab_plugins.innovation1.present_card_swap_mask import config, runtime
    from torch import nn

    geometry = {'public': 'a'*64, 'skinny_column': 'b'*64,
                'shared.stems.local.indices': 'd'*64, 'shared.stems.word.indices': 'e'*64,
                'shared.stems.nibble.indices': STRUCTURE_SHA256}

    def classifier() -> nn.Module:
        net = nn.Module()
        net.weight = nn.Parameter(torch.tensor([2.]))
        return net

    def masked() -> nn.Module:
        net = nn.Module()
        net.classifier = classifier()
        net.aux = nn.Parameter(torch.tensor([5.]))
        return net

    def start(self: Bridge) -> None:
        self.models[self.slot] = self._fresh_cpu()
        self._state = 'TRAINING'

    recipe = config.recipe()
    monkeypatch.setattr(pure_model, 'build', classifier)
    monkeypatch.setattr(pure_model, 'geometry', lambda net: geometry)
    monkeypatch.setattr(masked_model, 'build', masked)
    monkeypatch.setattr(masked_runtime, 'geometry', lambda net: geometry)
    monkeypatch.setattr(Bridge, 'start', start)
    point = dict(epoch=2, step=200, bce=.5)
    arm: dict[str, object] = dict(selected=point, dev=[point])
    source = {'toy/a.py': 'f'*64}
    pure = runtime.FitRuntime('swap-50', {}, True, source=source, run_id='toy', origin=0)
    writer = runtime.FitRuntime('swap-50-mask', arm, True, source=source,
                                run_id='toy', origin=2300)
    assert pure.initial_sha == pure.classifier_initial_sha == writer.classifier_initial_sha
    assert pure.initial_sha != writer.initial_sha
    assert pure.parameters == 1 and writer.parameters == 2
    assert pure.geometry == writer.geometry == geometry
    arm.update(initial_sha256=writer.initial_sha, parameters=writer.parameters, geometry=geometry)
    model = writer._runtime.models['swap-50-mask']
    model.get_parameter('aux').detach().fill_(7.)
    assert writer.dump()
    meta = writer._metadata(point)
    assert meta == dict(schema=1, method='innovation1-present-card-swap-mask-v1',
                        slot='swap-50-mask', round=7, recipe=recipe, source=source,
                        run_id='toy', epoch=2, step=200, bce=.5,
                        initial_sha256=writer.initial_sha, geometry=geometry)
    state = dict(model.state_dict())
    changes = dict(method='legacy', slot='swap-50', source={'toy/a.py': '0'*64},
                   run_id='other', recipe={}, step=2500, initial_sha256=pure.initial_sha)
    if fault in changes:
        meta[fault] = changes[fault]
    if fault == 'aux_state':
        del state['aux']
    raw = encode(meta, state)

    def load(stream: Any, *, map_location: str, weights_only: bool) -> dict[str, Any]:
        assert stream.getvalue() == raw and map_location == 'cpu' and weights_only
        return {'metadata': meta, 'state': state}

    monkeypatch.setattr(torch, 'load', load)
    reader = runtime.FitRuntime('swap-50-mask', arm, False, source=source,
                                run_id='toy', origin=2300)
    if fault:
        with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
            reader.restore(raw)
        assert reader._runtime.models == reader._runtime.optimizers == {}
    else:
        reader.restore(raw)
        assert reader.initial_sha == writer.initial_sha
        assert reader.classifier_initial_sha == pure.initial_sha
        assert reader._runtime.models['swap-50-mask'].get_parameter('aux').tolist() == [7.]

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from cipher_lab.execution import Meter


def test_schedule_admitted_before_transfer(monkeypatch: pytest.MonkeyPatch,
                                           bridge_meter: Meter) -> None:
    from cipher_lab_plugins.innovation1.models import TorchRuntime
    from cipher_lab_plugins.innovation1.present_card_fresh import config, runtime

    engine = runtime.Runtime(config.SLOT)
    engine._state = 'TRAINING'
    monkeypatch.setitem(engine.models, config.SLOT, None)
    monkeypatch.setitem(engine.optimizers, config.SLOT, None)
    transfer: list[int] = []
    rates: list[float] = []

    def device(x: Any) -> Any:
        transfer.append(len(x))
        return x

    def update(self: object, slot: str, task: Any, x: Any, y: Any, lr: float,
               meter: Meter) -> float:
        assert slot == config.SLOT and task.rounds == 7 and meter is bridge_meter
        rates.append(lr)
        return .5

    monkeypatch.setattr(engine, '_on_device', device)
    monkeypatch.setattr(TorchRuntime, 'update', update)
    monkeypatch.setattr(bridge_meter, '_policy', replace(bridge_meter._policy,
                                                       optimizer_updates=3001))
    x = np.zeros((5000, 1, 2, 64), dtype=np.float32)
    y = np.zeros((5000, 1), dtype=np.uint8)
    with pytest.raises(ValueError):
        engine.batch_update(x, y, bridge_meter)
    assert transfer == []
    for completed in (0, 1799, 2999, 3000):
        monkeypatch.setitem(bridge_meter._attempted, 'optimizer_updates', completed + 1)
        monkeypatch.setitem(bridge_meter._completed, 'optimizer_updates', completed)
        if completed == 3000:
            with pytest.raises(ValueError):
                engine.batch_update(x, y, bridge_meter)
        else:
            assert engine.batch_update(x, y, bridge_meter) == .5
    assert transfer == [5000] * 6
    assert rates == [config.learning_rate(i) for i in (0, 1799, 2999)]


@pytest.mark.parametrize('fault', ['', 'source', 'recipe', 'run_id', 'slot', 'epoch', 'step',
                                  'bce', 'initial_sha256', 'geometry', 'buffer', 'frozen',
                                  'nan', 'dtype', 'extra-state'])
def test_card_metadata_and_fixed_state_on_cpu_only(
    monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    import io

    import torch
    from cipher_lab_plugins.innovation1.checkpoint import encode
    from cipher_lab_plugins.innovation1.present_card_fresh import config, runtime
    from cipher_lab_plugins.innovation1.present_card_fresh.layout import STRUCTURE_SHA256
    from torch import nn

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.]))
            self.frozen = nn.Parameter(torch.tensor([1.]), requires_grad=False)
            self.register_buffer('indices', torch.tensor([2]))

    initial = 'c'*64
    geometry = {'public': 'a'*64, 'skinny_column': 'b'*64,
                'shared.stems.local.indices': 'd'*64, 'shared.stems.word.indices': 'e'*64,
                'shared.stems.nibble.indices': STRUCTURE_SHA256}
    arm = dict(selected=dict(epoch=2, step=200, bce=.5), initial_sha256=initial,
               geometry=geometry, parameters=3)
    fit = runtime.FitRuntime(config.SLOT, arm, False, source={'toy/a.py': 'f'*64}, run_id='toy')
    engine = fit._runtime
    toy = Toy()

    def fresh() -> Toy:
        monkeypatch.setattr(engine, '_initial', initial)
        monkeypatch.setattr(engine, '_parameters', 3)
        monkeypatch.setattr(engine, 'geometry', geometry)
        return toy

    monkeypatch.setattr(engine, '_fresh_cpu', fresh)
    actual = fit._metadata(arm['selected'])
    raw = encode(actual, toy.state_dict())
    metadata = actual.copy()
    state = dict(toy.state_dict(), weight=torch.tensor([.25]))
    changes = dict(source={'toy/a.py': '9'*64}, recipe={}, run_id='other', slot='r7-cipher',
                   epoch=3, step=201, bce=.6, initial_sha256='0'*64, geometry={})
    if fault in changes:
        metadata[fault] = changes[fault]
    if fault == 'buffer':
        state['indices'] = torch.tensor([3])
    if fault == 'frozen':
        state['frozen'] = torch.tensor([3.])
    if fault == 'nan':
        state['weight'] = torch.tensor([float('nan')])
    if fault == 'dtype':
        state['weight'] = torch.tensor([1], dtype=torch.long)
    if fault == 'extra-state':
        state['extra'] = torch.tensor(1.)

    def load(stream: io.BytesIO, *, map_location: str, weights_only: bool) -> dict[str, Any]:
        assert stream.getvalue() == raw and map_location == 'cpu' and weights_only
        return {'metadata': metadata, 'state': state}

    # No actual loader: exercise preflight and the strict checks on tiny fixture tensors.
    monkeypatch.setattr(torch, 'load', load)
    if fault:
        with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
            fit.restore(raw)
        assert engine.models == engine.optimizers == {}
    else:
        fit.restore(raw)
        assert engine.models[config.SLOT] is toy and not engine.optimizers
        assert not toy.training and toy.weight.detach().tolist() == [.25]
        fit.release()
        assert engine.models == {}

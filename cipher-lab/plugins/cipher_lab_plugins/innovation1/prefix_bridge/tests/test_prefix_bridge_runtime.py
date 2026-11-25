from __future__ import annotations

import io
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from cipher_lab_plugins.innovation1.checkpoint import encode
from cipher_lab_plugins.innovation1.models import TorchRuntime
from cipher_lab_plugins.innovation1.prefix_bridge.runtime import Runtime
from cipher_lab_plugins.innovation1.prefix_bridge.weights import identity, require_identity
from torch import nn

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Meter


def test_new_checkpoint_identity_and_mask() -> None:
    source = {'fake/a.py': 'f' * 64}
    s2 = identity('s2-only', source, 'run', 1, .5)
    cipher = identity('r2-cipher', source, 'run', 1, .5)
    joint = identity('r2-joint', source, 'run', 1, .5)
    assert s2['round'] is None and cipher['round'] == 2
    assert (s2['mask'], cipher['mask'], joint['mask']) == (7, 56, 63)
    with pytest.raises(ValueError):
        identity('r8-joint', source, 'run', 1, .5)
    with pytest.raises(ValueError, match='CHECKPOINT_IDENTITY'):
        require_identity('s2-only', {**s2, 'extra': [1, 2, 3]})
    with pytest.raises(ValueError, match='CHECKPOINT_IDENTITY'):
        require_identity('s2-only', joint)


def test_callback_uses_one_parent_update_and_no_default_slots(
    monkeypatch: pytest.MonkeyPatch, bridge_meter: Meter,
) -> None:
    runtime = Runtime('s2-only')
    assert not runtime.models and not runtime.optimizers
    prepared = object()
    monkeypatch.setattr(runtime, '_state', 'TRAINING')
    runtime.models['s2-only'] = object()  # type: ignore[assignment]
    runtime.optimizers['s2-only'] = object()
    monkeypatch.setattr(runtime, '_on_device', lambda value: prepared)
    calls: list[str] = []

    def update(self: Any, slot: str, task: Any, x: Any, y: Any, lr: float, meter: Meter) -> float:
        assert self is runtime and slot == 's2-only' and task.cipher == 'PRESENT'
        assert x is y is prepared and lr == .001 and meter is bridge_meter
        calls.append(slot)
        meter.reserve('forward_rows', 2)
        meter.complete('forward_rows', 2)
        return .75

    monkeypatch.setattr(TorchRuntime, 'update', update)
    x = np.zeros((2, 4, 6, 64), np.float32)
    y = np.zeros((2, 1), np.uint8)
    assert runtime.batch_update(x, y, bridge_meter) == .75
    assert calls == ['s2-only']
    assert bridge_meter.snapshot()['completed_forward_rows'] == 2
    assert bridge_meter.snapshot()['attempted_optimizer_updates'] == 0  # only fit charges it
    x[:, :, 3:] = 1
    with pytest.raises(ValueError, match='INPUT_MASK'):
        runtime.batch_update(x, y, bridge_meter)
    assert calls == ['s2-only']


@pytest.mark.parametrize('wrong', ['round', 'mask', 'buffer', 'sha', 'size', ''])
def test_checkpoint_strictly_restores_only_bound_artificial_cpu_state(
    wrong: str, tmp_path: Path, bridge_manual_only: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Tiny artificial state, bounded archive. Only the in-memory private test
    # loader is enabled, never a file-backed torch.load under the owner guard.
    guard, _ = bridge_manual_only

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer('indices', torch.tensor([1, 2]))
            self.weight = nn.Parameter(torch.tensor([0.]))

    runtime = Runtime('r2-joint')
    fresh = Toy()
    monkeypatch.setattr(runtime, '_fresh_cpu', lambda: fresh)
    source = {'fake/a.py': 'd' * 64}
    expected = identity('r2-joint', source, 'fake', 1, .25)
    bound = dict(expected)
    if wrong == 'mask':
        bound['mask'] = 7
    elif wrong == 'round':
        bound['round'] = 3
    toy_state = {'indices': torch.tensor([1, 3] if wrong == 'buffer' else [1, 2]),
                 'weight': torch.tensor([3.25])}
    raw = encode(bound, toy_state)
    store = RunStore.claim(tmp_path, 'fake', output_bytes=2**25, needs_models=True,
                           model_slots=('r2-joint',), source_supplier=lambda: source)
    ref = store.write_model('r2-joint', raw)
    sha = sha256(raw).hexdigest() if wrong != 'sha' else '0' * 64
    loads: list[bool] = []

    def load(stream: object, *, map_location: str, weights_only: bool) -> Any:
        assert type(stream) is io.BytesIO and map_location == 'cpu' and weights_only
        assert stream.getvalue() == raw
        loads.append(True)
        return {'metadata': bound, 'state': toy_state}

    guard.setattr(torch, 'load', load)
    if wrong == 'sha':
        with pytest.raises(ArtifactError, match='HASH_MISMATCH'):
            store.read_model(ref, sha, 2**24)
    elif wrong == 'size':
        with pytest.raises(ValueError, match='CHECKPOINT_SIZE'):
            runtime.restore_bound(bytes(2**24 + 1), expected)
    else:
        bounded = store.read_model(ref, sha, 2**24)
        if wrong:
            with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
                runtime.restore_bound(bounded, expected)
        else:
            runtime.restore_bound(bounded, expected)
            assert runtime.models['r2-joint'] is fresh and not runtime.optimizers
            assert fresh.weight.detach().tolist() == [3.25]
            assert fresh.weight.data_ptr() != toy_state['weight'].data_ptr()
    assert len(loads) == int(wrong not in ('sha', 'size'))
    if wrong:
        assert not runtime.models and not runtime.optimizers


def test_start_calls_inherited_adam_seam_once_and_predict_keeps_fp32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Toy(nn.Module):
        def to(self, *args: Any, **kwargs: Any) -> Any:
            assert kwargs == {'device': torch.device('cuda:0'), 'dtype': torch.float32}
            calls.append('artificial transfer')
            return self

    toy = Toy()
    runtime = Runtime('r3-cipher')
    monkeypatch.setattr(runtime, '_fresh_cpu', lambda: toy)

    def add_adam(self: TorchRuntime, slot: str) -> None:
        assert self is runtime and self.models == {'r3-cipher': toy} and slot == 'r3-cipher'
        assert not self.optimizers
        calls.append('artificial adam')
        self.optimizers[slot] = object()

    monkeypatch.setattr(TorchRuntime, 'add_adam', add_adam)
    runtime.start()
    inputs = np.zeros((2, 4, 6, 64), np.float32)
    mock_device = object()
    monkeypatch.setattr(runtime, '_on_device', lambda x: mock_device)

    def forward(slot: str, task: Any, value: object) -> Any:
        assert slot == 'r3-cipher' and task.cipher == 'PRESENT' and task.rounds == 3
        assert value is mock_device and not toy.training
        return torch.tensor([[1.], [-1.]], dtype=torch.float32)

    monkeypatch.setattr(runtime, '_forward', forward)
    output = runtime.predict_inputs(inputs)
    assert output.dtype == np.float32 and output.tolist() == [[1.], [-1.]]
    assert toy.training  # state restored after DEV
    assert calls == ['artificial transfer', 'artificial adam']
    runtime.release()
    assert not runtime.models and not runtime.optimizers
    with pytest.raises(ValueError, match='INVALID_RUNTIME_STATE'):
        runtime.start()

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter


@pytest.mark.parametrize('arm,fault', [
    ('a', ''), ('b', ''), ('b', 'shape'), ('b', 'aux-cpu'), ('b', 'class-cpu'),
    ('b', 'aux-invalid'), ('b', 'sync'),
])
def test_parent_runs_exactly_one_joint_backward_clip_step_and_sync(
    arm: str, fault: str, bridge_manual_only: Any, bridge_meter: Meter,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from cipher_lab_plugins.innovation1.mask_aux import config, model
    from cipher_lab_plugins.innovation1.mask_aux.masking import mask_for_step, masked_view
    from cipher_lab_plugins.innovation1.mask_aux.runtime import FitRuntime, Runtime

    guard, _ = bridge_manual_only
    calls: list[str] = []
    is_b = arm == 'b'
    slot = config.AUXILIARY if is_b else config.BASELINE
    runtime = Runtime(slot)
    x = np.zeros((2, 4, 6, 64), np.float32)
    x[:, :, 4:6] = 1
    y = np.ones((2, 1), np.uint8)
    expected, indices, targets = masked_view(x, mask_for_step(2, 1))
    class_logits = SimpleNamespace(shape=y.shape)
    selected = object()

    class AuxLogits:
        shape = (2, 4, 2, 64) if fault != 'shape' else (2, 4, 2, 63)
        dtype = torch.float32

        def reshape(self, n: int) -> AuxLogits:
            assert n == -1
            return self

        def index_select(self, dim: int, ix: Any) -> object:
            assert dim == 0 and np.array_equal(ix, indices)
            return selected

    class Loss:
        def __init__(self, val: float, terms: tuple[str, ...]) -> None:
            self.val, self.terms = val, terms

        def __rmul__(self, n: float) -> Loss:
            assert n == .1 and self.terms == ('aux',)
            return Loss(n * self.val, self.terms)

        def __add__(self, other: Loss) -> Loss:
            return Loss(self.val + other.val, self.terms + other.terms)

        def backward(self) -> None:
            assert self.terms == (('class', 'aux') if is_b else ('class',))
            calls.append('backward')

        def detach(self) -> Loss:
            return self

        def cpu(self) -> float:
            if f'{self.terms[0]}-cpu' == fault:
                raise RuntimeError('artificial host copy fault')
            return self.val

    toy = model.MaskAux.__new__(model.MaskAux)
    nn.Module.__init__(toy)

    def masked_logits(masked: Any) -> AuxLogits:
        assert masked is not x and np.array_equal(masked, expected)
        calls.append('masked-forward')
        return AuxLogits()

    monkeypatch.setattr(toy, 'masked_logits', masked_logits)
    runtime.models[slot] = toy
    optimizer = SimpleNamespace(param_groups=[{}],
                                zero_grad=lambda **kw: calls.append('zero'),
                                step=lambda: calls.append('step'))
    runtime.optimizers[slot] = optimizer
    runtime._state = 'TRAINING'

    def forward(s: str, task: Any, value: Any) -> Any:
        assert s == slot and task.rounds == 7 and value is x and not x[:, :, :3].any()
        calls.append('class-forward')
        return class_logits

    def bce(logits: Any, labels: Any) -> Loss:
        if logits is class_logits:
            assert labels is y
            calls.append('class-loss')
            return Loss(.3, ('class',))
        assert logits is selected and np.array_equal(labels, targets)
        calls.append('aux-loss')
        return Loss(float('nan') if fault == 'aux-invalid' else .7, ('aux',))

    monkeypatch.setattr(runtime, '_on_device', lambda value: value)
    monkeypatch.setattr(runtime, '_forward', forward)
    monkeypatch.setattr(runtime, '_loss', bce)
    monkeypatch.setattr(torch, 'isfinite', lambda value: SimpleNamespace(all=lambda: True))
    monkeypatch.setattr(torch, 'from_numpy', lambda v: SimpleNamespace(to=lambda **kw: v))
    guard.setattr(nn.utils, 'clip_grad_norm_', lambda *a, **kw: calls.append('clip'))
    def synchronize(device: object) -> None:
        calls.append('sync')
        if fault == 'sync':
            raise RuntimeError('artificial sync fault')

    guard.setattr(torch.cuda, 'synchronize', synchronize)
    via_fit = fault in ('aux-cpu', 'class-cpu', 'aux-invalid', 'sync')
    if fault == 'shape':
        with pytest.raises(ValueError, match='INVALID_AUX_LOGITS'):
            runtime.batch_update(x, y, bridge_meter)
        assert calls == ['zero', 'class-forward', 'class-loss', 'masked-forward', 'zero']
    elif via_fit:
        from cipher_lab_plugins.direct_block.fit import FitSettings, fit, seeded_order

        class Panel:
            def __len__(self) -> int:
                return len(x)

            def batch(self, at: Any) -> Any:
                assert sorted(at.tolist()) == [0, 1]
                return x, y

        source = {'fake/source.py': 'a' * 64}
        store = RunStore.claim(tmp_path, 'loss-fault', output_bytes=2**24,
                               source_supplier=lambda: source)
        arm_summary: dict[str, object] = {}
        engine = FitRuntime(slot, arm_summary, False, source=source, run_id='loss-fault')
        monkeypatch.setattr(engine, '_runtime', runtime)
        match = {'aux-cpu': 'artificial host copy fault', 'class-cpu': 'artificial host copy fault',
                 'aux-invalid': 'INVALID_AUX_BCE', 'sync': 'artificial sync fault'}[fault]
        with pytest.raises((RuntimeError, ValueError), match=match):
            fit(Panel(), Panel(), engine, seeded_order(2, 42), slot, arm_summary, {},
                FitSettings(1, 2, 1024), bridge_meter, store, batch_update=engine.batch_update)
        extra = ['backward', 'clip', 'step', 'sync'] if fault == 'sync' else []
        assert calls == ['zero', 'class-forward', 'class-loss', 'masked-forward', 'aux-loss',
                         *extra, 'zero']
        with pytest.raises(ValueError, match='MISSING_AUX_EPOCH'):
            runtime.take_aux_epoch()
    else:
        assert runtime.batch_update(x, y, bridge_meter) == .3
        extra = ['masked-forward', 'aux-loss'] if is_b else []
        assert calls == ['zero', 'class-forward', 'class-loss', *extra, 'backward', 'clip',
                         'step', 'sync', 'zero']
        if is_b:
            assert runtime.take_aux_epoch() == (.7, 2)
    snapshot = bridge_meter.snapshot()
    assert snapshot['attempted_forward_rows'] == (4 if is_b else 2)
    assert snapshot['completed_forward_rows'] == (4 if is_b and fault != 'shape' else 2)
    assert snapshot['attempted_optimizer_updates'] == int(via_fit)  # fit owns updates
    assert snapshot['completed_optimizer_updates'] == 0
    assert runtime._masked is None and runtime._aux is None


def test_class_only_prediction_and_shared_masked_trunk(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.mask_aux import config, model
    from cipher_lab_plugins.innovation1.mask_aux.runtime import Runtime
    from cipher_lab_plugins.innovation1.prefix_bridge.model import PrefixSharedM

    a, b = (Runtime(slot) for slot in config.SLOTS)
    first, second = a._fresh_cpu(), b._fresh_cpu()
    assert isinstance(first, PrefixSharedM) and isinstance(second, model.MaskAux)
    assert a._initial == b._initial
    assert b._parameters - a._parameters == 194
    assert second.aux_head.weight.shape == (2, 96, 1)
    for key, cpu in first.state_dict().items():
        assert cpu.data_ptr() != second.classifier.state_dict()[key].data_ptr()
    assert not a.models and not b.models
    b.models[b.slot] = second
    b._state = 'SCORING'
    calls: list[str] = []
    full = torch.empty((2, 4, 6, 64))
    masked = torch.empty_like(full)
    features = torch.empty((2, 4, 96, 64))

    def feature(view: Any) -> Any:
        calls.append('full-trunk' if view is full else 'masked-trunk')
        assert view is full or view is masked
        return features

    def head(hidden: Any) -> Any:
        calls.append('class-head')
        assert hidden is features
        return torch.tensor([[1.], [-1.]], dtype=torch.float32)

    def aux(hidden: Any) -> Any:
        calls.append('aux-head')
        assert hidden.shape == (8, 96, 64) and hidden.data_ptr() == features.data_ptr()
        return torch.empty((8, 2, 64))

    monkeypatch.setattr(second.classifier, 'features', feature)
    # Use a structural channel/shape value only; no readout executes a kernel.
    monkeypatch.setattr(second.classifier.readout, 'forward', head)
    monkeypatch.setattr(second.aux_head, 'forward', aux)
    monkeypatch.setattr(b, '_on_device', lambda value: full)
    x = np.zeros((2, 4, 6, 64), np.float32)
    assert b.predict_inputs(x).tolist() == [[1.], [-1.]]
    assert calls == ['full-trunk', 'class-head']
    assert second.masked_logits(masked).shape == (2, 4, 2, 64)
    assert calls == ['full-trunk', 'class-head', 'masked-trunk', 'aux-head']


@pytest.mark.parametrize('fault', ['', 'source', 'run', 'step', 'epoch', 'bce', 'other-arm',
                                   'old-domain', 'missing-head', 'fixed-buffer'])
def test_new_checkpoint_domain_and_strict_state(fault: str, bridge_manual_only: Any,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.checkpoint import encode
    from cipher_lab_plugins.innovation1.mask_aux import config, weights
    from cipher_lab_plugins.innovation1.mask_aux.runtime import Runtime
    from cipher_lab_plugins.innovation1.prefix_bridge import weights as old_weights

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = nn.Module()
            self.classifier.register_buffer('public', torch.tensor([1, 2]))
            self.aux_head = nn.Module()
            self.aux_head.register_parameter('weight', nn.Parameter(torch.tensor([0.])))

    guard, _ = bridge_manual_only
    fresh = Toy()
    runtime = Runtime(config.AUXILIARY)
    monkeypatch.setattr(runtime, '_fresh_cpu', lambda: fresh)
    source = {'fake/source.py': 'a' * 64}
    expected = weights.identity(config.AUXILIARY, source, 'fake', 1, .5)
    assert expected['step'] == 256
    bound = dict(expected)
    modifications: dict[str, object] = {'source': {'fake/source.py': 'b' * 64}, 'run': 'other',
                                        'step': 257, 'epoch': 2, 'bce': .4}
    if fault in modifications:
        bound['run_id' if fault == 'run' else fault] = modifications[fault]
    elif fault == 'other-arm':
        bound = weights.identity(config.BASELINE, source, 'fake', 1, .5)
    elif fault == 'old-domain':
        bound = old_weights.identity('r7-cipher', source, 'fake', 1, .5)
    state = {'classifier.public': torch.tensor([1, 3] if fault == 'fixed-buffer' else [1, 2])}
    if fault != 'missing-head':
        state['aux_head.weight'] = torch.tensor([4.])
    raw = encode(bound, state)

    def load(stream: Any, *, map_location: str, weights_only: bool) -> Any:
        assert isinstance(stream, io.BytesIO) and stream.getvalue() == raw
        assert map_location == 'cpu' and weights_only
        return {'metadata': bound, 'state': state}

    guard.setattr(torch, 'load', load)
    if fault:
        with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
            runtime.restore_bound(raw, expected)
        assert not runtime.models
    else:
        runtime.restore_bound(raw, expected)
        assert runtime.models == {config.AUXILIARY: fresh} and not runtime.optimizers
        assert fresh.aux_head.get_parameter('weight').tolist() == [4.]

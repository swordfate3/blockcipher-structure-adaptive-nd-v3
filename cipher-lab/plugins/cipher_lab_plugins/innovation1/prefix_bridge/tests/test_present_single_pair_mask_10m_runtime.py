from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter


@pytest.mark.parametrize('rows', [512, 320])
def test_mask_preserves_axis_order_and_hides_both_paths(rows: int) -> None:
    from cipher_lab_plugins.innovation1.mask_aux.masking import mask_for_step
    from cipher_lab_plugins.innovation1.mask_aux_single import config, runtime

    m = mask_for_step(rows // 4, 9766).reshape(rows, 1, 2, 64)
    x = np.zeros((rows, 1, 6, 64), np.float32)
    x[:, :, 4, ::2] = 1
    x[:, :, 5] = x[:, :, 4]
    changed = x.copy()
    changed[:, :, 3:5][m] = 1 - changed[:, :, 3:5][m]
    changed[:, :, 5] = changed[:, :, 3] != changed[:, :, 4]
    b = runtime.Runtime(config.AUXILIARY)
    u, i, a = b._prepare_mask(x, 9766)
    v, j, z = b._prepare_mask(changed, 9766)
    assert len(i) == 16*rows and not np.shares_memory(u, x)
    np.testing.assert_array_equal(i, np.flatnonzero(m))
    np.testing.assert_array_equal(i, j)
    np.testing.assert_array_equal(u, v)
    np.testing.assert_array_equal(u[:, :, :2], m)
    np.testing.assert_array_equal(u[:, :, 2], m.any(axis=2))
    np.testing.assert_array_equal(u[:, :, 3:5][~m], x[:, :, 3:5][~m])
    np.testing.assert_array_equal(u[:, :, 5][~m.any(axis=2)], x[:, :, 5][~m.any(axis=2)])
    np.testing.assert_array_equal(a, x[:, :, 3:5].reshape(-1)[i])
    np.testing.assert_array_equal(z, 1-a)
    assert np.all(m.sum(axis=(2, 3)) == 16) and np.all(m.sum(axis=2) <= 1)
    assert not x[:, :, :3].any()


@pytest.mark.parametrize('rows', [512, 320])
def test_pair_local_readout_and_no_aux_in_score(rows: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.mask_aux_single import config, runtime

    a, b = (runtime.Runtime(slot) for slot in config.SLOTS)
    ma, mb = a._fresh_cpu(), b._fresh_cpu()
    assert a._initial == b._initial and b._parameters - a._parameters == 194
    from cipher_lab_plugins.innovation1.mask_aux_single.model import PairAux

    assert isinstance(mb, PairAux)
    assert mb.aux_head.weight.shape == (2, 96, 1)
    sa, sb = ma.classifier.state_dict(), mb.classifier.state_dict()
    assert sa.keys() == sb.keys()
    for name in sa:
        assert sa[name].data_ptr() != sb[name].data_ptr()
        assert torch.equal(sa[name], sb[name])
    x = np.zeros((rows, 1, 6, 64), np.float32)
    x[:, 0, 3, 0] = np.arange(rows) % 2
    calls: list[str] = []

    def features(bits: torch.Tensor) -> torch.Tensor:
        calls.append('trunk')
        assert bits.shape == (rows // 4, 4, 6, 64)
        hidden = torch.zeros((rows // 4, 4, 96, 64))
        hidden[:, :, 0, 0] = bits[:, :, 3, 0]
        return hidden

    def base(flat: torch.Tensor) -> torch.Tensor:
        assert flat.shape == (rows, 96*64)
        calls.append('base')
        return flat[:, :1]

    def residual(flat: torch.Tensor) -> torch.Tensor:
        assert flat.shape == (rows, 96*64)
        calls.append('residual')
        return -flat[:, :1] * 2

    def aux(flat: torch.Tensor) -> torch.Tensor:
        assert flat.shape == (rows, 96, 64)
        calls.append('aux')
        return flat[:, :2]

    monkeypatch.setattr(mb.classifier, 'features', features)
    monkeypatch.setattr(mb.classifier.readout.base, 'forward', base)
    monkeypatch.setattr(mb.classifier.readout.residual, 'forward', residual)
    monkeypatch.setattr(mb.aux_head, 'forward', aux)
    monkeypatch.setattr(b, '_on_device', torch.from_numpy)
    b.models[b.slot] = mb
    b._state = 'TRAINING'
    before = b.predict_inputs(x)
    b._state = 'SCORING'
    x[3, 0, 3, 0] = 0  # Change a neighbor sharing the same temporary container.
    after = b.predict_inputs(x)
    assert before[3, 0] == -1 and after[3, 0] == 0
    np.testing.assert_array_equal(np.flatnonzero(before[:, 0] != after[:, 0]), [3])
    assert calls == ['trunk', 'base', 'residual'] * 2
    assert mb.masked_logits(torch.from_numpy(x)).shape == (rows, 1, 2, 64)
    assert calls[-2:] == ['trunk', 'aux']


@pytest.mark.parametrize('arm,fault', [('a', ''), ('a', 'class'), ('b', ''), ('b', 'sync')])
def test_single_update_path(
    arm: str, fault: str, bridge_meter: Meter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.fit import FitSettings, fit, seeded_order
    from cipher_lab_plugins.innovation1.mask_aux_single import config, model, runtime

    rows, is_b = 320, arm == 'b'
    r = runtime.Runtime(config.AUXILIARY if is_b else config.BASELINE)
    x = np.zeros((rows, 1, 6, 64), np.float32)
    y = np.ones((rows, 1), np.uint8)
    calls: list[str] = []
    class_logits = SimpleNamespace(shape=y.shape)
    aux_loss_logits = object()
    masked, indices, targets = r._prepare_mask(x, 1)

    class AuxLogits:
        shape = (rows, 1, 2, 64)
        dtype = torch.float32

        def reshape(self, n: int) -> AuxLogits:
            assert n == -1
            return self

        def index_select(self, dim: int, values: Any) -> object:
            assert dim == 0 and np.array_equal(values, indices)
            return aux_loss_logits

    class Loss:
        def __init__(self, val: float) -> None:
            self.val = val

        def __rmul__(self, multiplier: float) -> Loss:
            assert multiplier == .1
            return Loss(multiplier * self.val)

        def __add__(self, other: Loss) -> Loss:
            return Loss(self.val + other.val)

        def detach(self) -> Loss:
            return self

        def cpu(self) -> float:
            if fault == 'class':
                raise RuntimeError('class host copy')
            return self.val

        def backward(self) -> None:
            assert self.val == pytest.approx(.37 if is_b else .3)
            calls.append('backward')

    def loss(logits: Any, value: Any) -> Loss:
        if logits is class_logits:
            assert value is y
            return Loss(.3)
        assert is_b and logits is aux_loss_logits and np.array_equal(value, targets)
        return Loss(.7)

    def masked_logits(value: Any) -> AuxLogits:
        assert is_b and np.array_equal(value, masked)
        calls.append('mask')
        return AuxLogits()

    def sync(device: Any) -> None:
        calls.append('sync')
        if fault == 'sync':
            raise RuntimeError('synchronized failure')

    def predict(x: Any) -> Any:
        assert not fault
        calls.append('dev')
        return np.zeros_like(y, np.float32)

    class Panel:
        def __len__(self) -> int:
            return rows

        def batch(self, at: Any) -> Any:
            assert len(at) == rows
            return x, y

    net = model.PairAux.__new__(model.PairAux)
    torch.nn.Module.__init__(net)
    monkeypatch.setattr(net, 'masked_logits', masked_logits)
    r.models[r.slot] = net
    r.optimizers[r.slot] = SimpleNamespace(param_groups=[{}], zero_grad=lambda **kw: None,
                                           step=lambda: calls.append('step'))
    r._state = 'TRAINING'
    monkeypatch.setattr(r, '_forward', lambda *a: class_logits)
    monkeypatch.setattr(r, '_on_device', lambda v: v)
    monkeypatch.setattr(r, '_loss', loss)
    monkeypatch.setattr(torch, 'isfinite', lambda v: SimpleNamespace(all=lambda: True))
    monkeypatch.setattr(torch, 'from_numpy', lambda v: SimpleNamespace(to=lambda **kw: v))
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', lambda *a, **kw: calls.append('clip'))
    monkeypatch.setattr(torch.cuda, 'synchronize', sync)
    source = {'artificial/source.py': 'a' * 64}
    store = RunStore.claim(tmp_path, 'runtime', output_bytes=2**20, needs_models=True,
                           model_slots=(r.slot,), source_supplier=lambda: source)
    point: dict[str, Any] = {}
    e = runtime.FitRuntime(r.slot, point, False, source=source, run_id='runtime')
    monkeypatch.setattr(e, '_runtime', r)
    monkeypatch.setattr(e, 'dump', lambda: b'artificial')
    monkeypatch.setattr(e, 'predict', predict)

    def execute() -> None:
        fit(Panel(), Panel(), e, seeded_order(rows, 7), r.slot, point, {},
            FitSettings(1, 512, 1024, allow_partial_batch=True), bridge_meter, store,
            batch_update=e.batch_update)

    if fault:
        with pytest.raises(RuntimeError, match='class host copy|synchronized failure'):
            execute()
        assert 'dev' not in calls
    else:
        execute()
        assert calls[-1] == 'dev' and point['selected']['train_bce'] == .3
        assert point['epochs_completed'] == 1
        if is_b:
            assert e.take_aux_epoch() == (.7, rows)
    assert calls.count('mask') == int(is_b)
    assert calls.count('step') == calls.count('backward') == int(fault != 'class')
    assert r._masked is None and r._aux is None
    counts = bridge_meter.snapshot()
    assert counts['attempted_optimizer_updates'] == 1
    assert counts['completed_optimizer_updates'] == int(not fault)
    assert counts['completed_forward_rows'] == rows * (1+int(is_b)+int(not fault))


def test_checkpoint_domain() -> None:
    from cipher_lab_plugins.innovation1.mask_aux import config as old_config
    from cipher_lab_plugins.innovation1.mask_aux import weights as old
    from cipher_lab_plugins.innovation1.mask_aux_single import config, weights

    source = {'fake/source.py': 'a' * 64}
    meta = weights.identity(config.AUXILIARY, source, 'fake', 30, .5)
    assert meta['step'] == 292_980 and meta['layout'] == 'pair-classifier+position-aux'
    weights.require_identity(config.AUXILIARY, meta)
    with pytest.raises(ValueError, match='CHECKPOINT_IDENTITY'):
        weights.require_identity(config.BASELINE, meta)
    for bad in (dict(meta, step=292_950), dict(meta, bce=True),
                old.identity(old_config.AUXILIARY, source, 'fake', 30, .5)):
        with pytest.raises(ValueError, match='CHECKPOINT_IDENTITY'):
            weights.require_identity(config.AUXILIARY, bad)


@pytest.mark.parametrize('fault', ['', 'other-arm', 'old-domain', 'source', 'run',
                                  'epoch', 'bce', 'fixed-buffer', 'missing-head'])
def test_strict_pair_restore(
    fault: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io

    from cipher_lab_plugins.innovation1.checkpoint import encode
    from cipher_lab_plugins.innovation1.mask_aux import config as old_config
    from cipher_lab_plugins.innovation1.mask_aux import weights as old_weights
    from cipher_lab_plugins.innovation1.mask_aux_single import config, runtime, weights

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer('card', torch.tensor([0, 1]))
            self.register_parameter('head', torch.nn.Parameter(torch.tensor([0.])))

    source = {'artificial/owner.py': 'a'*64}
    slot = config.AUXILIARY
    state = {'card': torch.tensor([0, 2 if fault == 'fixed-buffer' else 1])}
    if fault != 'missing-head':
        state['head'] = torch.tensor([2.])
    bound = weights.identity(
        config.BASELINE if fault == 'other-arm' else slot,
        {'artificial/owner.py': 'b'*64} if fault == 'source' else source,
        'other' if fault == 'run' else 'fake',
        2 if fault == 'epoch' else 1, .4 if fault == 'bce' else .5,
    )
    if fault == 'old-domain':
        bound = old_weights.identity(old_config.AUXILIARY, source, 'fake', 1, .5)
    raw = encode(bound, state)

    def load(stream: io.BytesIO, *, map_location: str, weights_only: bool) -> dict[str, object]:
        assert stream.getvalue() == raw and map_location == 'cpu' and weights_only
        return {'metadata': bound, 'state': state}

    e = runtime.FitRuntime(slot, {'selected': {'epoch': 1, 'step': 9766, 'bce': .5},
                                 'initial_sha256': '', 'parameters': 0}, False,
                           source=source, run_id='fake')
    net = Toy()
    monkeypatch.setattr(e._runtime, '_fresh_cpu', lambda: net)
    monkeypatch.setattr(torch, 'load', load)
    if fault:
        with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
            e.restore(raw)
        assert not e._runtime.models
    else:
        e.restore(raw)
        assert e._runtime.models[slot] is net
        assert net.get_parameter('head').tolist() == [2.]
    assert not e._runtime.optimizers

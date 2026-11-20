"""Layout-training contract: only artificial oracle, inputs, scores, and stores."""
from __future__ import annotations

import io
import json
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from plugin_isolation import forbidden

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Clocks, Meter, Stopped

_PANEL_FAILURES = {'dev-write', 'test-write', 'test-registration'}
_EARLY_CASES = {'early-stop', 'early-ceiling', 'early-reload-fails', 'early-count-mismatch',
                'early-wall-stop'}
_EARLY_CONFIG = Path('configs/innovation1-skinny-layout-early-stop.json')
_STORAGE_FAILURES = _PANEL_FAILURES | {'batch-write', 'partial-snapshot', 'run-unwritable'}






def test_paired_statistics_from_aligned_rows() -> None:
    from cipher_lab_plugins.innovation1.layout_probe.metrics import paired_statistics

    y = np.array([[0], [1], [1], [0]])
    a = np.zeros_like(y)
    b = np.array([[-2.], [2.], [0.], [-4.]])
    result = paired_statistics(y, a, b)
    d = np.logaddexp(0, np.where(y == 1, -b, b)) - math.log(2)
    d0, d1 = d[y == 0], d[y == 1]
    se = .5 * math.sqrt((float(d0.var(ddof=1)) + float(d1.var(ddof=1))) / 2)
    mean = float(d.mean())
    assert result['delta_bce'] == pytest.approx(mean)
    assert result['standard_error'] == pytest.approx(se)
    assert result['ci95'] == pytest.approx([mean - 1.95996398454 * se, mean + 1.95996398454 * se])
    assert result['delta_acc_pp'] == 50
    assert result['strata']['0']['sample_variance'] == pytest.approx(float(d0.var(ddof=1)))
    for la, lb, direction in ((a, a, 'uncertain'), (a, (2*y-1)*4, 'lower'),
                              ((2*y-1)*4, a, 'higher')):
        row = paired_statistics(y, la, lb)
        assert row['direction'] == direction and row['standard_error'] == 0
    for labels, left, right in ((y, a[:-1], b), (y+2, a, b), (y[:2], a[:2], b[:2]),
                                 (y, a, b+np.nan), (np.zeros_like(y), a, b)):
        with pytest.raises(ValueError):
            paired_statistics(labels, left, right)


def test_fp32_guard_precedes_build_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.layout_probe import runtime

    monkeypatch.setattr(torch, 'get_default_dtype', lambda: torch.float64)
    monkeypatch.setattr(runtime, 'build_pair', forbidden)
    with pytest.raises(ValueError, match='FP32'):
        runtime.Runtime(lambda: None)


@pytest.mark.parametrize('case', ['success', 'stop', 'second-fails', 'disk-tamper', 'wrong-table',
                                  'dropped-table', 'cross-slot', 'both-fail', 'mutated',
                                  'reload-stop', 'interrupt', 'dev-write', 'test-write',
                                  'test-registration', 'batch-write', 'partial-snapshot',
                                  'run-unwritable', 'invalid-artifact', *sorted(_EARLY_CASES)])
def test_layout_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
                          capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab_plugins.innovation1.checkpoint import encode
    from cipher_lab_plugins.innovation1.cli import _finish as finish_cli
    from cipher_lab_plugins.innovation1.layout_probe import application, weights
    from cipher_lab_plugins.innovation1.layout_probe.config import EARLY_STOP, SLOTS
    from cipher_lab_plugins.innovation1.layout_probe.data import Provider
    from cipher_lab_plugins.innovation1.layout_probe.layout import nibble_indices
    from cipher_lab_plugins.innovation1.resources import Resources
    from cipher_lab_plugins.innovation1.single_train.config import estimate, make_policy
    from cipher_lab_plugins.innovation1.single_train.data import PairBatch

    early = case in _EARLY_CASES
    base = replace(EARLY_STOP, run_id='layout-fixture')
    if not early:  # Short artificial paths exercise the same loop without the stopper.
        base = replace(base, early_stopping=None, cosine_epochs=None)

    def _finish(store: RunStore, root: Path, meter: Meter) -> tuple[int, str]:
        assert meter is resources.meter
        code = finish_cli(store, root, run_id, resources=resources)
        return code, capsys.readouterr().out

    batch_cycles = 256 if case == 'batch-write' else 2
    epochs = (36 if case == 'early-ceiling' else 40) if early else 2
    training = replace(base.training, epochs=epochs, batches_per_epoch=batch_cycles,
                       batch_groups=4, dev_groups=4, test_groups=4, eval_batch_groups=4)
    recipe = replace(base, training=training)
    run_id = 'layout-fixture'
    source = {'artificial/fixture.py': 'b'*64}
    clocks = Clocks()
    resources = Resources(Meter(make_policy(recipe, clocks), clocks, lambda: 7))
    store = RunStore.claim(tmp_path, run_id, output_bytes=2**24, needs_models=True,
                           model_slots=SLOTS, source_supplier=lambda: source)
    record_path = tmp_path / 'data' / run_id / 'run.json'
    model_paths = {s: tmp_path / 'models' / run_id / f'{s}.pt' for s in SLOTS}
    events: list[tuple[str, str]] = []
    calls: list[str] = []
    lrs: list[float] = []
    draws = 0
    carrier: tuple[torch.Tensor, torch.Tensor]
    finalize = store.finalize
    read = store.read_model
    atomic = store._atomic
    faults: list[str] = []
    failure_seals: list[bytes] = []
    last_confirmed: bytes | None = None

    def fault_atomic(target: Path, content: bytes) -> None:
        nonlocal last_confirmed
        candidate = json.loads(content) if target == record_path else None
        sealing = candidate is not None and candidate['status'] == 'FAILED'
        if sealing:
            failure_seals.append(content)
        hit = (
            (case == 'dev-write' and target.name == 'dev-SKINNY.npz')
            or (case == 'test-write' and target.name == 'test-SKINNY.npz')
            or (case == 'batch-write' and target.name == 'batches-0000.json')
            or (case == 'test-registration' and candidate is not None and not sealing
                and 'dataset/test-SKINNY.npz' in candidate['artifacts'])
            or (case == 'partial-snapshot' and candidate is not None and not sealing
                and candidate['counts'].get('completed_optimizer_updates') == 2)
            or (case == 'run-unwritable' and candidate is not None
                and (faults or resources.snapshot()['completed_forward_rows'] == 56))
        )
        if hit:
            if not faults:
                last_confirmed = record_path.read_bytes()
            faults.append(target.name)
            raise OSError('injected write failure')
        atomic(target, content)

    def tracked_finalize(record: dict[str, Any]) -> None:
        calls.append(record['status'])
        finalize(record)

    def tracked_read(ref: str, digest: str, limit: int) -> bytes:
        events.append(('disk', Path(ref).stem))
        return read(ref, digest, limit)

    monkeypatch.setattr(store, 'finalize', tracked_finalize)
    monkeypatch.setattr(store, 'read_model', tracked_read)
    if case in _STORAGE_FAILURES:
        monkeypatch.setattr(store, '_atomic', fault_atomic)
    if case == 'invalid-artifact':
        # Real store's pre-I/O type validation raises directly, without sealing.
        from cipher_lab_plugins.innovation1.layout_probe import data
        monkeypatch.setattr(data, 'encode_panel', lambda *a, **kw: None)

    def fake_bytes(n: int) -> bytes:
        nonlocal draws
        draws += 1
        return draws.to_bytes(n, 'big')

    def fake_oracle(blocks: tuple[int, ...], key: tuple[int, ...], rounds: int) -> tuple[int, ...]:
        assert rounds == 8 and key == (draws, 0) and len(blocks) == 2
        return (0, draws)

    class Panels(Provider):
        def panel(self, split: str) -> PairBatch:
            if split == 'test':
                assert [e for e in events if e[0] in ('disk', 'load')] == [
                    (action, slot) for slot in SLOTS for action in ('disk', 'load')]
                record = json.loads(record_path.read_bytes())
                assert set(record['summary']['persisted_reloads']) == set(SLOTS)
                if early:
                    assert record['summary']['training_termination'] == dict(
                        reason='EPOCH_CEILING' if case == 'early-ceiling' else 'EARLY_STOPPING',
                        completed_epochs=36)
            events.append(('panel', split))
            return super().panel(split)

    def template(slot: str) -> dict[str, torch.Tensor]:
        scan = 'row' if slot == SLOTS[0] else 'column'
        return {'weight': torch.tensor([.5]),
                'shared.stems.local.indices': torch.tensor([[0, 1]]),
                'shared.stems.word.indices': torch.tensor([[0, 1]]),
                'shared.stems.nibble.indices': nibble_indices(scan)}

    class Runtime:
        def metadata(self) -> dict[str, bool]:
            return {'artificial': True}

        def prepare(self, batch: PairBatch) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal carrier
            carrier = batch.tensors()
            return carrier

        def update(self, slot: str, task: Any, x: torch.Tensor, y: torch.Tensor, lr: float,
                   meter: Resources) -> float:
            assert x is carrier[0] and y is carrier[1] and task.cipher == 'SKINNY'
            events.append(('update', slot))
            lrs.append(lr)
            meter.reserve('forward_rows', 4)
            if slot == SLOTS[1]:
                if case == 'early-wall-stop' and len(lrs) == 71 * len(SLOTS):
                    raise Stopped('WALL_LIMIT')  # Partway through the first eligible stop epoch.
                if case == 'second-fails':
                    raise ValueError('fake failure')
                if case == 'interrupt':
                    raise KeyboardInterrupt
            if case == 'early-count-mismatch' and len(lrs) == 2:
                # Both attempted and completed must match, not merely stay under the cap.
                meter.complete('forward_rows', 3)
            else:
                meter.complete('forward_rows', 4)
            if case == 'mutated':
                x[0, 0, 0, 0] = 1 - x[0, 0, 0, 0]
            if case == 'both-fail' or (case == 'stop' and slot == SLOTS[0]):
                raise Stopped('WALL_LIMIT') if case == 'stop' else ValueError('fake failure')
            return lr

        def predict(self, slot: str, task: Any, x: torch.Tensor) -> Any:
            assert x is carrier[0] and task.rounds == 8
            # Arm A makes a sub-delta improvement at epoch 25 that must still win selection.
            gain = 1e-6 if early and len(lrs) >= 25 * batch_cycles * len(SLOTS) else 0.
            return (2*carrier[1].numpy()-1) * (1 + gain if slot == SLOTS[0] else 2)

        def checkpoint(self, slot: str, meta: dict[str, Any]) -> bytes:
            state = template(slot)
            key = 'shared.stems.nibble.indices'
            if slot == SLOTS[1]:
                if case in ('wrong-table', 'early-reload-fails'):
                    state[key] = template(SLOTS[0])[key]
                if case == 'dropped-table':
                    del state[key]
                if case == 'cross-slot':
                    meta = {**meta, 'slot': SLOTS[0]}
            return encode(meta, state)

        def release(self) -> None:
            if case == 'disk-tamper':
                model_paths[SLOTS[1]].write_bytes(b'artificial tamper')

        def load_best(self, slot: str, raw: bytes, rec: Any, src: dict[str, str], epoch: int,
                      score: float, check: Callable[[], None]) -> None:
            assert raw == model_paths[slot].read_bytes()
            weights.decode(raw, rec, src, 'artificial', slot, epoch, score, template(slot))
            if case == 'reload-stop' and slot == SLOTS[1]:
                raise Stopped('WALL_LIMIT')
            check()
            events.append(('load', slot))

    application._execute(recipe, store, resources, 'artificial',
                         lambda rec, r: (Panels(rec, fake_oracle, fake_bytes, r), Runtime()))
    frozen = record_path.read_bytes()
    record = json.loads(frozen)
    summary, counts = record['summary'], record['counts']
    if case == 'early-wall-stop':
        assert record['status'] == 'STOPPED' and summary['reason'] == 'WALL_LIMIT'
        assert 'training_termination' not in summary and ('panel', 'test') not in events
        assert len(summary['curves']) == 35
        assert counts['attempted_optimizer_updates'] == 142
        assert counts['completed_optimizer_updates'] == 141
    elif early:
        from cipher_lab_plugins.innovation1.single_train.config import learning_rate

        assert record['recipe'] == json.loads(json.dumps(recipe.to_dict()))
        assert summary['method'] == record['recipe']['method']
        assert summary['training_termination'] == dict(
            reason='EPOCH_CEILING' if case == 'early-ceiling' else 'EARLY_STOPPING',
            completed_epochs=36)
        completed = 36 * batch_cycles
        assert summary['paired_completed'] == completed
        assert summary['updates'] == dict.fromkeys(SLOTS, dict(attempted=completed,
                                                               completed=completed))
        assert [v for k, v in events if k == 'update'] == list(SLOTS) * completed
        assert [summary['best'][s]['epoch'] for s in SLOTS] == [25, 1]
        warmup = replace(base, training=replace(training, epochs=24))
        assert lrs[::2] == [learning_rate(warmup, c) for c in range(24 * batch_cycles)] + [
            .0001] * (12 * batch_cycles)
        assert lrs[::2] == lrs[1::2]
        succeeded = case in ('early-stop', 'early-ceiling')
        assert record['status'] == ('SUCCEEDED' if succeeded else 'FAILED')
        assert summary['reason'] == ('COMPLETE' if succeeded else 'EXECUTION_FAILED')
        if case == 'early-reload-fails':
            assert ('panel', 'test') not in events and 'test_scores' not in summary
            assert set(summary['persisted_reloads']) == {SLOTS[0]}
        if succeeded:
            totals = estimate(replace(recipe, training=replace(training, epochs=36)))
            for kind in ('attempted', 'completed'):
                for unit in ('real_blocks', 'forward_rows', 'optimizer_updates', 'null_words'):
                    assert counts[f'{kind}_{unit}'] == totals[unit]
            for split in ('dev', 'test'):
                panel_path = record_path.parent / summary[f'{split}_panel']['ref']
                with np.load(io.BytesIO(panel_path.read_bytes()), allow_pickle=False) as saved:
                    meta = json.loads(saved['metadata'].tobytes())
                    assert meta['run_id'] == run_id and meta['recipe'] == record['recipe']
                    assert meta['method'] == summary['method']
            # The exact same slot/state/source cannot bridge run identities.
            selected = summary['best'][SLOTS[0]]
            with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
                old = replace(recipe, run_id='other-fixture')
                weights.decode(model_paths[SLOTS[0]].read_bytes(), old, source,
                               'artificial', SLOTS[0], selected['epoch'], selected['bce'],
                               template(SLOTS[0]))
        code, _ = _finish(store, tmp_path, resources.meter)
        assert code == (0 if succeeded else 4)
        report_path = tmp_path / 'result' / run_id / 'report.md'
        report = report_path.read_text(encoding='utf-8')
        assert summary['training_termination']['reason'] in report
        assert ('ΔBCE' in report) == succeeded
    elif case == 'success':
        assert calls == ['SUCCEEDED'] and counts['completed_forward_rows'] == 56 and draws == 24
        assert [v for k, v in events if k == 'update'] == list(SLOTS) * 4
        assert [summary['best'][s]['epoch'] for s in SLOTS] == [1, 1]
        assert summary['paired_test']['delta_bce'] < 0
        assert 'm_minus_d' not in summary and 'training_termination' not in summary
        panel_path = record_path.parent / summary['test_panel']['ref']
        with np.load(io.BytesIO(panel_path.read_bytes()), allow_pickle=False) as saved:
            assert set(saved.files) == {'metadata', 'ciphertext', 'labels', 'a_logits',
                                        'a_prime_logits'}
            meta = json.loads(saved['metadata'].tobytes())
            assert meta['run_id'] == run_id and meta['weights'] == summary['persisted_reloads']
        assert _finish(store, tmp_path, resources.meter)[0] == 0
        report = (tmp_path / 'result' / run_id / 'report.md').read_text(encoding='utf-8')
        assert 'A′' in report and 'ΔBCE' in report
        monkeypatch.setattr(store, 'write_report_files', forbidden)
        assert _finish(store, tmp_path, resources.meter)[0] == 6
    elif case in _STORAGE_FAILURES or case == 'invalid-artifact':
        assert calls == []  # No application finalize/retry for either kind of ArtifactError.
        assert len(failure_seals) == int(case in _STORAGE_FAILURES)
        assert not {'test_panel', 'test_scores', 'paired_test', 'constant_bce',
                    'bce_minus_constant', 'logits'} & summary.keys()
        assert ('dataset/test-SKINNY.npz' in record['artifacts']) == (case == 'test-registration')
        if case in ('run-unwritable', 'invalid-artifact'):
            if case == 'run-unwritable':
                assert faults == ['run.json', 'run.json'] and frozen == last_confirmed
                # Byte-for-byte last confirmed pre-TEST snapshot, not the live counts.
                assert counts['completed_real_blocks'] == 40
                assert counts['completed_forward_rows'] == 48
                assert draws == 24 and resources.snapshot()['completed_forward_rows'] == 56
            else:
                assert draws == 4 and not record['artifacts']
            assert record['status'] == 'RUNNING'
            code, message = _finish(store, tmp_path, resources.meter)
            assert code == 7
            assert json.loads(message)['scientific_status'] == 'UNKNOWN'
            assert not list((tmp_path / 'result' / run_id).iterdir())
        else:
            assert len(faults) == 1 and record['status'] == 'FAILED'
            assert summary['reason'] == 'WRITE_FAILED'
            expected_counts = {'dev-write': (8, 0, 0), 'test-write': (48, 56, 8),
                               'test-registration': (48, 56, 8), 'batch-write': (2056, 2048, 512),
                               'partial-snapshot': (16, 4, 1)}[case]
            units = ('real_blocks', 'forward_rows', 'optimizer_updates')
            for kind in ('attempted', 'completed'):
                assert tuple(counts[f'{kind}_{unit}'] for unit in units) == expected_counts
            assert _finish(store, tmp_path, resources.meter)[0] == 4
            if case in ('batch-write', 'partial-snapshot'):
                assert 'payload/batches-0000.json' not in record['artifacts']
                if not hasattr(store, '_failure_evidence_key'):
                    assert 'pending' not in summary  # main core retains only bounded identity
                    return
                actual = [(r['epoch'], r['cycle'], r['groups'], r['updates_completed'])
                          for r in summary['pending']]
                assert actual == ([(1, cycle, 4, list(SLOTS)) for cycle in range(256)]
                                  if case == 'batch-write' else [(1, 0, 4, [SLOTS[0]])])
    else:
        expected = 'STOPPED' if case in ('stop', 'reload-stop', 'interrupt') else 'FAILED'
        assert record['status'] == expected
        assert calls == ([] if case == 'disk-tamper' else [expected])
        assert ('panel', 'test') not in events and 'test_scores' not in summary
        if case in ('stop', 'second-fails', 'both-fail', 'mutated', 'interrupt'):
            complete = int(case in ('second-fails', 'mutated', 'interrupt'))
            attempts = 1 + int(case in ('second-fails', 'interrupt'))
            assert counts['completed_optimizer_updates'] == complete
            assert counts['attempted_optimizer_updates'] == attempts
            if case == 'interrupt':
                assert summary['reason'] == 'INTERRUPTED'
        else:
            assert ('load', SLOTS[0]) in events and ('load', SLOTS[1]) not in events
            assert set(summary.get('persisted_reloads', {})) == (
                set() if case == 'disk-tamper' else {SLOTS[0]})
        if case != 'disk-tamper':
            code, _ = _finish(store, tmp_path, resources.meter)
            assert code == {'FAILED': 4, 'STOPPED': 3}[expected]
            report = (tmp_path / 'result' / run_id / 'report.md').read_text(encoding='utf-8')
            assert '未成功封存，不报告 TEST 成绩' in report and 'ΔBCE' not in report
    assert record_path.read_bytes() == frozen and counts['completed_null_words'] == 0


def test_namespace_streams_and_panel_bytes() -> None:
    from cipher_lab_plugins.innovation1.layout_probe.config import EARLY_STOP, NAMESPACE, SEEDS
    from cipher_lab_plugins.innovation1.layout_probe.data import Provider, encode_panel
    from cipher_lab_plugins.innovation1.single_train.data import PairBatch

    training = replace(EARLY_STOP.training, dev_groups=4, eval_batch_groups=4)
    layout = replace(EARLY_STOP, training=training, run_id='first')
    other = replace(layout, run_id='second')
    assert (SEEDS.init, SEEDS.sample, SEEDS.shuffle) == (26111951, 26111952, 26111953)
    received: list[tuple[int, ...]] = []

    def oracle(blocks: tuple[int, ...], words: tuple[int, ...], rounds: int) -> tuple[int, ...]:
        received.append(blocks)
        return (0, 0)

    class MeterStub:
        def reserve(self, kind: str, amount: int) -> None:
            assert (kind, amount) == ('real_blocks', 2)

        complete = reserve

        def check(self) -> None:
            pass

    source = {'artificial/fixture.py': 'b'*64}
    a = Provider(layout, oracle, bytes, MeterStub()).panel('dev')
    public_a = received[:]
    received.clear()
    b = Provider(other, oracle, bytes, MeterStub()).panel('dev')
    assert public_a == received and a.digest == b.digest
    assert a.namespace == a.slice(0, 4).namespace == NAMESPACE
    wrong = PairBatch(a.ciphertext, a.labels, 'dev', 0, 'other-namespace')
    assert wrong.digest != a.digest
    with pytest.raises(ValueError, match='PANEL_IDENTITY'):
        encode_panel(wrong, layout, source, 'artificial')
    assert encode_panel(a, layout, source, 'artificial') != (
        encode_panel(b, other, source, 'artificial'))


def test_runtime_uses_two_fresh_slot_templates(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.checkpoint import encode
    from cipher_lab_plugins.innovation1.layout_probe import runtime, weights
    from cipher_lab_plugins.innovation1.layout_probe.config import EARLY_STOP, SEEDS, SLOTS
    from cipher_lab_plugins.innovation1.layout_probe.layout import nibble_indices
    from cipher_lab_plugins.innovation1.models import TorchRuntime
    from cipher_lab_plugins.innovation1.single_train.config import SingleRecipe

    recipe = replace(EARLY_STOP, run_id='templates-fixture')

    class Table(torch.nn.Module):
        def __init__(self, indices: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer('indices', indices)

    class Arm(torch.nn.Module):
        def __init__(self, scan: str) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([.5]))
            self.shared = torch.nn.Module()
            self.shared.stems = torch.nn.ModuleDict(dict(
                local=Table(torch.tensor([0, 1])), word=Table(torch.tensor([1, 0])),
                nibble=Table(nibble_indices(scan))))

        def to(self, *args: Any, **kwargs: Any) -> Arm:
            assert str(kwargs['device']) == 'cuda:0' and kwargs['dtype'] == torch.float32
            return self  # no GPU transfer

    pairs: list[tuple[Arm, Arm]] = []
    opts: list[tuple[list[torch.nn.Parameter], dict[str, Any], object]] = []

    def build(*, init_seed: int) -> tuple[Arm, Arm]:
        assert init_seed == SEEDS.init
        arms = Arm('row'), Arm('column')
        pairs.append(arms)
        return arms

    def optimizer(parameters: Any, **kwargs: Any) -> object:
        instance = object()
        opts.append((list(parameters), kwargs, instance))
        return instance

    tensor = torch.tensor

    def public(data: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        device = str(kwargs.get('device', 'cpu'))
        assert device in ('cpu', 'cuda:0')
        if device == 'cuda:0':
            del kwargs['device']
        return tensor(data, *args, **kwargs)

    monkeypatch.setattr(runtime, 'build_pair', build)
    monkeypatch.setattr(torch, 'tensor', public)
    monkeypatch.setattr(torch.optim, 'Adam', optimizer)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    r = runtime.Runtime(lambda: None)
    assert runtime.Runtime.update is TorchRuntime.update
    assert runtime.Runtime.predict is TorchRuntime.predict
    assert runtime.Runtime.metadata is TorchRuntime.metadata
    assert list(r.models.values()) == list(pairs[0])
    assert opts[0][0][0] is pairs[0][0].weight and opts[1][0][0] is pairs[0][1].weight
    assert opts[0][2] is not opts[1][2]
    assert all(kw == dict(lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0., amsgrad=False)
               for _, kw, _ in opts)
    assert len(pairs) == 1
    source = {'artificial/fixture.py': 'c'*64}
    blobs = [r.checkpoint(slot, weights.identity(recipe, source, 'ciphers', slot, 1, .25))
             for slot in SLOTS]
    with pytest.raises(ValueError, match='RELOAD_NOT_RELEASED'):
        r.load_best(SLOTS[0], blobs[0], recipe, source, 1, .25, lambda: None)
    r.release()
    for slot, raw in zip(SLOTS, blobs, strict=True):
        r.load_best(slot, raw, recipe, source, 1, .25, lambda: None)
    assert len(pairs) == 2 and list(r.models.values()) == list(pairs[1])
    assert all(not arm.training for arm in pairs[1]) and not r.optimizers
    for train, reloaded in zip(*pairs, strict=True):
        assert train is not reloaded and train.weight is not reloaded.weight
    state = pairs[1][0].state_dict()
    with pytest.raises(ValueError, match='CHECKPOINT_IDENTITY'):
        weights.decode(blobs[0], SingleRecipe(), source, 'ciphers', SLOTS[0], 1, .25, state)
    for field, value in (('schema', 1), ('epoch', True), ('score', .5), ('source', {}),
                         ('recipe', SingleRecipe().to_dict()), ('backend', 'artificial'),
                         ('run_id', 'old'), ('method', 'old')):
        meta = weights.identity(recipe, source, 'ciphers', SLOTS[0], 1, .25)
        meta[field] = value
        with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
            weights.decode(encode(meta, state), recipe, source, 'ciphers', SLOTS[0], 1, .25, state)
    for name in weights.FIXED_BUFFERS:
        for slot, arm in zip(SLOTS, pairs[1], strict=True):
            template = arm.state_dict()
            wrong = {**template, name: template[name] + 1}
            meta = weights.identity(recipe, source, 'ciphers', slot, 1, .25)
            with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
                blob = encode(meta, wrong)
                weights.decode(blob, recipe, source, 'ciphers', slot, 1, .25, template)


def test_layout_source_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    import stat
    from hashlib import sha256
    from types import SimpleNamespace

    from cipher_lab_plugins.innovation1 import source

    from cipher_lab.artifacts import ArtifactError

    layout = source.source_identity()
    assert 'cipher_lab/training/early_stopping.py' in layout
    folder = Path(source.__file__).parent / 'layout_probe'
    expected = {p.relative_to(folder.parents[2]).as_posix(): sha256(p.read_bytes()).hexdigest()
                for p in folder.glob('*.py')}
    assert {key: layout[key] for key in layout.keys() if '/layout_probe/' in key} == expected
    orig = Path.lstat

    def lstat(path: Path) -> Any:
        if path == folder / 'runtime.py':
            return SimpleNamespace(st_mode=stat.S_IFREG,
                                   st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return orig(path)

    monkeypatch.setattr(Path, 'lstat', lstat)
    with pytest.raises(ArtifactError, match='SOURCE_UNAVAILABLE'):
        source.source_identity()


def test_early_recipe_exact_budget_and_schedule(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.config import load_config
    from cipher_lab_plugins.innovation1.layout_probe.config import EARLY_STOP, LayoutRecipe
    from cipher_lab_plugins.innovation1.single_train.config import estimate, learning_rate

    recipe = load_config(_EARLY_CONFIG)
    assert isinstance(recipe, LayoutRecipe)
    payload = json.loads(_EARLY_CONFIG.read_bytes())
    assert payload['selection']['early_stopping'] == dict(
        warmup_epochs=24, patience=12, min_delta=1e-5, stop_when='all')
    assert payload['optimizer']['cosine_epochs'] == 24
    assert payload['method'] == 'innovation1-skinny-layout-early-stop-v1'
    assert payload['run_id'] is None
    assert payload['namespace'] == 'innovation1-skinny-layout-early-stop-v1-20261120'
    assert payload['seeds'] == dict(init=26111951, sample=26111952, shuffle=26111953)
    assert payload['resources'] == vars(EARLY_STOP.resources)
    warmup = replace(recipe, training=replace(recipe.training, epochs=24))
    assert estimate(recipe) == dict(fits=2, retries=0, cycles=102400, updates_each=102400,
                                    optimizer_updates=204800, real_blocks=105512960,
                                    forward_rows=118489088, train_groups=52428800,
                                    dev_groups=65536, test_groups=262144, null_words=0)
    for cycle in range(102400):
        lr = learning_rate(recipe, cycle, cosine_epochs=recipe.cosine_epochs)
        assert lr == (learning_rate(warmup, cycle) if cycle < 24576 else .0001)
    invalid = tmp_path / 'invalid.json'
    text = _EARLY_CONFIG.read_text()
    for changed in (text.replace('"all"', '"any"'),
                    text.replace('"patience": 12', '"patience": 12.0'),
                    text.replace('"min_delta":', '"extra": 0, "min_delta":')):
        assert changed != text
        invalid.write_text(changed)
        with pytest.raises(ValueError, match='CONFIG_EXACT'):
            load_config(invalid)

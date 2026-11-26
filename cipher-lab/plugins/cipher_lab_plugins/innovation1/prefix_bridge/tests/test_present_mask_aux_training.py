from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cipher_lab.artifacts import ArtifactError, RunStore, read_saved
from cipher_lab.execution import Meter, Stopped


@pytest.mark.parametrize('fault', [
    '', 'gate-count', 'capacity', 'second-fit', 'second-sha', 'second-restore', 'second-score',
    'final-source', 'final-deadline', 'final-write', 'invalid-bce', 'best-count-write',
    'final-count-write',
])
def test_two_best_restores_and_success_only_test(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bridge_meter: Meter,
) -> None:
    from cipher_lab_plugins.innovation1.mask_aux import config, data, training
    from cipher_lab_plugins.innovation1.prefix_bridge import config as bridge

    for name, value in (('TRAIN_ROWS', 4), ('DEV_ROWS', 2), ('TEST_ROWS', 2), ('BATCH_SIZE', 2)):
        monkeypatch.setattr(config, name, value)
        monkeypatch.setattr(bridge, name, value)
    meter = bridge_meter
    capacity = 67 if fault == 'capacity' else 68
    monkeypatch.setattr(meter, '_policy', replace(meter._policy, real_blocks=capacity))
    source = {'artificial/a.py': 'a' * 64}
    store = RunStore.claim(tmp_path, 'once', output_bytes=2**24, needs_models=True,
                           model_slots=config.SLOTS, source_supplier=lambda: dict(source))
    store.write_snapshot({'recipe': config.recipe()})
    path = tmp_path / 'data/once/run.json'
    summary: dict[str, object] = {}
    calls: list[str] = []
    updates: dict[str, list[bytes]] = {slot: [] for slot in config.SLOTS}
    holdout: list[bytes] = []
    test_labels: list[int] = []
    samples = 0

    def gate(m: Meter) -> None:
        calls.append('gate')
        if fault != 'gate-count':
            m.reserve('real_blocks', 4)
            m.complete('real_blocks', 4)

    def draw(bits: int) -> int:
        assert calls == ['gate'] and bits == 80
        calls.append('draw')
        return 55

    def sample(rng: Any, rows: int) -> Any:
        nonlocal samples
        if samples == 2:
            assert [c for c in calls if c.startswith('restored:')] == [
                f'restored:{slot}' for slot in config.SLOTS]
            assert [c for c in calls if c.endswith(':True')] == [
                event for slot in config.SLOTS
                for event in (f'make:{slot}:True', f'released:{slot}:True')]
            calls.append('test')
        samples += 1
        return np.arange(rows*8, dtype=np.uint64).reshape(rows, 4, 2) + np.uint64(100*samples)

    def encrypt(p: Any, k: Any, r: tuple[int, ...]) -> Any:
        assert r == (7,) and k == (55, 0, 0)
        if 'test' in calls:
            pairs = p.reshape(-1, 4, 2)
            test_labels.extend(int(v) for v in (pairs[:, 0, 0] ^ pairs[:, 0, 1]) == 0xD00000)
        return (p ^ np.uint64(13)).reshape(-1, 1)

    class Runtime:
        initial_sha = 'b'*64

        def __init__(self, slot: str, arm: dict[str, object], train: bool) -> None:
            self.slot, self.arm, self.train = slot, arm, train
            self.b = slot == config.AUXILIARY
            self.parameters = 2000 + (config.AUX_PARAMETERS if self.b else 0)
            self.aux_rows = 0
            self.aux_epochs = 0

        def batch_update(self, x: Any, y: Any, m: Meter) -> float:
            assert self.train and 'test' not in calls and not x[:, :, :3].any()
            updates[self.slot].append(x.tobytes() + y.tobytes())
            if fault == 'second-fit' and self.b:
                raise ValueError('fit boundary')
            for _ in range(2 if self.b else 1):
                m.reserve('forward_rows', len(x))
                m.complete('forward_rows', len(x))
            self.aux_rows += len(x)
            return float('nan') if fault == 'invalid-bce' and self.b else .25

        def take_aux_epoch(self) -> tuple[float, int]:
            assert self.train and self.b
            rows, self.aux_rows = self.aux_rows, 0
            self.aux_epochs += 1
            return 1. / self.aux_epochs, rows  # improves throughout; should not select on this

        def predict(self, x: Any) -> Any:
            assert not x[:, :, :3].any()
            if not self.train:
                assert 'test' in calls
                holdout.append(x.tobytes())
                if self.b and fault == 'second-score':
                    return np.full((len(x), 1), np.nan, np.float32)
                if self.b:
                    return np.array([[-1.], [1.]], np.float32)
                return np.zeros((2, 1), np.float32)
            assert 'test' not in calls
            return np.zeros((len(x), 1), np.float32)

        def dump(self) -> bytes:
            saved = json.loads(path.read_text())
            assert saved['counts']['completed_optimizer_updates'] == (38 if self.b else 2)
            assert 'test' not in saved['summary']
            return self.slot.encode()

        def restore(self, raw: bytes) -> None:
            assert not self.train and 'test' not in calls and raw == self.slot.encode()
            if self.b and fault == 'second-restore':
                raise ValueError('restore boundary')
            calls.append(f'restored:{self.slot}')

        def activate(self) -> None:
            assert 'test' in calls and self.arm['best_reloaded'] is True

        def release(self) -> None:
            calls.append(f'released:{self.slot}:{self.train}')

    def make(slot: str, arm: dict[str, object], train: bool) -> Any:
        calls.append(f'make:{slot}:{train}')
        return Runtime(slot, arm, train)

    original_atomic = store._atomic
    attempted: list[dict[str, Any]] = []
    recovered: list[dict[str, Any]] = []

    def atomic(target: Path, raw: bytes) -> None:
        if target == path and fault in ('best-count-write', 'final-count-write'):
            record = json.loads(raw)
            phase = 'BEST_SAVE' if fault == 'best-count-write' else 'FINAL_ACCOUNTING'
            if not attempted and record['summary'].get('phase') == phase:
                attempted.append(record)
                raise OSError('artificial first snapshot write fault')
            if record['status'] == 'FAILED':
                recovered.append(record)
        if fault == 'final-write' and target == path and json.loads(raw)['status'] == 'SUCCEEDED':
            raise OSError('artificial terminal fault')
        original_atomic(target, raw)
        if fault == 'second-sha' and target.name == f'{config.AUXILIARY}.pt':
            target.write_bytes(b'artificial corruption')

    monkeypatch.setattr(store, '_atomic', atomic)

    def current_source() -> dict[str, str]:
        if fault == 'final-source' and summary.get('phase') == 'FINALIZE':
            return {'artificial/a.py': 'c' * 64}
        return dict(source)

    monkeypatch.setattr(store, 'current_source_identity', current_source)
    original_check = meter.check

    def check() -> None:
        if fault == 'final-deadline' and summary.get('phase') == 'FINALIZE':
            raise Stopped('DEADLINE')
        original_check()

    monkeypatch.setattr(meter, 'check', check)

    def run() -> None:
        training.run(meter, store, summary, gate=gate, make=make,
                     make_panels=lambda m: data.make_panels(m, draw, encrypt, sample=sample),
                     peaks=lambda: {'artificial': 1})

    if fault:
        problems = {'gate-count': 'COUNT_MISMATCH', 'capacity': 'REAL_BLOCKS_LIMIT',
                    'second-fit': 'fit boundary', 'second-sha': 'HASH_MISMATCH',
                    'second-restore': 'restore boundary', 'second-score': 'invalid logits',
                    'final-source': 'SOURCE_CHANGED', 'final-deadline': 'DEADLINE',
                    'final-write': 'WRITE_FAILED', 'invalid-bce': 'invalid training BCE',
                    'best-count-write': 'WRITE_FAILED', 'final-count-write': 'WRITE_FAILED'}
        with pytest.raises((ArtifactError, ValueError, Stopped), match=problems[fault]):
            run()
        record = json.loads(path.read_text())
        assert record['status'] != 'SUCCEEDED'
        stored_summary: dict[str, Any] = record['summary']
        for live in (stored_summary, summary):
            hashes = live.get('panel_hashes', {})
            assert 'test' not in live and isinstance(hashes, dict) and 'test' not in hashes
        assert ('test' in calls) == (fault in {'second-score', 'final-source', 'final-deadline',
                                               'final-write', 'final-count-write'})
        if fault == 'capacity':
            assert not calls and meter.snapshot()['attempted_real_blocks'] == 0
        if fault == 'gate-count':
            assert calls == ['gate']
        if fault in ('best-count-write', 'final-count-write'):
            assert len(attempted) == 1 and recovered == [record]
            assert record['status'] == 'FAILED' and stored_summary == {'reason': 'WRITE_FAILED'}
            assert record['counts'] == attempted[0]['counts'] == meter.snapshot()
            assert record['counts']['completed_optimizer_updates'] == (
                2 if fault == 'best-count-write' else 72)
            if fault == 'best-count-write':
                assert record['counts']['completed_forward_rows'] == 6
                assert record['counts']['completed_real_blocks'] == 52
                assert record['artifacts'] == {}
        if fault.startswith('final-'):
            assert record['counts']['completed_forward_rows'] == 292
            assert record['counts']['completed_real_blocks'] == 68
        if fault == 'invalid-bce':
            assert meter.snapshot()['completed_optimizer_updates'] == 37
    else:
        run()
        record = read_saved(tmp_path, 'once')
        result: Any = record['summary']
        assert record['status'] == 'SUCCEEDED' and samples == 3
        assert updates[config.BASELINE] == updates[config.AUXILIARY]
        assert holdout[0] == holdout[1] and len(holdout) == 2
        artifacts = record['artifacts']
        assert isinstance(artifacts, dict)
        assert set(artifacts) == {f'model/{slot}.pt' for slot in config.SLOTS}
        assert result['test']['forward_rows_by_slot'] == dict.fromkeys(config.SLOTS, 2)
        for arm in result['fits'].values():
            assert arm['epochs_completed'] == 18 and arm['selected']['epoch'] == 1
            assert arm['selected']['train_bce'] == .25 and arm['best_reloaded'] is True
        b = result['fits'][config.AUXILIARY]
        assert b['aux_train'][-1] == dict(epoch=18, train_bce=1/18, masked_forward_rows=4,
                                        target_bits=256)
        assert b['work']['aux_target_bits'] == 18 * 256
        assert set(result['panel_hashes']) == {'train', 'dev', 'test'}
        assert sorted(test_labels) == [0, 1]
        wins = int(test_labels[0] == 0)
        assert result['test']['b_vs_a'] == {
            'joint_only_correct': wins, 'control_only_correct': 1-wins,
            'both_correct': wins, 'both_wrong': 1-wins,
            'percentage_points': 50. if wins else -50.,
        }
        from cipher_lab_plugins.innovation1.mask_aux.report import report_files

        files = report_files(record)
        assert '计算量不匹配' in files['report.md'].decode()
        assert json.loads(files['source.json'])['run_sha256'] == record['run_sha256']
    assert not list(tmp_path.rglob('*.npz'))
    events = (path.parent / 'events.jsonl')
    if events.exists():
        for line in events.read_text().splitlines():
            assert 'test' not in json.loads(line)['summary']


@pytest.mark.parametrize('phase', ['BEST_SAVE', 'FINAL_ACCOUNTING'])
@pytest.mark.parametrize('provider', [
    'default', 'exception', 'extra', 'missing', 'bool', 'negative', 'over-completed',
    'nonfinite', 'unreasonable', 'invalid-rss', 'persistent',
])
def test_failed_snapshot_preserves_committed_counts_without_valid_opt_in(
    phase: str, provider: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bridge_meter: Meter,
) -> None:
    meter = bridge_meter
    store = RunStore.claim(tmp_path, 'counts-fallback', output_bytes=2**24,
                           source_supplier=lambda: {'artificial/a.py': 'a' * 64})
    path = tmp_path / 'data/counts-fallback/run.json'
    meter.check()
    previous = meter.snapshot()
    store._failure_evidence_key = 'pending'
    committed = {'pending': {'bounded': 'committed'}, 'backend_identity': 'ciphers',
                 'scientific_validation': False}
    store.write_snapshot({'counts': previous, 'summary': committed})
    meter.reserve('optimizer_updates', 2)
    meter.complete('optimizer_updates', 1)
    meter.reserve('forward_rows', 8)
    meter.complete('forward_rows', 6)
    fresh = meter.snapshot()
    assert fresh != previous
    candidate = dict(fresh)
    if provider == 'extra':
        candidate['summary'] = {'test': 'must not leak', 'panel_hashes': {'test': 'secret'}}
    elif provider == 'missing':
        del candidate['completed_null_words']
    elif provider == 'bool':
        candidate['completed_optimizer_updates'] = True
    elif provider == 'negative':
        candidate['completed_real_blocks'] = -1
    elif provider == 'over-completed':
        candidate['completed_forward_rows'] = 9
    elif provider == 'nonfinite':
        candidate['wall_elapsed_seconds'] = float('nan')
    elif provider == 'unreasonable':
        candidate['cpu_elapsed_seconds'] = 10**50
    elif provider == 'invalid-rss':
        candidate['rss_bytes'] = -1
    reads: list[bool] = []

    def supply() -> dict[str, object]:
        reads.append(True)
        if provider == 'exception':
            raise OSError('artificial supplier unavailable')
        return candidate

    if provider != 'default':
        store._failure_counts_supplier = supply
    writes: list[str] = []
    original = store._atomic

    def atomic(target: Path, content: bytes) -> None:
        assert target == path
        record = json.loads(content)
        writes.append(record['status'])
        if len(writes) == 1:
            assert record['summary']['phase'] == phase
            # Distinct first error; fallback cannot replace its reason.
            raise ArtifactError('OUTPUT_LIMIT')
        if provider == 'persistent':
            raise OSError('artificial persistent write failure')
        original(target, content)

    monkeypatch.setattr(store, '_atomic', atomic)
    with pytest.raises(ArtifactError, match='OUTPUT_LIMIT'):
        store.write_snapshot({'counts': fresh, 'summary': {'phase': phase, 'test': 'uncommitted'}})
    with pytest.raises(ArtifactError, match='FINALIZED'):
        store.finalize({'status': 'FAILED'})
    assert writes == ['RUNNING', 'FAILED']
    assert reads == ([] if provider == 'default' else [True])
    saved = json.loads(path.read_text())
    assert saved['counts'] == previous
    assert saved['summary'] == (committed if provider == 'persistent'
                                else {**committed, 'reason': 'OUTPUT_LIMIT'})
    assert saved['status'] == ('RUNNING' if provider == 'persistent' else 'FAILED')

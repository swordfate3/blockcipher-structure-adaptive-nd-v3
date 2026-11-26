from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cipher_lab_plugins.innovation1.prefix_bridge.metrics import Score
from cipher_lab_plugins.innovation1.prefix_bridge.metrics import score as score_groups

from cipher_lab.artifacts import ArtifactError, RunStore, read_saved
from cipher_lab.execution import Meter, Stopped


@pytest.mark.parametrize('fault', [
    '', 'gate', 'gate-count', 'data-count', 'capacity', 'forward-capacity', 'best-write',
    'disk-sha', 'strict-restore', 'test-forward', 'test-data', 'test-score',
    'source', 'identity', 'final-source', 'final-deadline', 'final-write',
])
def test_single_fit_real_owner_with_artificial_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bridge_meter: Meter, fault: str,
) -> None:
    from cipher_lab_plugins.innovation1.s2_window import config, data, kat, report, training

    monkeypatch.setattr(os, 'fsync', lambda fd: None)
    meter = bridge_meter
    monkeypatch.setattr(meter, '_policy', replace(meter._policy,
                        real_blocks=148 if fault == 'capacity' else 149,
                        forward_rows=73 if fault == 'forward-capacity' else 10000))
    for name in ('TRAIN_ROWS', 'DEV_ROWS', 'TEST_ROWS', 'BATCH_SIZE'):
        monkeypatch.setattr(config, name, 2)
    identity = {'artificial/a.py': 'a'*64}
    store = RunStore.claim(tmp_path, 'once', output_bytes=2**24, needs_models=True,
                           model_slots=config.SLOTS, source_supplier=lambda: dict(identity))
    store.write_snapshot({'recipe': config.recipe()})
    calls: list[str] = []
    fit_epochs: list[int] = []
    inverse_inputs: list[Any] = []
    sampled: list[int] = []
    setup: list[str] = []

    def gate(m: Meter) -> None:
        calls.append('gate')
        if fault == 'gate-count':
            return
        kat.check(m, lambda p, k, r: -1 if fault == 'gate' else r,
                  lambda s, k: np.array([kat._PUBLIC_POINT], dtype=np.uint64),
                  lambda p, k, r: np.array([[2, 7]], dtype=np.uint64))

    def draw(bits: int) -> int:
        assert bits == 80 and calls == ['gate']
        calls.append('draw')
        return 0x321

    def sample(rng: Any, rows: int) -> Any:
        at = len(sampled)
        if at == 2:
            assert 'restore' in calls and len(fit_epochs) == 18
            assert 'activate' not in calls
            calls.append('test')
            if fault == 'test-data':
                raise ValueError('internal-only')
        sampled.append(at)
        return np.arange(rows*8, dtype=np.uint64).reshape(rows, 4, 2) + np.uint64(100 * at)

    def inverse(s: Any, k: Any) -> Any:
        assert k == (0x321, 0, 0)
        inverse_inputs.append(s.copy())
        return s ^ np.uint64(9001)

    def forward(p: Any, k: Any, r: Any) -> Any:
        assert r == (2, 7) and k == (0x321, 0, 0)
        assert np.array_equal(p ^ np.uint64(9001), inverse_inputs[-1])
        return np.stack((inverse_inputs[-1], np.full(len(p), len(sampled) == 3, np.uint64)), axis=1)

    def panels(m: Meter) -> Any:
        if fault == 'data-count':
            m.reserve('forward_rows', 1)
            m.complete('forward_rows', 1)
        return data.make_panels(m, draw, inverse, forward, sample=sample)

    class Runtime:
        parameters = 7 if fault == 'identity' else config.PARAMETERS
        initial_sha = 'b'*64

        def __init__(self, arm: dict[str, object], train: bool) -> None:
            setup.append('fit' if train else 'restore-engine')
            self.arm, self.train = arm, train

        def batch_update(self, x: Any, y: Any, m: Meter) -> float:
            assert self.train and 'test' not in calls
            assert (x == 0).all() and y.shape == (2, 1)
            fit_epochs.append(len(fit_epochs) + 1)
            m.reserve('forward_rows', 2)
            m.complete('forward_rows', 2)
            return .5

        def dump(self) -> bytes:
            assert self.train and len(fit_epochs) == 1
            saved = json.loads((tmp_path / run_ref).read_text())
            assert saved['counts']['completed_optimizer_updates'] == 1
            assert saved['counts']['completed_forward_rows'] == 4
            assert 'test' not in saved['summary']
            if fault == 'source':
                identity['artificial/a.py'] = 'c'*64
                # Fail at the latest next source gate; never cross into TEST.
            return b'opaque fake checkpoint'

        def restore(self, raw: bytes) -> None:
            assert not self.train and 'test' not in calls and raw == b'opaque fake checkpoint'
            if fault == 'strict-restore':
                raise ValueError('internal-only')
            calls.append('restore')

        def activate(self) -> None:
            assert not self.train and 'test' in calls and self.arm['best_reloaded'] is True
            calls.append('activate')

        def predict(self, x: Any) -> Any:
            assert not x[:, :, :3].any()
            if not self.train:
                assert 'activate' in calls and np.all(x[:, :, 4, 63] == 1)
                if fault == 'test-forward':
                    return np.array([[np.inf], [0]], np.float32)
                return np.zeros((2, 1), np.float32)
            assert not x.any()
            return np.zeros((2, 1), np.float32)

        def release(self) -> None:
            setup.append('release')

    def make(slot: str, arm: dict[str, object], train: bool) -> Any:
        assert slot == config.SLOT
        return Runtime(arm, train)

    def score(logits: Any, labels: Any) -> Score:
        assert labels.shape == logits.shape == (2, 1) and int(labels.sum()) == 1
        assert logits.tolist() == [[0], [0]]
        assert calls[-3:] == ['restore', 'test', 'activate']
        calls.append('score')
        if fault == 'test-score':
            raise ValueError('internal-only')
        return score_groups(logits, labels)

    monkeypatch.setattr(training, 'score', score)

    original_atomic = store._atomic
    final_faults: list[str] = []

    def stop_finalization() -> None:
        assert meter.snapshot()['completed_forward_rows'] == 74
        assert meter.snapshot()['completed_real_blocks'] == 149
        final_faults.append(fault)

    def atomic(path: Path, raw: bytes) -> None:
        if (fault == 'final-write' and path.name == 'run.json'
                and json.loads(raw)['status'] == 'SUCCEEDED'):
            stop_finalization()
            raise OSError('internal-only')
        if fault == 'best-write' and path.suffix == '.pt':
            raise OSError('internal-only')
        original_atomic(path, raw)
        if fault == 'disk-sha' and path.suffix == '.pt':
            path.write_bytes(b'bad' + raw[3:])

    monkeypatch.setattr(store, '_atomic', atomic)
    summary: dict[str, object] = {}
    run_ref = 'data/once/run.json'
    original_check = meter.check

    def check() -> None:
        if fault == 'final-deadline' and summary.get('phase') == 'FINALIZE':
            stop_finalization()
            raise Stopped('DEADLINE')
        original_check()

    monkeypatch.setattr(meter, 'check', check)
    original_source = store.current_source_identity

    def current_source() -> dict[str, str]:
        if fault == 'final-source' and summary.get('phase') == 'FINALIZE':
            stop_finalization()
            return {'artificial/a.py': 'e' * 64}
        return original_source()

    monkeypatch.setattr(store, 'current_source_identity', current_source)

    def run() -> None:
        training.run(meter, store, summary, gate=gate, make=make, make_panels=panels,
                     peaks=lambda: {'artificial': 2})

    if fault:
        with pytest.raises((ValueError, Stopped, ArtifactError)):
            run()
        saved = json.loads((tmp_path / run_ref).read_text())
        assert 'test' not in saved['summary'] and 'internal-only' not in str(saved)
        test_refs = [ref for ref in saved['artifacts'] if ref.startswith('dataset/')]
        test_files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*.npz'))
        assert (test_refs, test_files) == ([], [])  # No registered or orphaned TEST NPZ.
        if fault in ('final-deadline', 'final-source', 'final-write'):
            assert final_faults == [fault]
            assert saved['status'] == ('FAILED' if fault == 'final-write' else 'RUNNING')
            assert saved['counts']['completed_forward_rows'] == 74
            assert saved['counts']['completed_real_blocks'] == 149
        assert 'test' not in saved['summary'].get('panel_hashes', {})
        live_hashes = summary.get('panel_hashes', {})
        assert 'test' not in summary and isinstance(live_hashes, dict) and 'test' not in live_hashes
        assert ('draw' in calls) == (fault not in ('gate', 'gate-count'))
        assert ('test' in calls) == (fault in ('test-forward', 'test-data', 'test-score',
                                               'final-deadline', 'final-source', 'final-write'))
        if fault in ('gate', 'gate-count', 'data-count'):
            assert not setup
        if fault == 'best-write':
            assert saved['status'] == 'FAILED'
            assert saved['counts']['completed_optimizer_updates'] == 1
            assert saved['counts']['completed_forward_rows'] == 4
    else:
        run()
        saved_record: Any = read_saved(tmp_path, 'once')
        assert saved_record['status'] == 'SUCCEEDED'
        assert setup == ['fit', 'release', 'restore-engine', 'release']
        assert calls == ['gate', 'draw', 'restore', 'test', 'activate', 'score']
        assert saved_record['summary']['fit']['epochs_completed'] == 18
        assert saved_record['summary']['fit']['selected']['epoch'] == 1
        assert set(saved_record['artifacts']) == {f'model/{config.SLOT}.pt'}
        assert not list(tmp_path.rglob('*.npz'))
        hashes = saved_record['summary']['panel_hashes']
        assert set(hashes) == {'train', 'dev', 'test'} and len(hashes['test']) == 64
        assert saved_record['summary']['test'] == {
            'groups': 2, 'correct_groups': 1, 'accuracy': .5, 'bce': math.log(2.),
            'by_label': {'0': {'groups': 1, 'correct': 0}, '1': {'groups': 1, 'correct': 1}},
            'confusion': {'tn': 0, 'fp': 1, 'fn': 0, 'tp': 1},
            'unit': 'group', 'reference_accuracy': .5,
        }
        assert saved_record['summary']['science']['inverse_operations'] == 48
        assert saved_record['summary']['science']['full7_forward_trajectories'] == 48
        assert saved_record['summary']['science']['charged_output_units'] == 144
        completed = tuple(saved_record['counts'][f'completed_{k}'] for k in config.CAPS)
        assert completed == (149, 0, 18, 74)
        files = report.report_files(saved_record)
        markdown = files['report.md'].decode()
        assert '1/2' in markdown and '50.0000%' in markdown and '0.69314718' in markdown
        assert '52.3071%' in markdown and '不是同期对照' in markdown
        assert json.loads(files['source.json'])['run_sha256'] == saved_record['run_sha256']

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.cube_compare import config, data
from cipher_lab_plugins.direct_block.cube_compare.tests.test_cube_reuse import (
    cube_reuse50_bind,
)
from cipher_lab_plugins.direct_block.cube_compare.tests.test_cube_reuse import (
    cube_reuse50_source as cube_reuse50_source,
)

from cipher_lab import artifacts
from cipher_lab.cli import main as dispatch
from cipher_lab.execution import Clocks


@pytest.mark.parametrize('fault', ['', 'restore2', 'disk2', 'score2', 'testsha', 'traindtype',
                                  'testdeadline', 'reportdeadline', 'count',
                                  pytest.param('disk2-once', id='fix1-disk2-once'),
                                  pytest.param('sibling', id='fix1-sibling')])
def test_cube_reuse50_manual_cli_exact_counts_and_lazy_paired_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    cube_reuse50_source: Callable[..., tuple[Path, str, dict[str, bytes]]], fault: str,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import cli, engine, preflight, resources

    root, digest, datasets = cube_reuse50_source(
        fault='dtype' if fault == 'traindtype' else '', two_test_families=True)
    cube_reuse50_bind(monkeypatch, digest, datasets)
    monkeypatch.setattr(config, 'FAMILIES', (1, 1, 2))
    same_base = fault == 'sibling'
    if same_base:
        fault = ''
    out = root if same_base else tmp_path / 'new run'
    source_run = root / 'data' / config.SOURCE_RUN_ID
    if fault == 'testsha':
        (source_run / 'dataset/test.npz').write_bytes(b'!' + datasets['test'][1:])
    events: list[str] = []
    disk_faults: list[Path] = []
    report_reads: list[str] = []
    opened = Path.open
    now = datetime(2026, 11, 23, tzinfo=UTC)
    deadline = now + timedelta(hours=3)
    orders: dict[str, list[bytes]] = {arm: [] for arm in config.ARMS}

    def track(path: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal now
        if any(path.is_relative_to(root / kind / config.SOURCE_RUN_ID)
               for kind in ('data', 'models', 'result')):
            assert path.name in ('run.json', 'train.npz', 'dev.npz', 'test.npz')
            if path.suffix == '.npz':
                events.append(f'read/{path.stem}')
            if path.name == 'test.npz':
                assert 'loaded/mlp' in events and 'loaded/sharedm' in events
                if fault == 'testdeadline':
                    now = deadline
        if path == out / 'models/reuse/sharedm-best.pt' and fault in ('disk2', 'disk2-once'):
            if 'loaded/mlp' in events and args == ('rb',) and (fault == 'disk2' or not disk_faults):
                disk_faults.append(path)
                raise OSError('manual disk failure')
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', track)
    monkeypatch.setattr(cli, 'Clocks', lambda: Clocks(lambda: 0., lambda: 0., lambda: now))
    monkeypatch.setattr(resources, 'working_set_bytes', lambda: 512)
    monkeypatch.setattr(preflight, 'gpu_gate', lambda: {'uuid': config.GPU_UUID})
    monkeypatch.setattr(preflight, 'gpu_peaks', lambda: {'manual': True})
    monkeypatch.setattr(engine, 'configure_runtime', lambda: {'manual': True})
    for module, attr in ((preflight, 'run'), (data, 'production_panels')):
        monkeypatch.setattr(module, attr, lambda *a, **k: pytest.fail('legacy science invoked'))

    class ManualReuseEngine:
        def __init__(self, arm: str) -> None:
            assert events == ['read/train', 'read/dev'] or events[-1] == 'made/mlp'
            self.arm = arm
            self.step = 0
            self.loaded = False
            events.append(f'made/{arm}')

        def train_forward(self, inputs: data.Floats) -> Callable[[data.Labels], float]:
            orders[self.arm].append(inputs.tobytes())

            def loss(labels: data.Labels) -> float:
                assert labels.shape == (8, 128) and np.array_equal(labels, 1 - inputs)
                return .125

            return loss

        def predict(self, inputs: data.Floats) -> data.Floats:
            if self.loaded:
                events.append(f'test/{self.arm}')
                if fault == 'score2' and self.arm == 'sharedm':
                    raise ValueError('manual failure')
            strength = 2 if self.arm == 'sharedm' and self.step >= 12 else 1
            return strength * (1 - 2 * inputs)

        def update(self) -> None:
            self.step += 1

        def dump(self) -> bytes:
            return f'{self.arm}/{self.step}'.encode()

        def restore(self, raw: bytes) -> None:
            expected = 1 if self.arm == 'mlp' else 12
            assert raw == f'{self.arm}/{expected}'.encode()
            if fault == 'restore2' and self.arm == 'sharedm':
                raise ValueError('manual failure')
            self.loaded = True
            events.append(f'loaded/{self.arm}')

    monkeypatch.setattr(engine, 'make_engine', ManualReuseEngine)

    def read_report(path: Path, run_id: str) -> dict[str, object]:
        nonlocal now
        assert path == out and run_id == 'reuse'
        saved = artifacts.read_saved(path, run_id)
        report_reads.append(str(saved['status']))
        if fault == 'reportdeadline':
            now = deadline
        return saved

    monkeypatch.setattr(cli, 'read_saved', read_report)
    cfg = tmp_path / 'reuse.json'
    cfg.write_text(json.dumps(config.reuse_recipe()), encoding='utf-8')
    if fault == 'count':
        actual = config.reuse_counts
        monkeypatch.setattr(config, 'reuse_counts', lambda a, b: {
            **actual(a, b), 'optimizer_updates': a + b + 1 if a == 18 else a + b})
    code = dispatch(['run', '--plugin', cli.__name__, '--config', str(cfg), '--root', str(out),
                     '--source-root', str(root), '--run-id', 'reuse', '--deadline-utc',
                     deadline.isoformat()])
    logs = capsys.readouterr()
    assert 'manual failure' not in logs.err and 'Traceback' not in logs.err
    saved = json.loads((out / 'data/reuse/run.json').read_bytes())
    assert saved['counts']['completed_real_blocks'] == saved['counts']['attempted_null_words'] == 0
    if fault:
        assert code == (3 if fault.endswith('deadline') else 4)
        if fault.endswith('deadline'):
            assert not list((out / 'result/reuse').iterdir())
        assert ('test' in saved['summary']) == (fault == 'reportdeadline')
        assert ('read/test' in events) == (fault not in (
            'restore2', 'disk2', 'disk2-once', 'traindtype'))
        if fault == 'disk2-once':
            assert disk_faults == [out / 'models/reuse/sharedm-best.pt']
            assert report_reads == ['FAILED'] and saved['summary'] == {'reason': 'READ_FAILED'}
            assert json.loads(logs.err)['reason'] == 'READ_FAILED' and logs.out == ''
            assert events[-1] == 'loaded/mlp' and 'dataset/test.npz' not in saved['artifacts']
            for name in ('index.html', 'report.md'):
                text = (out / 'result/reuse' / name).read_text(encoding='utf-8')
                assert 'Configured data source' in text and digest in text
                assert 'not evidence that all source panels were verified' in text
                assert 'previously reported' in text and 'No final TEST score' in text
                assert 'Saved active-bit accuracy' not in text
                if name == 'index.html':
                    assert 'reuse fixed TRAIN/DEV panels' in text
                    assert 'Public preflight' not in text and 'generate TEST labels' not in text
        if fault == 'score2':
            assert events[-2:] == ['test/mlp', 'test/sharedm']
            assert saved['counts']['attempted_forward_rows'] == 624
            assert saved['counts']['completed_forward_rows'] == 616
        if fault == 'traindtype':
            assert not orders['mlp'] and not orders['sharedm']
        return
    assert code == 0 and json.loads(logs.out)['status'] == 'SUCCEEDED'
    summary = saved['summary']
    assert summary['preflight']['status'] == 'SKIPPED_REUSE'
    assert summary['source'] == config.source_recipe()
    counts = saved['counts']
    for kind, n in config.reuse_counts(18, 20).items():
        assert counts[f'attempted_{kind}'] == counts[f'completed_{kind}'] == n
    for arm, epochs, best in (('mlp', 18, 1), ('sharedm', 20, 12)):
        assert summary['arms'][arm]['epochs_completed'] == epochs
        assert summary['arms'][arm]['stop_reason'] == 'EARLY_STOPPING'
        assert summary['arms'][arm]['selected']['epoch'] == best
    assert orders['mlp'] == orders['sharedm'][:18]
    assert summary['test']['denominator_bits'] == 128
    assert summary['test']['accuracy'] == {'mlp': 1., 'sharedm': 1., 'majority': .625}
    assert events == ['read/train', 'read/dev', 'made/mlp', 'made/sharedm',
                      'loaded/mlp', 'loaded/sharedm', 'read/test',
                      'test/mlp', 'test/sharedm', 'test/mlp', 'test/sharedm']
    assert set(saved['artifacts']) == {'model/mlp-best.pt', 'model/sharedm-best.pt',
                                       *(f'dataset/{s}.npz' for s in config.SPLITS)}
    for split, raw in datasets.items():
        assert (out / 'data/reuse/dataset' / f'{split}.npz').read_bytes() == raw
    for report in ('index.html', 'report.md'):
        text = (out / 'result/reuse' / report).read_text(encoding='utf-8')
        assert 'EARLY_STOPPING' in text and digest in text and 'previously' in text


@pytest.mark.parametrize('fault', ['', 'ceiling', 'restore', 'bestsha', 'missing-best', 'testsha',
                                  'source', 'score', 'stop', 'count', 'commit', 'reportbudget',
                                  'reclaim'])
def test_cube_lstm_single_fit_manual_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    cube_reuse50_source: Callable[..., tuple[Path, str, dict[str, bytes]]], fault: str,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import (
        cli,
        engine,
        preflight,
        report,
        resources,
    )
    from cipher_lab_plugins.direct_block.source import source_identity

    root, digest, datasets = cube_reuse50_source(two_test_families=True)
    cube_reuse50_bind(monkeypatch, digest, datasets)
    monkeypatch.setattr(config, 'FAMILIES', (1, 1, 2))
    if fault == 'testsha':
        (root / 'data' / config.SOURCE_RUN_ID / 'dataset/test.npz').write_bytes(
            b'!' + datasets['test'][1:])
    now = datetime(2026, 11, 24, tzinfo=UTC)
    deadline = now + timedelta(hours=3)
    events: list[str] = []
    checkpoint = root / 'models/lstm/lstm-best.pt'
    opened = Path.open
    reading_best = False

    def track(path: Path, *args: Any, **kwargs: Any) -> Any:
        if any(path.is_relative_to(root / kind / config.SOURCE_RUN_ID)
               for kind in ('data', 'models', 'result')):
            assert path.name in ('run.json', 'train.npz', 'dev.npz', 'test.npz')
            if path.suffix == '.npz':
                events.append(f'read/{path.stem}')
            if path.name == 'test.npz':
                assert events[-3:-1] == ['read/best', 'loaded']
        if path == checkpoint and args == ('rb',) and reading_best:
            events.append('read/best')
        return opened(path, *args, **kwargs)

    read_model = artifacts.RunStore.read_model

    def guarded_read(store: artifacts.RunStore, ref: str, sha: str, size: int) -> bytes:
        nonlocal reading_best
        assert ref == 'model/lstm-best.pt' and size == 16 * 2**20
        reading_best = True
        try:
            return read_model(store, ref, sha, size)
        finally:
            reading_best = False

    monkeypatch.setattr(artifacts.RunStore, 'read_model', guarded_read)
    monkeypatch.setattr(Path, 'open', track)
    monkeypatch.setattr(cli, 'Clocks', lambda: Clocks(lambda: 0., lambda: 0., lambda: now))
    monkeypatch.setattr(resources, 'working_set_bytes', lambda: 1)
    monkeypatch.setattr(preflight, 'gpu_gate', lambda: {'uuid': config.GPU_UUID})
    monkeypatch.setattr(preflight, 'gpu_peaks', lambda: {'manual': True})
    monkeypatch.setattr(engine, 'configure_runtime', lambda: {'manual': True})
    for module, attr in ((preflight, 'run'), (data, 'production_panels'), (engine, 'make_engine')):
        monkeypatch.setattr(module, attr, lambda *a, **k: pytest.fail('legacy path invoked'))
    def source() -> dict[str, str]:
        identity = source_identity()
        if fault == 'source' and 'loaded' in events:
            identity[next(iter(identity))] = 'f' * 64
        return identity

    monkeypatch.setattr(cli, 'source_identity', source)
    if fault == 'count':
        actual = config.lstm_counts
        monkeypatch.setattr(config, 'lstm_counts', lambda e: {
            **actual(e), 'optimizer_updates': e + (e == 18)})

    class ManualLSTM:
        def __init__(self) -> None:
            assert events == ['read/train', 'read/dev']
            events.append('made')
            self.step = 0

        def train_forward(self, inputs: data.Floats) -> Callable[[data.Labels], float]:
            def backward(labels: data.Labels) -> float:
                assert labels.shape == (8, 128) and np.array_equal(labels, 1 - inputs)
                return .125
            return backward

        def predict(self, inputs: data.Floats) -> data.Floats:
            nonlocal now
            if 'loaded' in events:
                events.append('predict/test')
                if fault == 'score':
                    raise ValueError('private predictor detail')
                if fault == 'stop':
                    now = deadline
            strength = 1 + (self.step * .01 if fault == 'ceiling' else self.step * 1e-6)
            return np.float32(strength) * (1 - 2 * inputs)

        def update(self) -> None:
            self.step += 1

        def dump(self) -> bytes:
            return str(self.step).encode()

        def restore(self, raw: bytes) -> None:
            assert raw == (b'50' if fault == 'ceiling' else b'18')
            if fault == 'restore':
                raise ValueError('private restore detail')
            events.append('loaded')

    monkeypatch.setattr(engine, 'make_lstm_engine', ManualLSTM)
    # Tamper after the final best write, before strict read; a sub-delta minimum
    # can replace the checkpoint at epoch 18 even though the stopper terminates.
    write_snapshot = artifacts.RunStore.write_snapshot

    def snapshot(store: artifacts.RunStore, values: dict[str, object]) -> None:
        if fault in ('bestsha', 'missing-best') and 'made' in events and checkpoint.exists():
            summary = values.get('summary')
            assert isinstance(summary, dict)
            if summary['phase'] == 'BEST_RELOAD':
                if fault == 'bestsha':
                    checkpoint.write_bytes(b'??')
                else:
                    checkpoint.unlink()
        write_snapshot(store, values)

    monkeypatch.setattr(artifacts.RunStore, 'write_snapshot', snapshot)
    if fault == 'commit':
        atomic = artifacts.RunStore._atomic

        def fail_commit(store: artifacts.RunStore, path: Path, raw: bytes) -> None:
            if path.name == 'run.json' and b'"test":{' in raw:
                raise OSError('private disk detail')
            atomic(store, path, raw)

        monkeypatch.setattr(artifacts.RunStore, '_atomic', fail_commit)
    if fault == 'reportbudget':
        space = artifacts.RunStore._space
        render = report.report_files

        def tight_report(store: artifacts.RunStore, size: int) -> None:
            if 'report' in events:
                monkeypatch.setattr(store, '_limit', store._total())
            space(store, size)

        def track_report(saved: dict[str, object]) -> dict[str, bytes]:
            events.append('report')
            return render(saved)

        monkeypatch.setattr(artifacts.RunStore, '_space', tight_report)
        monkeypatch.setattr(report, 'report_files', track_report)
    cfg = tmp_path / 'lstm.json'
    cfg.write_text(json.dumps(config.lstm_recipe()), encoding='utf-8')
    argv = ['run', '--plugin', cli.__name__, '--config', str(cfg), '--root', str(root),
            '--source-root', str(root), '--run-id', 'lstm', '--deadline-utc', deadline.isoformat()]
    code = dispatch(argv)
    logs = capsys.readouterr()
    assert 'private ' not in logs.err and 'Traceback' not in logs.err
    assert events.count('made') == 1
    saved = json.loads((root / 'data/lstm/run.json').read_bytes())
    if fault not in ('', 'ceiling', 'reclaim'):
        reason = {'restore': 'BEST_RELOAD_FAILED', 'bestsha': 'HASH_MISMATCH',
                  'missing-best': 'HASH_MISMATCH', 'testsha': 'HASH_MISMATCH',
                  'source': 'SOURCE_CHANGED', 'score': 'TEST_SCORE_FAILED',
                  'stop': 'DEADLINE', 'count': 'FINALIZE_FAILED', 'commit': 'WRITE_FAILED',
                  'reportbudget': 'REPORT_FAILED'}[fault]
        assert code == (3 if fault == 'stop' else 4)
        assert json.loads(logs.err)['reason'] == reason
        assert ('test' in saved['summary']) == (fault == 'reportbudget')
        assert ('read/test' in events) == (fault not in ('source', 'restore', 'bestsha',
                                                        'missing-best'))
        if fault == 'reportbudget':
            assert not list((root / 'result/lstm').iterdir())
        return
    assert code == 0 and saved['status'] == 'SUCCEEDED'
    assert saved['summary']['preflight']['status'] == 'SKIPPED_REUSE'
    assert saved['summary']['mode'] == 'saved-panels'
    assert saved['summary']['topology'] == 'lstm-single'
    arm = saved['summary']['arms']['lstm']
    e = 50 if fault == 'ceiling' else 18
    assert arm['epochs_completed'] == arm['selected']['epoch'] == e
    assert arm['stop_reason'] == ('EPOCH_CEILING' if e == 50 else 'EARLY_STOPPING')
    assert arm['identity'] == f'{config.LSTM_METHOD}/lstm' and arm['best_reloaded'] is True
    assert saved['summary']['test']['accuracy'] == {'lstm': 1., 'majority': .625}
    assert set(saved['artifacts']) == {'model/lstm-best.pt',
                                      *(f'dataset/{s}.npz' for s in config.SPLITS)}
    assert events == ['read/train', 'read/dev', 'made', 'read/best', 'loaded', 'read/test',
                      'predict/test', 'predict/test']
    for kind, n in config.lstm_counts(e).items():
        assert saved['counts'][f'attempted_{kind}'] == saved['counts'][f'completed_{kind}'] == n
    for split, raw in datasets.items():
        assert (root / 'data/lstm/dataset' / f'{split}.npz').read_bytes() == raw
    source_files = saved['source']
    assert any(p.endswith('/cube_compare/lstm.py') for p in source_files)
    if fault == 'reclaim':
        assert dispatch(argv) == 4 and events.count('made') == 1
        assert json.loads(capsys.readouterr().err)['reason'] == 'OCCUPIED'

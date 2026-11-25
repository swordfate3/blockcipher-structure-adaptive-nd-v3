from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.cube_compare import cli, config, data
from cipher_lab_plugins.direct_block.cube_compare.tests.test_cube_direct_data import (
    cube_direct_pins,
)
from cipher_lab_plugins.direct_block.cube_compare.tests.test_cube_direct_data import (
    cube_direct_source as cube_direct_source,
)

from cipher_lab.artifacts import RunStore
from cipher_lab.cli import main as dispatch
from cipher_lab.execution import Clocks, Meter

_LAB = Path(__file__).resolve().parents[5]
_FIXED = _LAB / 'configs/direct-block-lstm-direct64-r1r3.json'


def test_direct64_fixed_contract_counts_and_preclaim_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    assert dispatch(['check', '--plugin', cli.__name__, '--config', str(_FIXED)]) == 0
    expected = config.direct_recipe()
    assert json.loads(capsys.readouterr().out)['recipe'] == expected
    assert expected['rows'] == {'train': 2097152, 'dev': 131072, 'test': 131072}
    assert expected['selection'] == 'full64-BCE/earliest'
    assert expected['secret_entropy_draws'] == 0
    assert config.direct_counts((50, 50, 50)) == {
        'real_blocks': 0, 'null_words': 0, 'optimizer_updates': 1228800, 'forward_rows': 334626816,
    }
    assert config.direct_counts((18, 19, 50)) == {
        'real_blocks': 0, 'null_words': 0, 'optimizer_updates': 8192 * 87,
        'forward_rows': 2228224 * 87 + 393216,
    }
    with pytest.raises(ValueError, match='completed epochs'):
        config.direct_counts((17, 18, 50))
    launcher = (_LAB / 'scripts/start-cube-sharedm.ps1').read_text(encoding='ascii')
    assert launcher.count(f"'{_FIXED.name}'") == 2  # ValidateSet and source-root mode
    assert 'if ($reuse) { 10800 }' in launcher
    assert '& $python @cliArgs 1> $stdout 2> $stderr\n$code = $global:LASTEXITCODE' in launcher
    spec_path = _LAB.parent / 'docs/superpowers/specs/2026-11-24-innovation2-direct64-design.md'
    spec = spec_path.read_text(encoding='utf-8')
    assert f'runSHA=`{config.DIRECT_SOURCE_RUN_SHA256}`' in spec
    assert config.DIRECT_SOURCE_RUN_ID in spec
    pins = config.DIRECT_SOURCE_ARTIFACTS
    assert set(pins) == {f'dataset/r{r}-{s}.npz' for r in config.LOW_ROUNDS for s in config.SPLITS}
    for ref, (size, digest) in pins.items():
        assert f'|{ref}|{size}|{digest}|' in spec
    assert config.SOURCE_RUN_ID != config.DIRECT_SOURCE_RUN_ID
    deadline = datetime.now(UTC) + timedelta(hours=2, minutes=59)
    root = tmp_path / 'original'
    argv = ['run', '--plugin', cli.__name__, '--config', str(_FIXED), '--run-id', 'direct',
            '--deadline-utc', deadline.isoformat()]
    monkeypatch.setattr(RunStore, 'claim',
                        lambda *a, **kw: pytest.fail('preclaim rejection missed'))
    assert dispatch([*argv, '--root', str(tmp_path / 'out')]) == 2
    assert json.loads(capsys.readouterr().err)['reason'] == 'INVALID_ARGUMENTS_OR_CONFIG'
    for kind in ('data', 'models', 'result'):
        old = root / kind / config.DIRECT_SOURCE_RUN_ID
        assert dispatch([*argv, '--root', str(old / 'child'), '--source-root', str(root)]) == 2
        assert json.loads(capsys.readouterr().err)['reason'] == 'SOURCE_OUTPUT_OVERLAP'
    old.mkdir(parents=True)
    alias = tmp_path / 'alias'
    alias.symlink_to(old, target_is_directory=True)
    assert dispatch([*argv, '--root', str(alias), '--source-root', str(root)]) == 2
    assert json.loads(capsys.readouterr().err)['reason'] == 'SOURCE_OUTPUT_OVERLAP'
    assert not cli._source_output_overlap(root, 'direct', root, config.DIRECT_SOURCE_RUN_ID)
    assert cli._source_output_overlap(root, config.DIRECT_SOURCE_RUN_ID, root,
                                      config.DIRECT_SOURCE_RUN_ID)


def test_direct64_first_recipe_write_failure_reports_unknown_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import preflight, resources

    source, out = tmp_path / 'source', tmp_path / 'out'
    source.mkdir()
    cfg = tmp_path / 'recipe.json'
    cfg.write_text(json.dumps(config.direct_recipe()), encoding='utf-8')
    now = datetime(2026, 11, 24, tzinfo=UTC)
    monkeypatch.setattr(cli, 'Clocks', lambda: Clocks(lambda: 0., lambda: 0., lambda: now))
    monkeypatch.setattr(resources, 'working_set_bytes', lambda: 1)
    monkeypatch.setattr(preflight, 'gpu_gate', lambda: pytest.fail('device admission was reached'))
    opened = Path.open

    def no_source_read(path: Path, *a: Any, **kw: Any) -> Any:
        assert not path.is_relative_to(source), 'source access after failed recipe write'
        return opened(path, *a, **kw)

    monkeypatch.setattr(Path, 'open', no_source_read)
    run_path = out / 'data/direct/run.json'
    atomic = RunStore._atomic
    faults = 0

    def fail_first_recipe(store: RunStore, path: Path, raw: bytes) -> None:
        nonlocal faults
        if path == run_path and faults == 0:
            proposed = json.loads(raw)
            if proposed['status'] == 'RUNNING' and proposed['recipe']:
                faults += 1
                raise OSError('private artificial first recipe write error')
        atomic(store, path, raw)

    monkeypatch.setattr(RunStore, '_atomic', fail_first_recipe)
    assert dispatch(['run', '--plugin', cli.__name__, '--config', str(cfg), '--root', str(out),
                     '--source-root', str(source), '--run-id', 'direct', '--deadline-utc',
                     (now + timedelta(hours=3)).isoformat()]) == 4
    log = capsys.readouterr()
    assert faults == 1 and not log.out and 'private' not in log.err
    error = json.loads(log.err)
    assert (error['status'], error['reason']) == ('FAILED', 'WRITE_FAILED')
    for unit in ('real_blocks', 'null_words', 'optimizer_updates', 'forward_rows'):
        assert error['counts'][f'attempted_{unit}'] == error['counts'][f'completed_{unit}'] == 0
    raw = run_path.read_bytes()
    saved = json.loads(raw)
    assert saved['status'] == 'FAILED' and saved['recipe'] == saved['artifacts'] == {}
    assert saved['summary'] == {'reason': 'WRITE_FAILED'}
    result = out / 'result/direct'
    assert json.loads((result / 'source.json').read_bytes()) == {
        'schema': 1, 'run_id': 'direct', 'run_ref': 'data/direct/run.json',
        'run_sha256': sha256(raw).hexdigest(),
    }
    for name in ('index.html', 'report.md'):
        text = (result / name).read_text(encoding='utf-8').lower()
        for claim in ('simon7', 'cube-pair', 'paired scoring', 'active-8', 'active8',
                      'full 128-bit', 'full128', 'full64', 'single-word lstm'):
            assert claim not in text
        assert 'method unknown' in text and 'failed' in text


@pytest.mark.parametrize('fault', ['', 'sha1', 'sha2', 'sha3', 'restore1', 'restore2', 'restore3',
                                  'testsha', 'score3', 'testdeadline', 'output', 'commit',
                                  'reportio', 'reportdeadline', 'reportoutput', 'runtime',
                                  'provider', 'sourcecode'])
def test_direct64_cli_matrix_fault_gate_and_atomic_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    cube_direct_source: Callable[..., tuple[Path, str, dict[str, bytes]]], fault: str,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import direct64, engine, preflight, resources

    root, digest, datasets = cube_direct_source()
    pins = cube_direct_pins(datasets)
    monkeypatch.setattr(config, 'FAMILIES', (1, 1, 1))
    monkeypatch.setattr(config, 'BATCH', 16)
    def manual_provider(source: Path, meter: Meter) -> direct64.DirectPanels:
        return direct64.DirectPanels(source, meter, run_id=config.DIRECT_SOURCE_RUN_ID,
                                     run_sha256=digest, artifacts=pins, families=(1, 1, 1))

    monkeypatch.setattr(direct64, 'production_direct_panels', manual_provider)
    source_dir = root / 'data' / config.DIRECT_SOURCE_RUN_ID
    if fault == 'testsha':
        (source_dir / 'dataset/r2-test.npz').write_bytes(b'!' * pins['dataset/r2-test.npz'][0])
    if fault == 'provider':
        (source_dir / 'run.json').write_bytes(b'wrong source')
    out = tmp_path / 'out'
    now = datetime(2026, 11, 24, tzinfo=UTC)
    deadline = now + timedelta(hours=3)
    events: list[str] = []
    orders: dict[int, list[bytes]] = {r: [] for r in config.LOW_ROUNDS}
    opened = Path.open

    def observe(path: Path, *a: Any, **kw: Any) -> Any:
        nonlocal now
        if path.is_relative_to(root):
            ref = f'dataset/{path.name}'
            assert path.name == 'run.json' or ref in datasets
            if ref in datasets:
                events.append(path.stem)
                if path.stem.endswith('-test'):
                    assert {event for event in events if event.startswith('loaded/')} == {
                        'loaded/1', 'loaded/2', 'loaded/3',
                    }
                    if fault == 'testdeadline':
                        now = deadline
        return opened(path, *a, **kw)

    monkeypatch.setattr(Path, 'open', observe)
    monkeypatch.setattr(cli, 'Clocks', lambda: Clocks(lambda: 0., lambda: 0., lambda: now))
    monkeypatch.setattr(resources, 'working_set_bytes', lambda: 1)
    monkeypatch.setattr(preflight, 'gpu_gate', lambda: {'uuid': config.GPU_UUID})
    monkeypatch.setattr(preflight, 'gpu_peaks', lambda: {'manual': True})

    def runtime() -> dict[str, object]:
        if fault == 'runtime':
            raise ValueError('private artificial runtime error')
        return {'manual': True}

    monkeypatch.setattr(engine, 'configure_runtime', runtime)
    for owner, attr in ((preflight, 'run'), (engine, 'make_engine'),
                        (engine, 'make_lstm_engine'), (engine, 'make_low_engine')):
        monkeypatch.setattr(owner, attr, lambda *a, **k: pytest.fail('legacy science path'))

    class ManualEngine:
        def __init__(self, rounds: int) -> None:
            assert len(events) == 6 + rounds - 1
            events.append(f'made/{rounds}')
            self.r, self.step, self.loaded = rounds, 0, False

        def train_forward(self, x: data.Floats) -> Callable[[data.Labels], float]:
            assert x.shape == (16, 64) and x.dtype == np.float32
            orders[self.r].append(x.tobytes())

            def labels(y: data.Labels) -> float:
                np.testing.assert_array_equal(y, self.target(x))
                return .125

            return labels

        def target(self, x: data.Floats) -> data.Labels:
            y = np.zeros_like(x, dtype=np.uint8)
            y[:, 0] = x[:, 3]  # right endpoints have original active LSB0 bit 60 set
            y[:, 62] = self.r >> 1
            y[:, 63] = self.r & 1
            return y

        def predict(self, x: data.Floats) -> data.Floats:
            z = 2 * self.target(x).astype(np.float32) - 1
            if self.loaded:
                events.append(f'score/{self.r}')
                if fault == 'score3' and self.r == 3:
                    raise ValueError('private score')
            else:
                # Never change active columns 63, 42, 21, 3; selection must see col 10.
                scale = {1: 1., 2: (1. if self.step == 1 else 2.), 3: 1. + self.step / 10}[self.r]
                z[:, 10] *= np.float32(scale)
            return z

        def update(self) -> None:
            self.step += 1

        def dump(self) -> bytes:
            return f'{self.r}/{self.step}'.encode()

        def restore(self, raw: bytes) -> None:
            assert raw == f'{self.r}/{(1, 2, 50)[self.r - 1]}'.encode()
            if fault == f'restore{self.r}':
                raise ValueError('private restore')
            self.loaded = True
            events.append(f'loaded/{self.r}')

    monkeypatch.setattr(engine, 'make_direct_engine', ManualEngine, raising=False)
    read_model = RunStore.read_model

    def watch_best(store: RunStore, ref: str, sha: str, size: int) -> bytes:
        n = sum(event.startswith('loaded/') for event in events) + 1
        assert ref == f'model/r{n}-best.pt' and size == 16 * 2**20
        if fault == f'sha{n}':
            (out / 'models/direct' / f'r{n}-best.pt').write_bytes(b'corrupted')
        return read_model(store, ref, sha, size)

    monkeypatch.setattr(RunStore, 'read_model', watch_best)
    atomic = RunStore._atomic

    def boundary(store: RunStore, path: Path, raw: bytes) -> None:
        nonlocal now
        if fault == 'output' and path.name == 'r3-test.npz':
            monkeypatch.setattr(store, '_limit', store._total())
        if fault == 'commit' and path.name == 'run.json' and b'"test":{' in raw:
            raise OSError('private write')
        if path.parent == out / 'result/direct':
            if fault == 'reportio':
                raise OSError('private report')
            if fault == 'reportdeadline':
                now = deadline
        atomic(store, path, raw)

    monkeypatch.setattr(RunStore, '_atomic', boundary)
    if fault == 'reportoutput':
        write_reports = RunStore.write_report_files

        def over_report_budget(store: RunStore, files: dict[str, bytes]) -> None:
            monkeypatch.setattr(store, '_limit', store._total())
            write_reports(store, files)

        monkeypatch.setattr(RunStore, 'write_report_files', over_report_budget)
    if fault == 'sourcecode':
        from cipher_lab_plugins.direct_block.source import source_identity

        def identity() -> dict[str, str]:
            value = source_identity()
            if 'made/1' in events:
                value['artificial/changed.py'] = '0' * 64
            return value

        monkeypatch.setattr(cli, 'source_identity', identity)
    cfg = tmp_path / 'recipe.json'
    cfg.write_text(json.dumps(config.direct_recipe()), encoding='utf-8')
    argv = ['run', '--plugin', cli.__name__, '--config', str(cfg), '--root', str(out),
            '--source-root', str(root), '--run-id', 'direct',
            '--deadline-utc', deadline.isoformat()]
    code = dispatch(argv)
    log = capsys.readouterr()
    assert 'private' not in log.err and 'Traceback' not in log.err
    saved = json.loads((out / 'data/direct/run.json').read_bytes())
    counts, summary = saved['counts'], saved['summary']
    for unit in ('real_blocks', 'null_words'):
        assert counts[f'attempted_{unit}'] == counts[f'completed_{unit}'] == 0
    assert saved['recipe']['topology'] == 'direct64-r1r3'
    assert {'cipher_lab_plugins/direct_block/cube_compare/' + f for f in
            ('direct64.py', 'lstm64.py', 'direct_metrics.py')} <= saved['source'].keys()
    complete = not fault or fault.startswith('report')
    assert ('test' in summary) == complete
    assert ('r1-test' in events) == (fault not in ('runtime', 'provider', 'sourcecode')
                                      and not fault.startswith(('sha', 'restore')))
    if fault:
        reason = {'testsha': 'HASH_MISMATCH', 'score3': 'TEST_SCORE_FAILED',
                  'testdeadline': 'DEADLINE', 'output': 'OUTPUT_LIMIT', 'commit': 'WRITE_FAILED',
                  'reportio': 'REPORT_FAILED', 'reportdeadline': 'DEADLINE',
                  'reportoutput': 'REPORT_FAILED',
                  'runtime': 'RUNTIME_FAILED', 'provider': 'HASH_MISMATCH',
                  'sourcecode': 'SOURCE_CHANGED'}.get(fault)
        if reason is None:
            reason = 'HASH_MISMATCH' if fault.startswith('sha') else 'BEST_RELOAD_FAILED'
        assert code == (3 if fault.endswith('deadline') else 4)
        assert not log.out and json.loads(log.err)['reason'] == reason
        exit_counts = json.loads(log.err)['counts']
        assert exit_counts['completed_optimizer_updates'] == (0 if fault in {
            'runtime', 'provider',
        } else 1 if fault == 'sourcecode' else 86)
        if fault == 'score3':
            assert counts['attempted_forward_rows'] - counts['completed_forward_rows'] == 16
    else:
        assert code == 0
    if complete:
        assert saved['status'] == 'SUCCEEDED'
        assert set(summary['test']['rounds']) == {'r1', 'r2', 'r3'}
        epochs = (18, 18, 50)
        assert orders[1] == orders[2] == orders[3][:18]
        for unit, expected in config.direct_counts(epochs).items():
            assert counts[f'attempted_{unit}'] == counts[f'completed_{unit}'] == expected
        for r, e, best in zip(config.LOW_ROUNDS, epochs, (1, 2, 50), strict=True):
            arm = summary['arms'][f'r{r}']
            assert arm['epochs_completed'] == e and arm['selected']['epoch'] == best
            assert arm['stop_reason'] == ('EPOCH_CEILING' if r == 3 else 'EARLY_STOPPING')
            assert arm['identity'] == f'{config.DIRECT_METHOD}/r{r}' and arm['best_reloaded']
            assert all('active_bce' not in point for point in arm['dev'])
            result = summary['test']['rounds'][f'r{r}']
            assert result['scores']['lstm']['correct']['full64'] == 1024
            for split in config.SPLITS:
                ref = f'dataset/r{r}-{split}.npz'
                assert (out / 'data/direct' / ref).read_bytes() == datasets[ref]
    for name in ('index.html', 'report.md'):
        path = out / 'result/direct' / name
        if path.exists():
            html = path.read_text(encoding='utf-8')
            assert 'Planned method' in html and 'active-8' not in html and 'active_bce' not in html
            assert 'uniform' in html and 'absolute' in html and 'family results' not in html
            if not complete:
                assert 'No final TEST score was committed.' in html
    if not fault:
        assert dispatch(argv) == 4
        assert json.loads(capsys.readouterr().err)['reason'] == 'OCCUPIED'

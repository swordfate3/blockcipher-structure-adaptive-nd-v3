from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from cipher_lab_plugins.direct_block.cube_compare.config import GPU_UUID, recipe


def test_check_fixed_recipe_does_not_claim(tmp_path: Path) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import cli

    path = tmp_path / 'config.json'
    path.write_text(json.dumps(recipe()), encoding='utf-8')
    assert cli.main(['check', '--config', str(path)]) == 0
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('edit', [dict(rounds=6), dict(key_protocol='rotating'),
                                  dict(target_width=64), dict(retries=False),
                                  dict(gpu_uuid='GPU-wrong')])
def test_check_rejects_recipe_drift(tmp_path: Path, edit: dict[str, object]) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import cli

    path = tmp_path / 'config.json'
    path.write_text(json.dumps({**recipe(), **edit}), encoding='utf-8')
    assert cli.main(['check', '--config', str(path)]) == 2


def test_launcher_contract() -> None:
    script = Path('scripts/start-cube-sharedm.ps1').read_text(encoding='ascii')
    assert GPU_UUID in script
    assert 'Remove-Item Env:PYTHONNOUSERSITE' in script
    assert ' -I ' not in script and ' -S ' not in script
    assert '& $python @cliArgs' in script
    assert '$code = $LASTEXITCODE' in script and 'exit $code' in script
    assert 'AddSeconds(7200)' in script and 'G:\\lxy\\' in script


def test_launcher_exit_evidence_failure_contract() -> None:
    # Static PowerShell contract: no Windows interpreter or native process here.
    script = Path('scripts/start-cube-sharedm.ps1').read_text(encoding='ascii')
    closing = script.split('$code = $LASTEXITCODE\n', 1)[1]
    assert "if ($null -eq $code) { throw 'NATIVE_EXIT_UNKNOWN' }" in closing
    assert "try {\n    Set-Content -LiteralPath (Join-Path $work 'exit.txt')" in closing
    assert "-ErrorAction Stop\n} catch {\n    Write-Warning 'EXIT_EVIDENCE_WRITE_FAILED'" in closing
    assert "'EXIT_EVIDENCE_WRITE_FAILED' -WarningAction Continue\n" in closing
    assert '    if ($code -eq 0) { exit 1 }\n}\nexit $code\n' in closing


@pytest.mark.parametrize('fault', [
    '', 'occupied', 'source', 'expired', 'cpu', 'rss', 'device', 'kat', 'preflight-reload',
    'preflight-reload-probe-error', 'preflight-expired-failure', 'preflight-stop',
    'preflight-read', 'preflight-read-bad-media', 'preflight-write', 'preflight-write-bad-media',
    'preflight-probe', 'reload', 'score', 'output', 'commit', 'report', 'archived-stop',
    'read-deadline', 'render-deadline', 'failed-read-deadline',
])
def test_cli_manual_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                              capsys: pytest.CaptureFixture[str], fault: str) -> None:
    from datetime import UTC, datetime, timedelta

    import numpy as np
    from cipher_lab_plugins.direct_block.cube_compare import (
        cli,
        config,
        data,
        engine,
        preflight,
        report,
        resources,
    )
    from cipher_lab_plugins.direct_block.cube_compare.tests.test_training import (
        ManualEngine,
        ManualPanels,
    )
    from cipher_lab_plugins.direct_block.source import source_identity

    from cipher_lab import artifacts
    from cipher_lab.ciphers import simon
    from cipher_lab.cli import main as dispatch
    from cipher_lab.execution import Clocks, Meter, Stopped

    # Test-only plan injected in-process; the checked-in JSON remains full-sized.
    monkeypatch.setattr(config, 'FAMILIES', (2, 1, 2))
    monkeypatch.setattr(config, 'EPOCHS', 2)
    monkeypatch.setattr(config, 'BATCH', 8)
    assert config.total_caps() == dict(real_blocks=98, null_words=0,
                                       optimizer_updates=10, forward_rows=160)
    events: list[str] = []
    calls: list[tuple[str, int, int]] = []
    made: list[str] = []
    configurations: list[str] = []
    cpu = 0.
    now = datetime(2026, 11, 23, tzinfo=UTC)
    deadline = now + timedelta(seconds=10 if fault != 'expired' else 0)
    monkeypatch.setattr(cli, 'Clocks', lambda: Clocks(lambda: 0., lambda: cpu, lambda: now))

    def rss() -> int:
        if fault == 'rss':
            raise Stopped('RSS_UNAVAILABLE')
        return 53

    monkeypatch.setattr(resources, 'working_set_bytes', rss)
    original_source = source_identity
    scans = 0

    def source() -> dict[str, str]:
        nonlocal scans
        scans += 1
        identity = original_source()
        if fault == 'source' and scans > 1:
            identity[next(iter(identity))] = 'f' * 64
        return identity

    monkeypatch.setattr(cli, 'source_identity', source)

    def gate() -> dict[str, object]:
        if fault == 'device':
            raise preflight.DeviceError('GPU_IDENTITY')
        return {'uuid': config.GPU_UUID, 'logical_device': 'cuda:0'}

    monkeypatch.setattr(preflight, 'gpu_gate', gate)
    peak_calls: list[tuple[str, ...]] = []

    def peaks() -> dict[str, int]:
        peak_calls.append(tuple(made))
        if fault in ('preflight-probe', 'preflight-reload-probe-error'):
            raise ValueError('private sentinel')
        return {'peak_allocated_bytes': 1024, 'peak_reserved_bytes': 4096}

    monkeypatch.setattr(preflight, 'gpu_peaks', peaks)

    def configure() -> dict[str, object]:
        configurations.append('configure')
        return {'manual': True}

    monkeypatch.setattr(engine, 'configure_runtime', configure)

    def scalar(points: tuple[int, ...], key: tuple[int, ...], rounds: int) -> tuple[int, ...]:
        nonlocal cpu
        calls.append(('scalar', len(points), rounds))
        assert key == preflight._PUBLIC_KEY
        if fault == 'cpu':
            cpu = 28800.
        return ((0 if fault == 'kat' else 0x44C8FC20B9DFA07A),) if rounds == 44 else points

    def batch(points: data.Words, key: tuple[int, ...], rounds: int) -> data.Words:
        assert key == preflight._PUBLIC_KEY
        calls.append(('batch', len(points), rounds))
        return np.array([0x44C8FC20B9DFA07A], dtype=np.uint64) if rounds == 44 else points.copy()

    monkeypatch.setattr(simon, 'encrypt_many', scalar)
    monkeypatch.setattr(simon, 'encrypt_batch', batch)

    class PreflightEngine(ManualEngine):
        gate_loaded = False

        def restore(self, raw: bytes) -> None:
            nonlocal now
            assert raw == f'{self.arm}/1'.encode()
            self.events.append(f'gate-reload/{self.arm}')
            if self.arm == 'sharedm':
                if fault in ('preflight-stop', 'preflight-expired-failure'):
                    now = deadline
                if fault in ('preflight-reload', 'preflight-reload-probe-error',
                             'preflight-expired-failure'):
                    raise ValueError('private sentinel')
            self.gate_loaded = True

        def predict(self, inputs: data.Floats) -> data.Floats:
            assert self.gate_loaded and inputs.shape == (8, 128)
            self.events.append(f'gate-predict/{self.arm}')
            return super().predict(inputs)

    def make_engine(arm: str) -> ManualEngine:
        if len(made) < 2:
            result: ManualEngine = PreflightEngine(arm, events)
        else:
            arm_fault = 'score' if fault == 'failed-read-deadline' else fault
            result = ManualEngine(arm, events, arm_fault if arm == 'sharedm' else '')
        made.append(arm)
        return result

    monkeypatch.setattr(engine, 'make_engine', make_engine)
    panels: list[ManualPanels] = []

    def make_panels(meter: Meter) -> ManualPanels:
        assert events == ['update/mlp', 'gate-reload/mlp', 'gate-predict/mlp',
                          'update/sharedm', 'gate-reload/sharedm', 'gate-predict/sharedm']
        panel = ManualPanels(meter, events)
        panels.append(panel)
        return panel

    monkeypatch.setattr(data, 'production_panels', make_panels)
    root = tmp_path / 'run space'
    cfg = tmp_path / 'config space.json'
    cfg.write_text(json.dumps(recipe()), encoding='utf-8')
    if fault == 'occupied':
        (root / 'models/manual').mkdir(parents=True)
    atomic = artifacts.RunStore._atomic
    sealed_writes: list[artifacts.RunStore] = []
    write_faults: list[Path] = []
    checkpoint = root / 'models/manual/sharedm-preflight.pt'
    disk_read_fault = fault in ('preflight-read', 'preflight-read-bad-media')
    disk_write_fault = fault in ('preflight-write', 'preflight-write-bad-media')

    def inject(store: artifacts.RunStore, target: Path, content: bytes) -> None:
        if disk_write_fault and target == checkpoint:
            write_faults.append(target)
            raise OSError('private sentinel')
        if (disk_read_fault or disk_write_fault) and target.name == 'run.json':
            if b'"status":"FAILED"' in content:
                sealed_writes.append(store)
                if fault.endswith('-bad-media'):
                    raise OSError('private sentinel')
        if fault == 'commit' and target.name == 'run.json' and b'"test":{' in content:
            raise OSError('private sentinel')
        if fault == 'report' and target.name == 'index.html':
            raise OSError('private sentinel')
        atomic(store, target, content)

    monkeypatch.setattr(artifacts.RunStore, '_atomic', inject)
    read_faults: list[Path] = []
    if disk_read_fault:
        opened = Path.open

        def fail_read(path: Path, *args: Any, **kwargs: Any) -> Any:
            # Fail only the first disk read of the newly written file, through
            # real RunStore.read_model/read_binary; later saved verification works.
            if path == checkpoint and args == ('rb',) and not read_faults:
                read_faults.append(path)
                raise OSError('private sentinel')
            return opened(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'open', fail_read)
    report_stages: list[str] = []
    render = report.report_files
    write_report = artifacts.RunStore.write_report_files
    write_dataset = artifacts.RunStore.write_dataset

    def read_report(root: Path, run_id: str) -> dict[str, object]:
        nonlocal now
        report_stages.append('read')
        saved = artifacts.read_saved(root, run_id)
        if fault in ('read-deadline', 'failed-read-deadline'):
            now = deadline
        return saved

    def render_report(saved: dict[str, object]) -> dict[str, bytes]:
        nonlocal now
        report_stages.append('render')
        files = render(saved)
        if fault == 'render-deadline':
            now = deadline
        return files

    def persist_report(store: artifacts.RunStore, files: dict[str, bytes]) -> None:
        report_stages.append('write')
        write_report(store, files)

    def archive(store: artifacts.RunStore, name: str, content: bytes) -> str:
        nonlocal now
        ref = write_dataset(store, name, content)
        if fault == 'archived-stop' and name == 'test':
            now = deadline
        return ref

    monkeypatch.setattr(cli, 'read_saved', read_report)
    monkeypatch.setattr(report, 'report_files', render_report)
    monkeypatch.setattr(artifacts.RunStore, 'write_report_files', persist_report)
    monkeypatch.setattr(artifacts.RunStore, 'write_dataset', archive)
    if fault == 'output':
        # Real core _space rejects the next simultaneous copy, not a fake store.
        space = artifacts.RunStore._space

        def tight_limit(store: artifacts.RunStore, additional: int) -> None:
            if made == ['mlp', 'sharedm', 'mlp', 'sharedm'] and 'test/sharedm' in events:
                monkeypatch.setattr(store, '_limit', store._total())
            space(store, additional)

        monkeypatch.setattr(artifacts.RunStore, '_space', tight_limit)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', config.GPU_UUID)
    code = dispatch(['run', '--plugin', cli.__name__, '--config', str(cfg), '--root', str(root),
                     '--run-id', 'manual', '--deadline-utc', deadline.isoformat()])
    logs = capsys.readouterr()
    assert 'private sentinel' not in logs.out + logs.err
    assert 'Traceback' not in logs.err
    if fault == 'occupied':
        assert code == 4 and 'OCCUPIED' in logs.err
        assert not (root / 'data').exists() and not made and not calls
        return
    saved = artifacts.read_saved(root, 'manual')
    summary = saved['summary']
    counts = saved['counts']
    refs = saved['artifacts']
    assert isinstance(summary, dict) and isinstance(counts, dict) and isinstance(refs, dict)
    if fault in ('expired', 'cpu', 'rss', 'source', 'device', 'kat', 'preflight-reload'):
        assert not panels and 'test' not in summary
    if fault in ('expired', 'rss', 'source', 'device'):
        assert not calls and not configurations
    if fault in ('expired', 'rss'):
        assert code == 3 and counts['rss_bytes'] is None
    if fault == 'cpu':
        assert code == 3 and counts['completed_real_blocks'] == 1
        assert counts['cpu_elapsed_seconds'] == 28800.
    if fault == 'kat':
        assert code == 4 and counts['attempted_real_blocks'] == 1
        assert counts['completed_real_blocks'] == 0 and not made
        assert not peak_calls and 'cuda_peaks_after_preflight' not in summary
    if disk_read_fault or disk_write_fault:
        reason = 'READ_FAILED' if disk_read_fault else 'WRITE_FAILED'
        assert code == 4 and json.loads(logs.err)['reason'] == reason
        assert not panels and 'label/test' not in events and 'test' not in summary
        assert read_faults == ([checkpoint] if disk_read_fault else [])
        assert write_faults == ([checkpoint] if disk_write_fault else [])
        assert len(sealed_writes) == 1 and sealed_writes[0]._sealed
        assert sealed_writes[0]._failure_evidence_key is None
        assert made == ['mlp', 'sharedm'] and 'gate-reload/sharedm' not in events
        assert set(refs) == ({'model/mlp-preflight.pt', 'model/sharedm-preflight.pt'}
                             if disk_read_fault else {'model/mlp-preflight.pt'})
        assert counts['completed_forward_rows'] == 24
        if fault.endswith('-bad-media'):
            assert saved['status'] == 'RUNNING' and 'reason' not in summary  # UNKNOWN
            assert summary['cuda_peaks_after_preflight'] is None
            assert not list((root / 'result/manual').iterdir())
        else:
            assert saved['status'] == 'FAILED'
            assert summary == {'reason': reason, 'cuda_peaks_after_preflight': None}
        return
    if fault.startswith('preflight-'):
        assert not panels and 'test' not in summary
        assert counts['completed_optimizer_updates'] == 2
        assert counts['completed_forward_rows'] == (32 if fault == 'preflight-probe' else 24)
        assert code == (3 if fault == 'preflight-stop' else 4)
        expected_peaks = {'peak_allocated_bytes': 1024, 'peak_reserved_bytes': 4096}
        assert summary['cuda_peaks_after_preflight'] == (
            expected_peaks if fault == 'preflight-reload' else None)
        assert peak_calls == ([] if fault in ('preflight-stop', 'preflight-expired-failure')
                              else [('mlp', 'sharedm')])
        expected_reason = 'DEADLINE' if fault == 'preflight-stop' else 'PREFLIGHT_FAILED'
        assert json.loads(logs.err)['reason'] == expected_reason
        if fault in ('preflight-stop', 'preflight-expired-failure'):
            assert not report_stages
    if fault == 'archived-stop':
        assert code == 3 and saved['status'] == 'STOPPED'
        assert json.loads(logs.err)['reason'] == summary['reason'] == 'DEADLINE'
        assert {'dataset/train.npz', 'dataset/dev.npz', 'dataset/test.npz'} <= refs.keys()
        assert 'test' not in summary and not report_stages
        assert peak_calls == [('mlp', 'sharedm')]
    if fault in ('read-deadline', 'render-deadline', 'failed-read-deadline'):
        assert report_stages == (['read', 'render'] if fault == 'render-deadline' else ['read'])
        assert not list((root / 'result/manual').iterdir())
        if fault == 'failed-read-deadline':
            assert code == 4 and saved['status'] == 'FAILED'
            assert json.loads(logs.err)['reason'] == summary['reason'] == 'TEST_SCORE_FAILED'
        else:
            assert code == 3 and saved['status'] == 'SUCCEEDED'
            assert json.loads(logs.err)['reason'] == 'DEADLINE'
    if fault in ('reload', 'score'):
        assert code == 4 and saved['status'] == 'FAILED'
        assert 'test' not in summary and 'dataset/test.npz' not in refs
        assert ('label/test' in events) == (fault == 'score')
    if fault == 'commit':
        assert code == 4 and saved['status'] == 'FAILED' and 'test' not in summary
        assert counts['completed_forward_rows'] == 160
    if fault == 'output':
        assert code == 4 and 'OUTPUT_LIMIT' in logs.err and 'test' not in summary
        assert not report_stages and peak_calls == [('mlp', 'sharedm')]
    if fault in ('', 'report'):
        assert code == (4 if fault else 0) and saved['status'] == 'SUCCEEDED'
        assert 'test' in summary and configurations == ['configure'] and len(panels) == 1
        assert made == ['mlp', 'sharedm'] * 2
        assert calls == [('scalar', 1, 44), ('batch', 1, 44), ('scalar', 8, 7), ('batch', 8, 7)]
        for key, value in config.total_caps().items():
            assert counts[f'attempted_{key}'] == counts[f'completed_{key}'] == value
        assert set(refs) == {
            'model/mlp-preflight.pt', 'model/sharedm-preflight.pt', 'model/mlp-best.pt',
            'model/sharedm-best.pt', 'dataset/train.npz', 'dataset/dev.npz', 'dataset/test.npz',
        }
        if not fault:
            result = root / 'result/manual'
            assert {x.name for x in result.iterdir()} == {
                'index.html', 'report.md', 'source.json',
            }
            provenance = json.loads((result / 'source.json').read_text(encoding='utf-8'))
            assert provenance['run_sha256'] == saved['run_sha256']
            html = (result / 'index.html').read_text(encoding='utf-8')
            assert 'Saved active-bit accuracy' in html


def test_production_recipe_and_policy_are_fixed() -> None:
    from datetime import UTC, datetime, timedelta

    from cipher_lab_plugins.direct_block.cube_compare import cli, config

    from cipher_lab.execution import Clocks

    actual = cli._fixed(Path('configs/direct-block-cube-sharedm-simon7.json'))
    assert actual['rounds'] == 7 and actual['key_protocol'] == 'single-key'
    assert actual['input_width'] == actual['target_width'] == 128
    assert actual['pairs'] == {'train': 1048576, 'dev': 65536, 'test': 65536}
    assert actual['epochs'] == 5 and actual['batch_size'] == 256
    assert config.planned_counts() == dict(real_blocks=2359296, null_words=0,
                                           optimizer_updates=40960, forward_rows=11272192)
    assert config.total_caps() == dict(real_blocks=2359314, null_words=0,
                                       optimizer_updates=40962, forward_rows=11272224)
    now = datetime(2026, 11, 23, tzinfo=UTC)
    clocks = Clocks(lambda: 0., lambda: 0., lambda: now)
    policy = cli._policy((now + timedelta(hours=2)).isoformat(), clocks)
    assert policy.wall_seconds == 7200 and policy.cpu_seconds == 28800
    assert policy.rss_stop_bytes == 8 * 2**30 and policy.output_bytes == 512 * 2**20
    with pytest.raises(ValueError, match='INVALID_DEADLINE'):
        cli._policy((now + timedelta(seconds=7201)).isoformat(), clocks)

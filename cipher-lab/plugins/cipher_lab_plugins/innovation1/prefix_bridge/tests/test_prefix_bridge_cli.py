from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from cipher_lab_plugins.direct_block.cube_compare import resources
from cipher_lab_plugins.innovation1.prefix_bridge import cli, training
from cipher_lab_plugins.innovation1.prefix_bridge import source as bridge_source

from cipher_lab.artifacts import read_saved
from cipher_lab.execution import Clocks, Stopped
from cipher_lab.plugin_cli import dispatch

_PLUGIN = 'cipher_lab_plugins.innovation1.prefix_bridge.cli'
_LAB = Path(__file__).resolve().parents[5]


def test_check_exact_config_no_claim_or_source(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        pytest.fail('check attempted a run side effect')

    monkeypatch.setattr(cli, '_run', blocked)
    config = _LAB / 'configs' / 'innovation1-present-prefix2-bridge.json'
    args = ['check', '--plugin', _PLUGIN, '--config', str(config)]
    assert dispatch(args) == 0
    saved = json.loads(capsys.readouterr().out)
    assert saved['recipe']['count_caps']['real_blocks'] == 3538958
    invalid = tmp_path / 'invalid.json'
    invalid.write_text(config.read_text().replace('"amp": false', '"amp": 0'))
    assert dispatch([*args[:-1], str(invalid)]) == 2
    assert not (tmp_path / 'data').exists()


def test_launcher_fixed_dispatch_and_untouched_exit_capture() -> None:
    script = (_LAB / 'scripts' / 'start-cube-sharedm.ps1').read_text()
    assert "'innovation1-present-prefix2-bridge.json'" in script
    assert "if ($prefixBridge) { $wallSeconds = 14400 }" in script
    assert f"if ($prefixBridge) {{ $plugin = '{_PLUGIN}' }}" in script
    assert "if ($reuse) { $cliArgs += @('--source-root', $SourceRoot) }" in script
    assert '$code = $global:LASTEXITCODE' in script


@pytest.mark.parametrize('fault', ['deadline', 'private', 'report'])
def test_cli_run_sanitizes_failure_and_does_not_replace_terminal(
    fault: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    from cipher_lab_plugins.innovation1.prefix_bridge import report, runtime

    monkeypatch.setattr(bridge_source, 'source_identity', lambda: {'fake/a.py': 'a' * 64})
    monkeypatch.setattr(resources, 'working_set_bytes', lambda: 1)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(cli, 'Clocks', lambda: Clocks(lambda: 0., lambda: 0., lambda: now))
    monkeypatch.setattr(cli, '_device', lambda: {'artificial': True})
    monkeypatch.setattr(runtime, 'configure', lambda: {'artificial': True})
    entered: list[str] = []

    def run(meter: Any, store: Any, summary: dict[str, object], **unused: object) -> None:
        entered.append('run')
        summary['phase'] = 'TRAIN_DEV_DATA'
        if fault == 'deadline':
            raise Stopped('DEADLINE')
        if fault == 'private':
            raise ValueError('private-sentinel-value')
        store.finalize({'status': 'SUCCEEDED', 'summary': {'phase': 'COMPLETE', 'test': {}}})

    def broken_report(saved: object) -> dict[str, bytes]:
        raise OSError('private-sentinel-value')

    monkeypatch.setattr(training, 'run', run)
    monkeypatch.setattr(report, 'report_files', broken_report)
    args = ['run', '--plugin', _PLUGIN, '--config',
            str(_LAB / 'configs' / 'innovation1-present-prefix2-bridge.json'),
            '--root', str(tmp_path), '--run-id', 'cli-fake',
            '--deadline-utc', '2026-01-01T04:00:00Z']
    assert dispatch([*args, '--source-root', 'nope']) == 2
    assert not entered and not (tmp_path / 'data').exists()
    capsys.readouterr()
    assert dispatch(args) == (3 if fault == 'deadline' else 4)
    saved = read_saved(tmp_path, 'cli-fake')
    expected = {'deadline': ('STOPPED', 'DEADLINE'), 'private': ('FAILED', 'TRAIN_DEV_DATA_FAILED'),
                'report': ('SUCCEEDED', 'REPORT_FAILED')}[fault]
    assert saved['status'] == expected[0] and entered == ['run']
    err = capsys.readouterr().err
    assert 'private-sentinel-value' not in err and 'private-sentinel-value' not in str(saved)
    assert json.loads(err)['reason'] == expected[1]

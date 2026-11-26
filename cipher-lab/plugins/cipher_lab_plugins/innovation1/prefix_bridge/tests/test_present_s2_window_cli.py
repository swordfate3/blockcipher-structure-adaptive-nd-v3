from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from cipher_lab.artifacts import read_saved
from cipher_lab.execution import Clocks, Stopped
from cipher_lab.plugin_cli import dispatch

_PLUGIN = 'cipher_lab_plugins.innovation1.s2_window.cli'
_LAB = Path(__file__).resolve().parents[5]
_CONFIG = _LAB / 'configs' / 'innovation1-present-s2-window.json'


def test_exact_new_recipe_separate_from_old_check(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.innovation1.prefix_bridge import cli as old_cli
    from cipher_lab_plugins.innovation1.s2_window import cli, config

    def forbidden(*args: object, **kwargs: object) -> Any:
        pytest.fail('check crossed a run boundary')

    monkeypatch.setattr(cli, '_run', forbidden)
    monkeypatch.setattr(cli, 'Clocks', forbidden)
    args = ['check', '--plugin', _PLUGIN, '--config']
    assert dispatch([*args, str(_CONFIG)]) == 0
    recipe = json.loads(capsys.readouterr().out)['recipe']
    assert recipe['rounds'] == [2, 7] and recipe['fits'] == 1 and recipe['fresh'] is True
    assert recipe['difference'] == 0xD00000 and recipe['difference_at_round'] == 2
    assert [recipe[f'{split}_rows'] for split in ('train', 'dev', 'test')] == [32768, 8192, 32768]
    assert recipe['gate'] == config.KAT_NAME and recipe['kat_real_blocks'] == 5
    assert recipe['count_caps'] == dict(real_blocks=1769477, null_words=0, optimizer_updates=7680,
                                       forward_rows=1261568)
    assert recipe['wall_seconds'] == 3600 and recipe['cpu_seconds'] == 14400
    assert config.counts(18) == dict(real_blocks=1769477, null_words=0, optimizer_updates=4608,
                                     forward_rows=770048)
    edits = ({'fresh': 1}, {'rounds': [2, 9]}, {'difference_at_round': 0}, {'source_root': 'a'})
    for edit in edits:
        bad = tmp_path / 'bad.json'
        bad.write_text(json.dumps({**recipe, **edit}))
        assert dispatch([*args, str(bad)]) == 2
    old = _CONFIG.with_name('innovation1-present-prefix2-bridge.json')
    assert dispatch([*args, str(old)]) == 2
    assert old_cli.main(['check', '--config', str(_CONFIG)]) == 2
    assert old_cli.main(['check', '--config', str(old)]) == 0
    assert not (tmp_path / 'data').exists()


def test_launcher_only_explicit_fresh_dispatch() -> None:
    script = (_LAB / 'scripts' / 'start-cube-sharedm.ps1').read_text()
    assert f"$s2Window = $ConfigName -ieq '{_CONFIG.name}'" in script
    assert f"if ($s2Window) {{ $wallSeconds = 3600; $plugin = '{_PLUGIN}' }}" in script
    assert ("($prefixBridge -or $s2Window) -and "
            "$PSBoundParameters.ContainsKey('SourceRoot')") in script
    assert "if ($reuse) { $cliArgs += @('--source-root', $SourceRoot) }" in script
    assert '$code = $global:LASTEXITCODE' in script


def test_source_extends_actual_current_prefix_closure() -> None:
    from cipher_lab_plugins.innovation1.prefix_bridge.source import source_identity as old
    from cipher_lab_plugins.innovation1.s2_window.source import source_identity

    got = source_identity()
    prefix = 'cipher_lab_plugins/innovation1/s2_window/'
    actual = old()
    assert {name: got[name] for name in actual} == actual
    required = {prefix + p.name for p in (_LAB / 'plugins' / prefix).glob('*.py')}
    assert got.keys() - actual.keys() == required
    assert required and not any('tests/' in name or '__pycache__' in name for name in got)
    assert got['cipher_lab/ciphers/present_s2.py'] == sha256(
        (_LAB / 'src/cipher_lab/ciphers/present_s2.py').read_bytes()).hexdigest()


@pytest.mark.parametrize('fault', ['', 'device', 'deadline', 'data', 'report'])
def test_cli_run_with_artificial_admission_and_saved_terminal(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import resources
    from cipher_lab_plugins.innovation1.prefix_bridge import runtime
    from cipher_lab_plugins.innovation1.s2_window import cli, report, source, training

    entered: list[str] = []
    terminal_hash: list[str] = []
    now = datetime(2026, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(cli, 'Clocks', lambda: Clocks(lambda: 0., lambda: 0., lambda: now))
    monkeypatch.setattr(resources, 'working_set_bytes', lambda: 1)
    monkeypatch.setattr(source, 'source_identity', lambda: {'artificial/a.py': 'a' * 64})
    monkeypatch.setattr(runtime, 'configure', lambda: {'artificial': True})

    def device() -> dict[str, bool]:
        if fault == 'device':
            raise ValueError('internal-only')
        entered.append('device')
        return {'artificial': True}

    def train(meter: Any, store: Any, summary: dict[str, object], **kwargs: Any) -> None:
        from cipher_lab_plugins.innovation1.s2_window import data, kat

        assert kwargs['gate'] is kat.production and kwargs['make_panels'] is data.production_panels
        assert meter._policy.real_blocks == 1769477 and meter._policy.forward_rows == 1261568
        assert meter._policy.wall_seconds == 3600 and meter._policy.cpu_seconds == 14400
        assert meter._policy.output_bytes == 512*2**20 and meter._policy.rss_stop_bytes == 8*2**30
        entered.append('train')
        summary['phase'] = 'TRAIN_DEV_DATA'
        if fault == 'deadline':
            raise Stopped('DEADLINE')
        if fault == 'data':
            raise ValueError('internal-only')
        store.finalize({'status': 'SUCCEEDED', 'summary': {'phase': 'COMPLETE', 'test': {}}})

    def render(saved: dict[str, object]) -> dict[str, bytes]:
        assert saved['status'] == 'SUCCEEDED' and entered == ['device', 'train']
        entered.append('report')
        terminal_hash.append(str(saved['run_sha256']))
        if fault == 'report':
            raise ValueError('internal-only')
        ref = {key: saved[key] for key in ('run_id', 'run_ref', 'run_sha256')}
        return {'report.md': b'fake', 'index.html': b'fake', 'source.json': json.dumps(
            {'schema': 1, **ref}).encode()}

    monkeypatch.setattr(cli, '_device', device, raising=False)
    monkeypatch.setattr(training, 'run', train)
    monkeypatch.setattr(report, 'report_files', render)
    args = ['run', '--plugin', _PLUGIN, '--config', str(_CONFIG), '--root', str(tmp_path),
            '--run-id', 'fake', '--deadline-utc', '2026-01-01T01:00:00Z']
    assert dispatch([*args, '--source-root', '']) == 2
    assert dispatch([*args[:-1], '2026-01-01T01:00:01Z']) == 2
    assert not entered and not (tmp_path / 'data').exists()
    capsys.readouterr()
    status, reason, code = {
        '': ('SUCCEEDED', 'COMPLETE', 0), 'report': ('SUCCEEDED', 'REPORT_FAILED', 4),
        'data': ('FAILED', 'TRAIN_DEV_DATA_FAILED', 4), 'deadline': ('STOPPED', 'DEADLINE', 3),
        'device': ('FAILED', 'DEVICE_FAILED', 4),
    }[fault]
    assert dispatch(args) == code
    saved = read_saved(tmp_path, 'fake')
    assert saved['status'] == status
    assert ('test' in saved['summary']) == (status == 'SUCCEEDED')  # type: ignore[operator]
    assert ('report' in entered) == (status == 'SUCCEEDED')
    if status == 'SUCCEEDED':
        assert terminal_hash == [saved['run_sha256']]
    out = capsys.readouterr()
    assert json.loads(out.err if code else out.out)['reason'] == reason
    assert 'internal-only' not in str(saved) + str(out)

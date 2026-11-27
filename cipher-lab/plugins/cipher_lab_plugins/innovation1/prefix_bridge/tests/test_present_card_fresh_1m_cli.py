from __future__ import annotations

import json
import sys
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from cipher_lab.plugin_cli import dispatch


@pytest.mark.parametrize('module', [
    'cipher_lab_plugins.innovation1.source',
    'cipher_lab_plugins.direct_block.source',
    'cipher_lab_plugins.innovation1.prefix_bridge.source',
    'cipher_lab.artifacts',
])
def test_legacy_source_omits_card_public_file(module: str) -> None:
    source = import_module(module).source_identity()
    assert 'cipher_lab/present_public.py' not in source
    assert 'cipher_lab_plugins/innovation1/present_card_fresh/present_public.py' not in source


def test_source_closure_and_launcher() -> None:
    from cipher_lab_plugins.innovation1.present_card_fresh.source import source_identity

    source = source_identity()
    plugin = 'cipher_lab_plugins/innovation1/'
    for name in ('present_card_fresh/layout.py', 'four_single/models.py', 'mask_aux/kat.py',
                 'mask_aux_single/data.py', 's2_window/training.py', 'prefix_bridge/runtime.py'):
        data = Path('plugins', plugin + name).read_bytes()
        assert source[plugin + name] == sha256(data).hexdigest()
    public_key = plugin + 'present_card_fresh/present_public.py'
    public_digest = sha256(Path('plugins', public_key).read_bytes()).hexdigest()
    assert source[public_key] == public_digest
    assert public_digest == '4599a76e5012582911a503b6127f8347204fae73fc014173c8cb2d4bba36652f'
    assert 'cipher_lab/present_public.py' not in source
    excluded = ('tests/', '__pycache__', 'replay.py', 's2_window/data.py')
    assert not any(part in p for p in source for part in excluded)
    launcher = Path('scripts/start-cube-sharedm.ps1').read_text()
    assert "if ($cardFresh -and $PSBoundParameters.ContainsKey('SourceRoot'))" in launcher
    assert ("if ($cardFresh) { $wallSeconds = 7200; "
            "$plugin = 'cipher_lab_plugins.innovation1.present_card_fresh.cli' }") in launcher


def test_exact_check_and_no_source_root(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.present_card_fresh import cli

    args = ['--plugin', cli.__name__, 'check', '--config']
    path = 'configs/innovation1-present-card-fresh-1m.json'
    modules = set(sys.modules)
    assert dispatch([*args, path]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'VALID'
    lazy = ('present_card_fresh.runtime', 'present_card_fresh.model', 'present_card_fresh.training')
    assert not any(p.endswith(lazy) for p in sys.modules.keys() - modules)
    invalid = tmp_path / 'other.json'
    invalid.write_text(Path(path).read_text().replace('500000', '500001'))
    assert dispatch([*args, str(invalid)]) == 2

    def no_run(*args: object, **kwargs: object) -> None:
        pytest.fail('invalid input reached the runner')

    monkeypatch.setattr(cli, '_run', no_run)
    assert dispatch(['--plugin', cli.__name__, 'run', '--config', path, '--root', str(tmp_path),
                     '--run-id', 'toy', '--deadline-utc', '2026-01-01T00:00:00Z',
                     '--source-root', str(tmp_path)]) == 2


def test_saved_only_report_escapes_and_marks_exits_unknown() -> None:
    from cipher_lab_plugins.innovation1.present_card_fresh.report import report_files

    saved: dict[str, Any] = {
        'run_id': 'toy', 'run_ref': 'data/toy/run.json', 'run_sha256': 'b'*64,
        'status': 'FAILED', 'recipe': {}, 'summary': {'phase': '<toy>', 'test': {'pairs': 1}},
    }
    files = report_files(saved)
    assert set(files) == {'index.html', 'report.md', 'source.json'}
    assert b'&lt;toy&gt;' in files['index.html'] and b'<toy>' not in files['index.html']
    assert b'native exit: UNKNOWN' in files['report.md']
    assert b'SSH exit: UNKNOWN' in files['report.md']
    assert b'"pairs"' not in files['report.md']


def test_production_owner_wiring_is_lazy_and_binds_one_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    from cipher_lab_plugins.direct_block.cube_compare import resources
    from cipher_lab_plugins.direct_block.cube_compare.preflight import gpu_peaks
    from cipher_lab_plugins.innovation1.mask_aux import kat
    from cipher_lab_plugins.innovation1.prefix_bridge import runtime as bridge
    from cipher_lab_plugins.innovation1.present_card_fresh import cli, config, source, training
    from cipher_lab_plugins.innovation1.present_card_fresh.data import production_panels
    from cipher_lab_plugins.innovation1.present_card_fresh.runtime import FitRuntime

    from cipher_lab.artifacts import RunStore, read_saved
    from cipher_lab.execution import Clocks, Meter, Stopped

    fixed_source = {'artificial/owner.py': 'a'*64}
    monkeypatch.setattr(source, 'source_identity', lambda: fixed_source.copy())
    monkeypatch.setattr(resources, 'working_set_bytes', lambda: 1)
    monkeypatch.setattr(cli, '_device', lambda: {'artificial': True})
    monkeypatch.setattr(bridge, 'configure', lambda: {'artificial': True})
    monkeypatch.setattr(cli, 'Clocks', lambda: Clocks(
        lambda: 0., lambda: 0., lambda: datetime(2026, 1, 1, tzinfo=UTC)))
    calls: list[str] = []

    def fake_run(meter: Meter, store: RunStore, summary: dict[str, object], **deps: Any) -> None:
        assert deps['gate'] is kat.production and deps['make_panels'] is production_panels
        assert deps['peaks'] is gpu_peaks and summary['runtime'] == {'artificial': True}
        assert store._model_slots == {config.SLOT}
        assert meter._policy.optimizer_updates == 3000
        runtime = deps['make'](config.SLOT, {}, False)
        assert isinstance(runtime, FitRuntime) and runtime._source == fixed_source
        assert runtime._run_id == 'toy' and not runtime._runtime.models
        calls.append('run')
        meter.reserve('real_blocks', 1)
        raise Stopped('DEADLINE')

    monkeypatch.setattr(training, 'run', fake_run)
    assert cli.main(['run', '--config', 'configs/innovation1-present-card-fresh-1m.json',
                     '--root', str(tmp_path), '--run-id', 'toy', '--deadline-utc',
                     '2026-01-01T02:00:00Z']) == 3
    saved = read_saved(tmp_path, 'toy')
    assert calls == ['run'] and saved['status'] == 'STOPPED'
    assert isinstance(saved['counts'], dict) and saved['counts']['attempted_real_blocks'] == 1
    assert isinstance(saved['summary'], dict) and 'test' not in saved['summary']

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

_LAB = Path(__file__).resolve().parents[5]
_PLUGIN = 'cipher_lab_plugins.innovation1.mask_aux_single.cli'
_CONFIG = 'innovation1-present-single-pair-mask-10m.json'


def test_exact_check_and_fixed_launcher(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab.cli import main

    args = ['check', '--plugin', _PLUGIN, '--config']
    assert main([*args, str(_LAB / 'configs' / _CONFIG)]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed['status'] == 'VALID'
    recipe = parsed['recipe']
    assert (recipe['wall_seconds'], recipe['cpu_seconds'], recipe['rss_bytes'],
            recipe['output_bytes'], recipe['checkpoint_bytes']) == (
        43200, 172800, 8*2**30, 512*2**20, 16*2**20)
    bad = tmp_path / 'wrong-config.json'
    changes: tuple[dict[str, object], ...] = (
        {'train_rows': 10_000_000}, {'early_stop': {}}, {'gpu_device': 'cuda:1'},
        {'batch_size': 512.0}, {'source_root': '.'},
    )
    for change in changes:
        bad.write_text(json.dumps({**recipe, **change}))
        assert main([*args, str(bad)]) == 2
    bad.write_bytes((_LAB / 'configs/innovation1-present-mask-aux.json').read_bytes())
    assert main([*args, str(bad)]) == 2
    script = (_LAB / 'scripts/start-cube-sharedm.ps1').read_text()
    assert f"$singlePairMask = $ConfigName -ieq '{_CONFIG}'" in script
    assert f"if ($singlePairMask) {{ $wallSeconds = 43200; $plugin = '{_PLUGIN}' }}" in script
    assert "$singlePairMask -and $PSBoundParameters.ContainsKey('SourceRoot')" in script
    assert 'GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5' in script


def test_source_closure_and_actual_bytes() -> None:
    from cipher_lab_plugins.innovation1.mask_aux.source import source_identity as predecessor
    from cipher_lab_plugins.innovation1.mask_aux_single.source import source_identity

    hashes = source_identity()
    old = predecessor()
    assert all(hashes[path] == digest for path, digest in old.items())
    owner = 'cipher_lab_plugins/innovation1/mask_aux_single/'
    modules = ('__init__', 'cli', 'config', 'data', 'model', 'report', 'runtime',
               'source', 'training', 'weights')
    assert hashes.keys() - old.keys() == {owner + name + '.py' for name in modules}
    for name, digest in hashes.items():
        base = 'src' if name.startswith('cipher_lab/') else 'plugins'
        assert sha256((_LAB / base / name).read_bytes()).hexdigest() == digest
        assert '/tests/' not in name and '__pycache__' not in name


def test_owner_run_wiring_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from datetime import UTC, datetime
    from typing import Any

    from cipher_lab_plugins.direct_block.cube_compare import resources
    from cipher_lab_plugins.direct_block.cube_compare.preflight import gpu_peaks
    from cipher_lab_plugins.innovation1.mask_aux import kat
    from cipher_lab_plugins.innovation1.mask_aux_single import cli, config, source, training
    from cipher_lab_plugins.innovation1.mask_aux_single.data import production_panels
    from cipher_lab_plugins.innovation1.mask_aux_single.runtime import FitRuntime
    from cipher_lab_plugins.innovation1.prefix_bridge import runtime

    from cipher_lab.artifacts import read_saved
    from cipher_lab.execution import Clocks, Stopped

    fixed_source = {'artificial/owner.py': 'a'*64}
    monkeypatch.setattr(source, 'source_identity', lambda: fixed_source)
    monkeypatch.setattr(resources, 'working_set_bytes', lambda: 1)
    monkeypatch.setattr(cli, '_device', lambda: {'fake_device': True})
    monkeypatch.setattr(runtime, 'configure', lambda: {'fake_configure': True})
    monkeypatch.setattr(cli, 'Clocks', lambda: Clocks(
        lambda: 0., lambda: 0., lambda: datetime(2026, 1, 1, tzinfo=UTC)))
    called: list[bool] = []

    def run(meter: Any, store: Any, summary: dict[str, object], **deps: Any) -> None:
        assert deps['gate'] is kat.production and deps['make_panels'] is production_panels
        assert deps['peaks'] is gpu_peaks
        assert summary['runtime'] == {'fake_configure': True}
        for slot in config.SLOTS:
            e = deps['make'](slot, {}, False)
            assert isinstance(e, FitRuntime) and e._source == fixed_source
            assert e._run_id == 'pair-cli' and e._slot == slot and not e._runtime.models
        called.append(True)
        meter.reserve('real_blocks', 4)
        raise Stopped('DEADLINE')

    monkeypatch.setattr(training, 'run', run)
    args = ['run', '--config', str(_LAB / 'configs' / _CONFIG), '--root', str(tmp_path),
            '--run-id', 'pair-cli', '--deadline-utc', '2026-01-01T12:00:00Z']
    assert cli.main([*args, '--source-root', 'x']) == 2 and not called
    assert cli.main(args) == 3 and called == [True]
    saved = read_saved(tmp_path, 'pair-cli')
    assert saved['status'] == 'STOPPED' and saved['recipe'] == config.recipe()
    assert isinstance(saved['counts'], dict) and saved['counts']['attempted_real_blocks'] == 4
    assert isinstance(saved['summary'], dict) and 'test' not in saved['summary']

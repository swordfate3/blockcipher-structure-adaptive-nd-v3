from __future__ import annotations

import json
from pathlib import Path

import pytest

_LAB = Path(__file__).resolve().parents[5]
_PLUGIN = 'cipher_lab_plugins.innovation1.mask_aux.cli'
_CONFIG = 'innovation1-present-mask-aux.json'


def test_config_and_launcher_dispatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab.cli import main

    args = ['check', '--plugin', _PLUGIN, '--config']
    assert main([*args, str(_LAB / 'configs' / _CONFIG)]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed['status'] == 'VALID' and parsed['recipe']['wall_seconds'] == 3600
    parsed['recipe']['mask_seed'] += 1
    bad = tmp_path / 'config.json'
    bad.write_text(json.dumps(parsed['recipe']))
    assert main([*args, str(bad)]) == 2
    script = (_LAB / 'scripts/start-cube-sharedm.ps1').read_text()
    assert f"$maskAux = $ConfigName -ieq '{_CONFIG}'" in script
    assert f"if ($maskAux) {{ $wallSeconds = 3600; $plugin = '{_PLUGIN}' }}" in script
    assert "$maskAux -and $PSBoundParameters.ContainsKey('SourceRoot')" in script
    assert 'GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5' in script


def test_source_closure_includes_owner_and_imported_runtime() -> None:
    from cipher_lab_plugins.innovation1.mask_aux.source import source_identity

    source = source_identity()
    owner = 'cipher_lab_plugins/innovation1/mask_aux/'
    modules = ('__init__', 'cli', 'config', 'data', 'kat', 'masking', 'model', 'report',
               'runtime', 'source', 'training', 'weights')
    assert {name for name in source if name.startswith(owner)} == {
        owner + module + '.py' for module in modules}
    assert {'cipher_lab/ciphers/present_prefix.py', 'cipher_lab/ciphers/present.py',
            'cipher_lab/artifacts.py', 'cipher_lab_plugins/direct_block/fit.py',
            'cipher_lab_plugins/innovation1/models.py',
            'cipher_lab_plugins/innovation1/prefix_bridge/runtime.py',
            'cipher_lab_plugins/innovation1/single_train/weights.py'} <= source.keys()
    assert all('/tests/' not in name and '__pycache__' not in name for name in source)

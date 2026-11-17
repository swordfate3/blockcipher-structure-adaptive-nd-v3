"""Task5 public boundary tests: stdlib children install their own import trap."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cipher_lab_plugins.speck_study.config import SpeckStudyRecipe
from cipher_lab_plugins.speck_study.tests.clean_source import clean_source
from cipher_lab_plugins.speck_study.tests.locations import PLUGIN, ROOT, RUNTIME

CHILD = r'''
import sys, importlib.abc
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','numpy','tensorflow','h5py') or any(
            (fullname == 'cipher_lab.'+x or fullname.startswith('cipher_lab.'+x+'.')) for x in
            ('artifacts','execution','ciphers','data','training')):
            raise AssertionError('Task5 child scientific/machine import forbidden: '+fullname)
        if any(fullname == 'cipher_lab_plugins.speck_study.'+x
               or fullname.startswith('cipher_lab_plugins.speck_study.'+x+'.')
               for x in ('platform_resources','application','method','teacher')):
            raise AssertionError('plugin scientific/machine import forbidden: '+fullname)
sys.meta_path.insert(0,Guard())
from cipher_lab.cli import main
raise SystemExit(main(sys.argv[1:]))
'''


def policy(tmp_path: Path) -> Path:
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(dict(schema=2, wall_seconds=28800, cpu_seconds=1,
        deadline_utc='2099-01-01T00:00:00Z', real_blocks=542000000, null_words=0,
        rss_stop_bytes=32*2**30, output_bytes=4*2**30, optimizer_updates=0,
        forward_rows=1161000000)))
    return path


def child(args: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, '-B', '-c', CHILD, args[0], '--plugin',
                           'cipher_lab_plugins.speck_study.cli', *args[1:]], cwd=cwd,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONPATH':os.pathsep.join((str(cwd/'src'),str(cwd/'plugins')))},  # noqa: E501 - explicit migrated source/argv identity
        capture_output=True, text=True, timeout=15)


def test_check_formal_stdlib_only_no_claim_no_machine(tmp_path: Path) -> None:
    config = tmp_path/'recipe.json'
    config.write_text(json.dumps(SpeckStudyRecipe().to_dict()))
    p = policy(tmp_path)
    before = sorted(str(x) for x in tmp_path.rglob('*'))
    result = child(['check','--config',str(config),'--policy',str(p)])
    assert result.returncode == 0, result.stderr
    saved = json.loads(result.stdout)
    assert saved['device'] == 'gpu-uuid-and-pci-required'
    assert saved['estimate']['planned_real_blocks'] == 541000512
    assert saved['estimate']['planned_forward_rows'] == 1161000000
    assert set(saved['planned_calls_fit_policy'].values()) == {True}
    assert sorted(str(x) for x in tmp_path.rglob('*')) == before


@pytest.mark.parametrize('flag', ['--teacher-python','--teacher-weights','--gpu-uuid'])
def test_check_rejects_runtime_flags(tmp_path: Path, flag: str) -> None:
    result = child(['check','--config',str(ROOT/'configs/speck-study-paper.json'),
        '--policy',str(policy(tmp_path)),flag,'invalid'])
    assert result.returncode == 2
    assert not result.stdout


@pytest.mark.parametrize('backend', ['demo','simon','ciphers'])
def test_new_recipe_rejects_old_backend_before_claim(tmp_path: Path, backend: str) -> None:
    result = child(['run','--config',str(ROOT/'configs/speck-study-paper.json'),
        '--policy',str(policy(tmp_path)),'--root',str(tmp_path/'out'),'--id','test',
        '--backend',backend])
    assert result.returncode == 2
    assert not (tmp_path/'out').exists()


def test_new_backend_requires_runtime_options_before_claim(tmp_path: Path) -> None:
    result = child(['run','--config',str(ROOT/'configs/speck-study-paper.json'),
        '--policy',str(policy(tmp_path)),'--root',str(tmp_path/'out'),'--id','test',
        '--backend','speck-study'])
    assert result.returncode == 2
    assert 'invalid' in result.stderr
    assert not (tmp_path/'out').exists()


def test_public_import_closure_parses_as_py310() -> None:
    from cipher_lab_plugins.speck_study.tests.test_speck_source import tool
    module = tool()
    for name in module.CLOSURE:
        path = module.source_path(ROOT/'src',ROOT/'plugins',name)
        ast.parse(path.read_bytes(), filename=str(path), feature_version=(3,10))


def test_plugin_claim_declares_models_and_its_own_source_supplier() -> None:
    from cipher_lab_plugins.speck_study import cli

    tree = ast.parse(Path(cli.__file__).read_bytes())
    claims = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == 'claim']
    assert len(claims) == 1
    keywords = {keyword.arg: keyword.value for keyword in claims[0].keywords}
    assert 'needs_models' in keywords and 'source_supplier' in keywords
    assert isinstance(keywords['source_supplier'], ast.Name)
    assert keywords['source_supplier'].id == 'source_identity'


def test_both_shipped_configs_are_complete_formal_not_silent_demo() -> None:
    from cipher_lab_plugins.speck_study.cli import load_experiment
    for name in ('paper','example'):
        assert load_experiment(ROOT/f'configs/speck-study-{name}.json') == SpeckStudyRecipe()


def test_clean_source_fake_cli_check_run_read_saved_strict_codec(tmp_path: Path) -> None:
    import shutil

    from cipher_lab_plugins.speck_study.launch import OFFICIAL_REVISION, OFFICIAL_WEIGHTS_SHA256

    clean = clean_source(tmp_path/'clean')
    tools = clean/'plugins/cipher_lab_plugins/speck_study/runtime'
    p = policy(tmp_path)
    check = child(['check','--config','configs/speck-study-paper.json','--policy',str(p)],clean)
    assert check.returncode == 0, check.stderr
    # Metadata validation never reads/loads H5; this is explicitly fake raw bytes.
    weights = tmp_path/'fake.h5'
    weights.write_bytes(b'FAKE NEVER LOAD')
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    driver = tmp_path/'guarded_driver.py'
    shutil.copyfile(PLUGIN/'tests/speck_cli_child.py',driver)
    result = subprocess.run([sys.executable,'-B',str(driver),str(tools/'source.py'),
        str(clean/'src'),str(clean/'plugins'),str(manifest),'run',
        '--plugin','cipher_lab_plugins.speck_study.cli',
        '--config','configs/speck-study-paper.json','--policy',str(p),
        '--backend','speck-study','--root',str(tmp_path/'out'),'--id','clean-fake',
        '--teacher-python',str(Path(sys.executable).resolve()),'--teacher-weights',str(weights),
        '--weights-revision',OFFICIAL_REVISION,'--weights-sha256',OFFICIAL_WEIGHTS_SHA256,
        '--gpu-uuid','GPU-12345678-1234-1234-1234-123456789abc','--gpu-pci','0000:01:00.0',
        '--source-manifest',str(manifest)],cwd=clean,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':os.pathsep.join((str(clean/'src'),str(clean/'plugins')))},
        capture_output=True,text=True,timeout=20)
    assert result.returncode == 0, result.stderr+result.stdout
    assert json.loads(result.stdout)['backend_identity'] == 'speck-study'
    frozen = json.loads(manifest.read_text())
    assert 'cipher_lab_plugins/speck_study/codec.py' in frozen
    assert not list(clean.rglob('*.pyc'))
    assert (tmp_path/'out/models/clean-fake/A-best.pt').is_file()
    assert (tmp_path/'out/result/clean-fake/index.html').is_file()


@pytest.mark.parametrize('field,value', [
    ('revision','main'), ('weights_sha256','0'*64), ('gpu_uuid','GPU-1'),
    ('gpu_pci','ordinal1'), ('teacher_python',Path('/missing/python')),
])
def test_launch_rejects_unpinned_metadata_and_missing_exe(
    tmp_path: Path, field: str, value: object
) -> None:
    from cipher_lab_plugins.speck_study.launch import (
        OFFICIAL_REVISION,
        OFFICIAL_WEIGHTS_SHA256,
        LaunchSettings,
    )
    weight = tmp_path/'fake.h5'
    weight.write_bytes(b'FAKE')
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    args = dict(teacher_python=Path(sys.executable).resolve(),weights=weight,
        revision=OFFICIAL_REVISION,weights_sha256=OFFICIAL_WEIGHTS_SHA256,
        gpu_uuid='GPU-12345678-1234-1234-1234-123456789abc',gpu_pci='0000:01:00.0',
        source_manifest=manifest)
    args[field] = value
    with pytest.raises(ValueError):
        LaunchSettings(**args)  # type: ignore[arg-type]


@pytest.mark.parametrize('stamp', ['2099-01-01T00:00:00Z','2099-01-01T00:00:00+00:00',
                                  '2099-01-01T00:00:00.123456Z'])
def test_py310_z_normalization_keeps_old_accepted_values(tmp_path: Path, stamp: str) -> None:
    from cipher_lab.config import load_policy
    path = policy(tmp_path)
    value = json.loads(path.read_text())
    value['deadline_utc'] = stamp
    path.write_text(json.dumps(value))
    assert load_policy(path).deadline_utc == stamp


@pytest.mark.parametrize('stamp', ['2099-01-01','2099-01-01T00:00:00z',
                                  '2099-01-01T00:00:00-00:00',
                                  '2099-01-01T00:00:00.1234567Z','2099-02-30T00:00:00Z'])
def test_py310_normalization_does_not_broaden_old_rejections(tmp_path: Path, stamp: str) -> None:
    from cipher_lab.config import load_policy
    path = policy(tmp_path)
    value = json.loads(path.read_text())
    value['deadline_utc'] = stamp
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_policy(path)


def test_actual_native_cli_saves_blocked_no_scientific_import_or_false_success(
    tmp_path: Path,
) -> None:
    from cipher_lab_plugins.speck_study.launch import OFFICIAL_REVISION, OFFICIAL_WEIGHTS_SHA256

    weight = tmp_path/'fake.h5'
    weight.write_bytes(b'not H5, must not read')
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    script = r'''
import sys, importlib.abc
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','numpy','tensorflow','h5py') or any(
            fullname == 'cipher_lab.'+x or fullname.startswith('cipher_lab.'+x+'.')
            for x in ('ciphers','data','training','platform_resources')):
            raise AssertionError('native blocked path touched science/machine '+fullname)
sys.meta_path.insert(0,Guard())
from uuid import UUID
import cipher_lab.artifacts as artifacts
artifacts.uuid4=lambda:UUID(int=1)
from cipher_lab.cli import main
raise SystemExit(main(sys.argv[1:]))
'''
    clean = clean_source(tmp_path/'clean')
    out = tmp_path/'blocked'
    result = subprocess.run([sys.executable,'-B','-c',script,'run','--plugin','cipher_lab_plugins.speck_study.cli',  # noqa: E501 - explicit migrated source/argv identity
        '--config',str(clean/'configs/speck-study-paper.json'),'--policy',str(policy(tmp_path)),
        '--backend','speck-study','--root',str(out),'--id','blocked',
        '--teacher-python',str(Path(sys.executable).resolve()),'--teacher-weights',str(weight),
        '--weights-revision',OFFICIAL_REVISION,'--weights-sha256',OFFICIAL_WEIGHTS_SHA256,
        '--gpu-uuid','GPU-12345678-1234-1234-1234-123456789abc','--gpu-pci','0000:01:00.0',
        '--source-manifest',str(manifest)],cwd=clean,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1',
             'PYTHONPATH':os.pathsep.join((str(clean/'src'),str(clean/'plugins')))},
        capture_output=True,text=True,timeout=15)
    assert result.returncode == 3, result.stderr+result.stdout
    exit_record = json.loads(result.stdout)
    assert exit_record['scientific_status'] == 'STOPPED'
    assert exit_record['scientific_validation'] is False
    saved = json.loads((out/'data/blocked/run.json').read_text())
    assert saved['summary']['reason'] == 'ENVIRONMENT_BLOCKED'
    assert saved['summary']['test'] is None and not saved['counts']
    assert not list((out/'models/blocked').iterdir())


def test_old_method_cannot_use_study_backend_before_claim(tmp_path: Path) -> None:
    result = child(['run','--config',str(ROOT/'configs/integral-example.json'),
        '--policy',str(policy(tmp_path)),'--root',str(tmp_path/'out'),'--id','test',
        '--backend','speck-study'])
    assert result.returncode == 2 and not (tmp_path/'out').exists()


def test_source_frozen_entry_check_executes_only_stdlib(tmp_path: Path) -> None:
    source_script = RUNTIME/'source.py'
    script = CHILD[:CHILD.index('from cipher_lab.cli import main')] + r'''
import importlib.util, json
from pathlib import Path
spec=importlib.util.spec_from_file_location('frozen_main',sys.argv[1])
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
root=Path(sys.argv[2]); plugins=Path(sys.argv[3]); manifest=Path(sys.argv[4])
manifest.write_text(json.dumps(module.snapshot(root,plugins)))
raise SystemExit(module.main(['run',str(manifest),str(root),str(plugins)]+sys.argv[5:]))
'''
    manifest = tmp_path/'manifest.json'
    result = subprocess.run([sys.executable,'-B','-c',script,str(source_script),str(ROOT/'src'),
        str(ROOT/'plugins'),str(manifest),'check','--plugin','cipher_lab_plugins.speck_study.cli','--config',str(ROOT/'configs/speck-study-paper.json'),
        '--policy',str(policy(tmp_path))],cwd=ROOT,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':os.pathsep.join((str(ROOT/'src'),str(ROOT/'plugins')))},
        capture_output=True,text=True,timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['estimate']['teacher_rows'] == 1161000000

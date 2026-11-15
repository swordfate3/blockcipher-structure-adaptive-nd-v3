"""Task5 public boundary tests: stdlib children install their own import trap."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cipher_lab.speck_study_config import SpeckStudyRecipe

ROOT = Path(__file__).resolve().parents[1]
CHILD = r'''
import sys, importlib.abc
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','numpy','tensorflow','h5py') or any(
            (fullname == 'cipher_lab.'+x or fullname.startswith('cipher_lab.'+x+'.')) for x in
            ('artifacts','execution','ciphers','data','training','platform_resources','speck_study')):
            raise AssertionError('Task5 child scientific/machine import forbidden: '+fullname)
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
    return subprocess.run([sys.executable, '-B', '-c', CHILD, *args], cwd=cwd,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONPATH':str(cwd/'src')},
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
    for name in ('cli','config','training_config','speck_study_config',
                 'execution','artifacts'):
        path = ROOT/'src/cipher_lab'/f'{name}.py'
        ast.parse(path.read_bytes(), filename=str(path), feature_version=(3,10))


def test_head_model_types_plus_study_are_preserved() -> None:
    from cipher_lab.cli import _needs_models
    from cipher_lab.config import Recipe
    from cipher_lab.training_config import PredictionRecipe, SharedMRecipe
    for cls in (PredictionRecipe, SharedMRecipe, SpeckStudyRecipe):
        assert _needs_models(object.__new__(cls))
    assert not _needs_models(object.__new__(Recipe))
    assert not _needs_models(object())


def test_both_shipped_configs_are_complete_formal_not_silent_demo() -> None:
    from cipher_lab.config import load_experiment
    for name in ('paper','example'):
        assert load_experiment(ROOT/f'configs/speck-study-{name}.json') == SpeckStudyRecipe()


def test_clean_source_fake_cli_check_run_read_saved_strict_codec(tmp_path: Path) -> None:
    import shutil

    from cipher_lab.speck_study_launch import OFFICIAL_REVISION, OFFICIAL_WEIGHTS_SHA256

    clean = tmp_path/'clean'
    clean.mkdir()
    shutil.copytree(ROOT/'src',clean/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'configs',clean/'configs')
    for name in ('pyproject.toml','.gitignore'):
        shutil.copyfile(ROOT/name,clean/name)
    tools = clean/'tools/speck-study'
    tools.mkdir(parents=True)
    shutil.copyfile(ROOT/'tools/speck-study/source.py',tools/'source.py')
    p = policy(tmp_path)
    check = child(['check','--config','configs/speck-study-paper.json','--policy',str(p)],clean)
    assert check.returncode == 0, check.stderr
    # Metadata validation never reads/loads H5; this is explicitly fake raw bytes.
    weights = tmp_path/'fake.h5'
    weights.write_bytes(b'FAKE NEVER LOAD')
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    driver = tmp_path/'guarded_driver.py'
    shutil.copyfile(ROOT/'tests/speck_cli_child.py',driver)
    result = subprocess.run([sys.executable,'-B',str(driver),str(tools/'source.py'),
        str(clean/'src'),str(manifest),'run',
        '--config','configs/speck-study-paper.json','--policy',str(p),
        '--backend','speck-study','--root',str(tmp_path/'out'),'--id','clean-fake',
        '--teacher-python',str(Path(sys.executable).resolve()),'--teacher-weights',str(weights),
        '--weights-revision',OFFICIAL_REVISION,'--weights-sha256',OFFICIAL_WEIGHTS_SHA256,
        '--gpu-uuid','GPU-12345678-1234-1234-1234-123456789abc','--gpu-pci','0000:01:00.0',
        '--source-manifest',str(manifest)],cwd=clean,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(clean/'src')},
        capture_output=True,text=True,timeout=20)
    assert result.returncode == 0, result.stderr+result.stdout
    assert json.loads(result.stdout)['backend_identity'] == 'speck-study'
    frozen = json.loads(manifest.read_text())
    assert 'cipher_lab/speck_study/codec.py' in frozen
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
    from cipher_lab.speck_study_launch import (
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
    from cipher_lab.speck_study_launch import OFFICIAL_REVISION, OFFICIAL_WEIGHTS_SHA256

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
    out = tmp_path/'blocked'
    result = subprocess.run([sys.executable,'-B','-c',script,'run',
        '--config',str(ROOT/'configs/speck-study-paper.json'),'--policy',str(policy(tmp_path)),
        '--backend','speck-study','--root',str(out),'--id','blocked',
        '--teacher-python',str(Path(sys.executable).resolve()),'--teacher-weights',str(weight),
        '--weights-revision',OFFICIAL_REVISION,'--weights-sha256',OFFICIAL_WEIGHTS_SHA256,
        '--gpu-uuid','GPU-12345678-1234-1234-1234-123456789abc','--gpu-pci','0000:01:00.0',
        '--source-manifest',str(manifest)],cwd=ROOT,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'src')},
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
    source_script = ROOT/'tools/speck-study/source.py'
    script = CHILD[:CHILD.index('from cipher_lab.cli import main')] + r'''
import importlib.util, json
from pathlib import Path
spec=importlib.util.spec_from_file_location('frozen_main',sys.argv[1])
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
root=Path(sys.argv[2]); manifest=Path(sys.argv[3])
manifest.write_text(json.dumps(module.snapshot(root)))
raise SystemExit(module.main(['run',str(manifest),str(root)]+sys.argv[4:]))
'''
    manifest = tmp_path/'manifest.json'
    result = subprocess.run([sys.executable,'-B','-c',script,str(source_script),str(ROOT/'src'),
        str(manifest),'check','--config',str(ROOT/'configs/speck-study-paper.json'),
        '--policy',str(policy(tmp_path))],cwd=ROOT,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'src')},
        capture_output=True,text=True,timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['estimate']['teacher_rows'] == 1161000000


@pytest.mark.parametrize('raw,canonical', [
    ('00000000:AF:00.0', '0000:af:00.0'),
    ('0000:AF:00.0', '0000:af:00.0'),
    ('0000af01:AB:CD.7', 'af01:ab:cd.7'),
    ('AF01:AB:CD.7', 'af01:ab:cd.7'),
])
def test_launch_pci_matches_binding_normalization(
    tmp_path: Path, raw: str, canonical: str,
) -> None:
    from cipher_lab.platform_resources.gpu import GpuBinding, GpuError, pci, verify_runtime
    from cipher_lab.speck_study_launch import (
        OFFICIAL_REVISION,
        OFFICIAL_WEIGHTS_SHA256,
        LaunchSettings,
    )
    weight = tmp_path / 'fake.h5'
    weight.write_bytes(b'NEVER LOAD')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    uuid = 'GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5'
    settings = LaunchSettings(Path(sys.executable).resolve(), weight, OFFICIAL_REVISION,
                              OFFICIAL_WEIGHTS_SHA256, uuid, raw, manifest)
    assert settings.gpu_pci == canonical == pci(raw)
    # Same model name intentionally irrelevant: identity remains UUID AND PCI.
    binding = GpuBinding.from_snapshot(1, [dict(index=0, model='same',
        gpu_uuid='GPU-00000000-0000-0000-0000-000000000000', pci_bus_id='0000:01:00.0'),
        dict(index=1, model='same', gpu_uuid=uuid, pci_bus_id=raw)])
    assert (binding.gpu_uuid, binding.pci_bus_id) == (settings.gpu_uuid, settings.gpu_pci)
    evidence = dict(gpu_uuid=uuid, pci_bus_id=raw, logical_device=0, pid=1,
                    evidence='cuda-current-context')
    verify_runtime(binding, evidence)  # pure synthetic identity comparison
    for field, wrong in (('gpu_uuid', 'GPU-other'), ('pci_bus_id', '0000:02:00.0')):
        with pytest.raises(GpuError):
            verify_runtime(binding, {**evidence, field: wrong})


@pytest.mark.parametrize('raw', [True, False, None, 0, '', 'ordinal1',
    '10000000:AF:00.0', '00010000:AF:00.0', '00000:AF:00.0', '0000000:AF:00.0',
    '000000000:AF:00.0', '0x000000:AF:00.0', '00000000:AF:00.8', '0000:AF:00.-1',
    '0000:AG:00.0', '0000:AF:0.0', '0000:AF:00.00', '0000:AF:00.0\n',
    ' 0000:AF:00.0', '0000:AF:00.0 '])
def test_launch_pci_rejects_non_equivalent_or_malformed(tmp_path: Path, raw: object) -> None:
    from cipher_lab.speck_study_launch import (
        OFFICIAL_REVISION,
        OFFICIAL_WEIGHTS_SHA256,
        LaunchSettings,
    )
    weight = tmp_path / 'fake.h5'
    weight.write_bytes(b'NEVER LOAD')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    with pytest.raises(ValueError):
        LaunchSettings(Path(sys.executable).resolve(), weight, OFFICIAL_REVISION,
            OFFICIAL_WEIGHTS_SHA256, 'GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5',
            raw, manifest)  # type: ignore[arg-type]

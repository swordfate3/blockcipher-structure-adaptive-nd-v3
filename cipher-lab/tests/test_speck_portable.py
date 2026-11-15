"""Local acceptance only; portable entry itself must never require pytest."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'tests/speck_portable_selfcheck.py'


@pytest.mark.parametrize('mode', ['main', 'provider'])
def test_portable_owned_guarded_entry(mode: str, tmp_path: Path) -> None:
    owned = tmp_path / mode
    result = subprocess.run(
        [sys.executable, '-B', str(ENTRY), '--mode', mode, '--rootowned', str(owned)],
        cwd=ROOT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary['status'] == '通过'
    assert summary['counts']['real_blocks'] == summary['counts']['real_labels'] == 0
    assert summary['guards']['installed'] and summary['guards']['violations'] == 0
    assert summary['checks'] >= 3
    assert summary['actual_interpreter'] == sys.executable
    assert summary['actual_version'] == sys.version
    assert summary['scope'] == 'synthetic-only; formal R0 compatibility NOT verified'
    assert not list(owned.rglob('*.pyc'))
    assert not list(owned.rglob('.pytest_cache'))
    if mode == 'provider':
        assert summary['checks'] == 22
    else:
        assert summary['checks'] == 8
        saved = json.loads((owned / 'out/data/portable-fake/run.json').read_text())
        assert saved['environment']['python_implementation'] == 'CPython'
    again = subprocess.run(
        [sys.executable, '-B', str(ENTRY), '--mode', mode, '--rootowned', str(owned)],
        cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert again.returncode != 0


def test_windows_identity_marker_before_claim(tmp_path: Path) -> None:
    """Child runs its real claim; synthetic identity must not use uname/ver."""
    script = '''
import platform, runpy, sys
# Model the actual Windows platform probe conflict even on Linux.
def windows_ver_probe():
    import subprocess
    subprocess.check_output('ver', shell=True)
    raise AssertionError('Windows probe must not be called by synthetic claim')
platform.system = windows_ver_probe
platform.machine = windows_ver_probe
sys.argv = [sys.argv[1], '--mode', 'main', '--rootowned', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(ENTRY),
                             str(tmp_path / 'win')], cwd=ROOT,
                            capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    saved = json.loads((tmp_path / 'win/out/data/portable-fake/run.json').read_text())
    assert saved['environment']['os'] == 'Windows'
    assert saved['environment']['architecture'] == 'AMD64'
    summary = json.loads(result.stdout)
    assert summary['synthetic_environment_identity'] == {'os': 'Windows', 'architecture': 'AMD64'}


@pytest.mark.parametrize('module', ['numpy', 'numpy.random', 'pytest', 'torch',
                                   'tensorflow', 'h5py', 'cipher_lab.ciphers.speck'])
def test_import_gate_precedes_project_and_science(module: str, tmp_path: Path) -> None:
    script = '''
import importlib, runpy, sys
ns = runpy.run_path(sys.argv[1], run_name='portable_import_gate_test')
trap = ns['Traps'](deny_numpy=True)
trap.install()
sys.path.insert(0, str(ns['ROOT'] / 'src'))
try:
    importlib.import_module(sys.argv[2])
except AssertionError:
    assert trap.violations == 1
else:
    raise AssertionError('import gate failed')
assert 'numpy' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(ENTRY), module],
                            cwd=tmp_path, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('call', ['ctypes.CDLL("never-load")',
                                 'ctypes.WinDLL("never-load")',
                                 'ctypes.windll.LoadLibrary("never-load")',
                                 'subprocess.run(["never-start"])',
                                 'os.urandom(8)', 'secrets.token_bytes(8)'])
def test_native_process_and_entropy_traps(call: str, tmp_path: Path) -> None:
    script = '''
import ctypes, os, runpy, secrets, subprocess, sys
ns = runpy.run_path(sys.argv[1], run_name='portable_native_gate_test')
trap = ns['Traps']()
trap.install()
trap.finish_entropy()
try:
    eval(sys.argv[2])
except AssertionError:
    assert trap.violations == 1
else:
    raise AssertionError('native/process/entropy trap failed')
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(ENTRY), call],
                            cwd=tmp_path, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr



def test_provider_does_not_bootstrap_numpy_native_runtime(tmp_path: Path) -> None:
    """Reproduce NumPy1.21 Windows distributor WinDLL bootstrap without a DLL."""
    script = '''
import ctypes, importlib.abc, runpy, sys
from pathlib import Path
ns = runpy.run_path(sys.argv[1], run_name='portable_numpy_bootstrap_test')
trap = ns['Traps']()
trap.install()
class WindowsNumpyBootstrap(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'numpy':
            ctypes.WinDLL('NEVER-LOAD-numpy-distributor')
sys.meta_path.insert(0, WindowsNumpyBootstrap())
assert ns['worker_provider'](Path(sys.argv[2]), trap) == 22
assert trap.violations == 0
assert not any(n == 'numpy' or n.startswith('numpy.') for n in sys.modules)
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(ENTRY), str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_provider_manual_array_strict_shape_and_cleanup(tmp_path: Path) -> None:
    script = '''
import runpy, sys
ns = runpy.run_path(sys.argv[1], run_name='portable_manual_array_test')
trap = ns['Traps'](deny_numpy=True)
trap.install()
trap.finish_entropy()
sys.path.insert(0, str(ns['ROOT'] / 'src'))
from cipher_lab.speck_study.teacher_network import points_to_bits
from cipher_lab.speck_study.teacher_protocol import ProtocolError
convert = lambda raw: ns['manual_points_to_bits'](points_to_bits, raw)
for raw in (b'', b'x', bytes(7), bytes(9)):
    try:
        convert(raw)
    except ProtocolError:
        pass
    else:
        raise AssertionError('bad packed rows accepted')
    assert 'numpy' not in sys.modules
assert convert(bytes(8))[0].tolist() == [0] * 64
assert 'numpy' not in sys.modules
try:
    ns['ManualByteArray'](bytes(8)).reshape(-1, 4)
except AssertionError:
    pass
else:
    raise AssertionError('wrong stride accepted')
assert trap.violations == 0
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(ENTRY)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


def metadata_argv(tmp_path: Path) -> list[str]:
    """Real proposal constants; paths are explicit local file-only fixtures."""
    import runpy
    source = runpy.run_path(str(ROOT / 'tools/speck-study/source.py'))
    manifest = tmp_path / 'main.json'
    source['freeze_all'](manifest, ROOT / 'src')
    weights = tmp_path / 'net8_small.h5'
    weights.write_bytes(b'METADATA FIXTURE, NEVER READ OR LOAD H5')
    return ['run', '--config', str(ROOT / 'configs/speck-study-paper.json'),
        '--policy', str(ROOT / 'configs/speck-study-policy.json'),
        '--root', str(tmp_path / 'FORMAL-ROOT-MUST-NOT-EXIST'), '--id', 'speck-final-001',
        '--backend', 'speck-study', '--teacher-python', str(Path(sys.executable).resolve()),
        '--teacher-weights', str(weights),
        '--weights-revision', '96344da116ff624be30f503038b22f26c7f7f693',
        '--weights-sha256', '15c58fff86855e882be1f977e42006761eeb5b9af28739d7a7fc8f44834902c0',
        '--gpu-uuid', 'GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5',
        '--gpu-pci', '00000000:AF:00.0', '--source-manifest', str(manifest)]


@pytest.mark.parametrize('pci', ['00000000:AF:00.0', '0000:AF:00.0'])
def test_portable_launch_metadata_full_frozen_cli(pci: str, tmp_path: Path) -> None:
    argv = metadata_argv(tmp_path)
    argv[argv.index('--gpu-pci') + 1] = pci
    owned = tmp_path / 'metadata'
    result = subprocess.run([sys.executable, '-B', str(ENTRY), '--mode', 'launch-metadata',
        '--rootowned', str(owned), '--', *argv], cwd=ROOT,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    saved = json.loads(result.stdout)
    assert saved['parse'] is True and saved['gpu_pci'] == '0000:af:00.0'
    assert saved['gpu_uuid'] == 'GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5'
    assert saved['actual_full_argv'] == argv
    canonical = saved['canonical_full_argv']
    assert canonical[canonical.index('--gpu-pci') + 1] == '0000:af:00.0'
    assert saved['guards']['violations'] == 0
    assert not saved['claim'] and not saved['native'] and not saved['origin']
    assert all(count == 0 for count in saved['counts'].values())
    assert saved['sourceonly'] is True
    assert not (tmp_path / 'FORMAL-ROOT-MUST-NOT-EXIST').exists()
    assert not list(tmp_path.rglob('*.pyc'))
    assert not list((ROOT / 'src').rglob('*.pyc'))


@pytest.mark.parametrize('flag,value', [
    ('--gpu-pci', '00010000:AF:00.0'), ('--gpu-pci', '10000000:AF:00.0'),
    ('--gpu-pci', '0x000000:AF:00.0'), ('--gpu-pci', '00000000:AF:00.8'),
    ('--gpu-pci', ''), ('--teacher-python', '/missing/python'),
    ('--teacher-weights', '/missing/weights.h5'), ('--backend', 'demo'),
    ('--weights-revision', 'main'), ('--gpu-uuid', 'GPU-1'),
    ('--UNRECOGNIZED', 'value'),
])
def test_portable_launch_metadata_negative_cli(
    flag: str, value: str, tmp_path: Path,
) -> None:
    argv = metadata_argv(tmp_path)
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv.extend([flag, value])
    owned = tmp_path / 'metadata-negative'
    result = subprocess.run([sys.executable, '-B', str(ENTRY), '--mode', 'launch-metadata',
        '--rootowned', str(owned), '--', *argv], cwd=ROOT,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 2, result.stdout + result.stderr
    saved = json.loads(result.stdout)
    assert saved['parse'] is False and saved['exit_code'] == 2
    assert saved['guards']['violations'] == 0
    assert not saved['claim'] and not saved['native'] and not saved['origin']
    assert not (tmp_path / 'FORMAL-ROOT-MUST-NOT-EXIST').exists()
    assert not list(tmp_path.rglob('*.pyc'))


@pytest.mark.parametrize('case', ['check', 'no-manifest', 'missing-manifest', 'wrong-manifest',
                                  'windows-dispatch', 'py39-adapter'])
def test_metadata_boundary_and_test_only_compatibility(case: str, tmp_path: Path) -> None:
    argv = metadata_argv(tmp_path)
    if case == 'check':
        argv[0] = 'check'
    elif case == 'no-manifest':
        del argv[-2:]
    elif case == 'missing-manifest':
        argv[-1] = str(tmp_path / 'missing.json')
    elif case == 'wrong-manifest':
        Path(argv[-1]).write_text('{}')
    # Simulate Windows dispatch without native import/probe. Also exercise the
    # 3.9 test-only decorator adapter on this interpreter; NOT a 3.9 runtime test.
    script = '''
import runpy, sys
ns = runpy.run_path(sys.argv[1], run_name='metadata_boundary_test')
if sys.argv[2] == 'windows-dispatch':
    import builtins
    from types import SimpleNamespace
    original = builtins.__import__
    def windows_cli_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'os' and globals and globals.get('__name__') == 'cipher_lab.cli':
            return SimpleNamespace(name='nt', environ=ns['os'].environ)
        return original(name, globals, locals, fromlist, level)
    builtins.__import__ = windows_cli_import
elif sys.argv[2] == 'py39-adapter':
    sys.version_info = (3, 9, 13)
# Path must be constructed before synthetic os.name changes.
root = ns['Path'](sys.argv[3]); root.mkdir()
raise SystemExit(ns['launch_metadata'](root, sys.argv[4:]))
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(ENTRY), case,
        str(tmp_path / 'owned'), *argv], cwd=ROOT,
        capture_output=True, text=True, timeout=15)
    expected = 0 if case in ('windows-dispatch', 'py39-adapter') else 2
    assert result.returncode == expected, result.stdout + result.stderr
    saved = json.loads(result.stdout)
    assert saved['parse'] is (expected == 0)
    assert saved['py39_test_only_slots_adapter'] is (case == 'py39-adapter')
    assert saved['guards']['violations'] == 0
    assert not (tmp_path / 'FORMAL-ROOT-MUST-NOT-EXIST').exists()

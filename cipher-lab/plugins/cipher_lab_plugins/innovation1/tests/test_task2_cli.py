"""Clean-source disposable children install their own guard before product imports."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_POP = subprocess.Popen
PROJECT = Path(__file__).resolve().parents[4]


def child(mode: str, root: Path) -> tuple[int, str, str]:
    # Exact test-created source copy. Parent/cwd neither points at production artifacts.
    project = root / 'source'
    project.mkdir()
    for name in ('src', 'plugins', 'configs'):
        shutil.copytree(PROJECT / name, project / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy(PROJECT / 'tests/checkpoint_child_guard.py', project)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(
        str(project / n) for n in ('src', 'plugins', '.',
                                  'plugins/cipher_lab_plugins/innovation1/tests')))
    script = 'from task2_child import main; main()'
    process = _POP([sys.executable, '-c', script, mode, str(root / 'outputs')],
                   cwd=project, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate(timeout=30)
        return process.returncode, stdout, stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        shutil.rmtree(project)  # only this invocation's exact test-owned source path


@pytest.mark.parametrize('mode', ['help', 'check', 'demo', 'isolated-evaluate'])
def test_guarded_clean_source_cli(tmp_path: Path, mode: str) -> None:
    code, out, err = child(mode, tmp_path)
    assert code == 0, (out, err)
    assert not (tmp_path / 'source').exists()
    assert 'CHILD_GUARD_OK' in out
    if mode in ('demo', 'isolated-evaluate'):
        assert 'MANUAL_CLOSED_LOOP_OK' in out
        print(out.strip())
    else:
        assert not (tmp_path / 'outputs').exists()


def test_cli_invalid_and_explicit_backend_device(tmp_path: Path, capsys: Any) -> None:
    from cipher_lab_plugins.innovation1.cli import main
    assert main(['run', '--config', 'configs/innovation1-fresh.json', '--root', str(tmp_path),
                 '--id', 'bad', '--backend', 'ciphers', '--device', 'cpu']) == 2
    assert not (tmp_path / 'data').exists()
    assert main(['check', '--config', str(tmp_path / 'missing')]) == 2
    assert 'invalid' in capsys.readouterr().err


def test_production_rejects_manual_dependency_identity(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.tests.test_task2 import setup
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    m.backend = 'ciphers'
    with pytest.raises(ValueError, match='PRODUCTION_DEPENDENCY_MISMATCH'):
        run(r, p, m, s, resources, backend='ciphers')
    assert not p.train_calls and not p.test_calls


def test_application_missing_telemetry_stops_without_science(tmp_path: Path,
                                                           monkeypatch: Any) -> None:
    from cipher_lab_plugins.innovation1 import application
    from cipher_lab_plugins.innovation1.resources import ResourceStop
    from cipher_lab_plugins.innovation1.tests.test_task2 import owner, small

    from cipher_lab.artifacts import read_saved
    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise ResourceStop()
    monkeypatch.setattr(application, 'production_resources', unavailable)
    r = small()
    s = owner(tmp_path, r)
    application.execute(r, s, backend='ciphers', device='cuda:0',
                        gpu_uuid='GPU-manual', gpu_pci='0000:01:00.0')
    saved: Any = read_saved(tmp_path, 'manual')
    assert saved['status'] == 'STOPPED'
    assert saved['summary']['reason'] == 'GPU_UNVERIFIED'
    assert not saved['summary']['synthetic']
    assert saved['counts'].get('completed_optimizer_updates', 0) == 0

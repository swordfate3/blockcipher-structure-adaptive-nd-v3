"""Clean-source disposable children install their own guard before product imports."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

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
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
               PYTHONPATH=os.pathsep.join(
        str(project / n) for n in ('src', 'plugins', '.',
                                  'plugins/cipher_lab_plugins/innovation1/tests')))
    script = 'from task2_child import main; main()'
    process = _POP([sys.executable, '-c', script, mode],
                   cwd=project, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate(timeout=30)
        return process.returncode, stdout, stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        shutil.rmtree(project)  # only this invocation's exact test-owned source path


@pytest.mark.parametrize('mode', ['help', 'innovation1-fresh',
                                  'innovation1-skinny-layout-early-stop'])
def test_guarded_clean_source_cli(tmp_path: Path, mode: str) -> None:
    code, out, err = child(mode, tmp_path)
    assert code == 0, (out, err)
    assert 'CHILD_GUARD_OK' in out
    assert not (tmp_path / 'source').exists()
    assert not (tmp_path / 'outputs').exists()


def test_production_rejects_manual_dependency_identity(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.tests.test_task2 import setup
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    m.backend = 'ciphers'
    with pytest.raises(ValueError, match='PRODUCTION_DEPENDENCY_MISMATCH'):
        run(r, p, m, s, resources, backend='ciphers')
    assert not p.train_calls and not p.test_calls

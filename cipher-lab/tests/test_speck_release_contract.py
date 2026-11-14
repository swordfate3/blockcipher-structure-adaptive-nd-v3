"""Candidate-only HEAD contracts, guards live in the child before imports."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT/'tests/speck_release_child.py'


@pytest.mark.parametrize('mode', ['integral', 'prediction', 'shared-m'])
def test_head_assembly_contract_without_science(mode: str) -> None:
    result = subprocess.run([sys.executable, '-B', str(DRIVER), mode], cwd=ROOT,
        env={**os.environ, 'PYTHONPATH': str(ROOT/'src')},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout+result.stderr
    saved = json.loads(result.stdout)
    assert saved['collector_calls' if mode == 'integral' else 'training_calls'] == 1


def test_no_unpublished_experiment_parsers(tmp_path: Path) -> None:
    from cipher_lab.config import load_experiment
    for method in ('bit-fourier-search-v1', 'bit-public-feature-frontier-v1',
                   'bit-public-feature-probe-v1', 'bit-residual-paired-v1',
                   'integral-search-v1'):
        path = tmp_path/'recipe.json'
        path.write_text(json.dumps({'method': method}))
        with pytest.raises(ValueError, match='unsupported experiment'):
            load_experiment(path)


def test_no_old_feature_dependencies_in_frozen_closure() -> None:
    import importlib.util
    entry = ROOT/'tools/speck-study/source.py'
    spec = importlib.util.spec_from_file_location('release_source', entry)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = {'cipher_lab/spectral_config.py', 'cipher_lab/representations/__init__.py',
             'cipher_lab/representations/simon_anf.py',
             'cipher_lab/representations/simon_frontier.py'}
    assert not names.intersection(module.CLOSURE)

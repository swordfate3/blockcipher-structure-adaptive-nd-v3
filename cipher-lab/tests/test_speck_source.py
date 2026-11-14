"""Source bootstrap checks compile bytes, never project bytecode."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def tool() -> Any:
    path = ROOT/'tools/speck-study/source.py'
    spec = importlib.util.spec_from_file_location('task5_source',path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freeze_main_closure_missing_and_changed_source_rejected(tmp_path: Path) -> None:
    module = tool()
    source = ROOT/'src'
    manifest = module.snapshot(source)
    assert 'cipher_lab/cli.py' in manifest
    assert 'cipher_lab/speck_study/application.py' in manifest
    assert 'cipher_lab/training/application.py' not in manifest
    frozen = module.checked(source,manifest)
    assert set(frozen) == set(module.CLOSURE)
    manifest['cipher_lab/cli.py'] = '0'*64
    with pytest.raises(ValueError):
        module.checked(source,manifest)


def test_source_only_loader_cannot_load_outside_frozen_closure() -> None:
    module = tool()
    loader = module.SourceOnly(ROOT/'src',module.checked(ROOT/'src',module.snapshot(ROOT/'src')))
    with pytest.raises(ImportError):
        loader.find_spec('cipher_lab.training.application')


def test_source_manifest_cannot_be_subset_self_approval_or_bytecode(tmp_path: Path) -> None:
    module = tool()
    with pytest.raises(ValueError):
        module.checked(tmp_path,{'cipher_lab/cli.py':'0'*64})
    manifest = module.snapshot(ROOT/'src')
    assert json.loads(json.dumps(manifest)) == manifest


def test_freeze_writes_separate_provider_bootstrap_and_package_manifests(tmp_path: Path) -> None:
    module = tool()
    destination = tmp_path/'main.json'
    assert module.main(['freeze',str(destination),str(ROOT/'src')]) == 0
    provider = json.loads(Path(str(destination)+'.provider.json').read_text())
    boot = json.loads(Path(str(destination)+'.bootstrap.json').read_text())
    package = json.loads(Path(str(destination)+'.package.json').read_text())
    assert set(boot) == {'source.py','freeze.py','supervise.py'}
    assert 'cipher_lab/speck_study/gpu_telemetry.py' in provider
    assert 'cipher_lab/platform_resources/dxgi.py' in provider
    assert 'cipher_lab/speck_study/teacher_network.py' in module.CLOSURE
    assert 'cipher_lab/speck_study/native_parent.py' in module.CLOSURE
    assert set(module.CLOSURE) <= set(package)
    assert 'cipher_lab/training/application.py' in package


def test_duplicate_manifest_keys_not_hidden_by_json_last_wins(tmp_path: Path) -> None:
    module = tool()
    path = tmp_path/'bad.json'
    path.write_text('{"x":"first","x":"second"}')
    with pytest.raises(ValueError):
        module.read_manifest(path)

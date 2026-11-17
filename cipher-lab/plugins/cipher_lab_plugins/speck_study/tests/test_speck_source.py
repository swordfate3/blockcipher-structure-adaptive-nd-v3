"""Source bootstrap checks compile bytes, never project bytecode."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from cipher_lab_plugins.speck_study.tests.clean_source import clean_source
from cipher_lab_plugins.speck_study.tests.locations import ROOT, RUNTIME


def tool() -> Any:
    path = RUNTIME/'source.py'
    spec = importlib.util.spec_from_file_location('task5_source',path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freeze_main_closure_missing_and_changed_source_rejected(tmp_path: Path) -> None:
    module = tool()
    source = ROOT/'src'
    manifest = module.snapshot(source,ROOT/'plugins')
    assert 'cipher_lab/cli.py' in manifest
    assert 'cipher_lab_plugins/speck_study/application.py' in manifest
    assert 'cipher_lab/training/application.py' not in manifest
    frozen = module.checked(source,ROOT/'plugins',manifest)
    assert set(frozen) == set(module.CLOSURE)
    manifest['cipher_lab/cli.py'] = '0'*64
    with pytest.raises(ValueError):
        module.checked(source,ROOT/'plugins',manifest)


def test_source_only_loader_cannot_load_outside_frozen_closure() -> None:
    module = tool()
    loader = module.SourceOnly(ROOT/'src',ROOT/'plugins',module.checked(ROOT/'src',ROOT/'plugins',module.snapshot(ROOT/'src',ROOT/'plugins')))  # noqa: E501 - explicit migrated source/argv identity
    with pytest.raises(ImportError):
        loader.find_spec('cipher_lab.training.application')


def test_source_manifest_cannot_be_subset_self_approval_or_bytecode(tmp_path: Path) -> None:
    module = tool()
    with pytest.raises(ValueError):
        module.checked(tmp_path,tmp_path,{'cipher_lab/cli.py':'0'*64})
    manifest = module.snapshot(ROOT/'src',ROOT/'plugins')
    assert json.loads(json.dumps(manifest)) == manifest


def test_freeze_writes_separate_provider_bootstrap_and_package_manifests(tmp_path: Path) -> None:
    module = tool()
    clean = clean_source(tmp_path/'clean')
    destination = tmp_path/'main.json'
    assert module.main(['freeze',str(destination),str(clean/'src'),str(clean/'plugins')]) == 0
    provider = json.loads(Path(str(destination)+'.provider.json').read_text())
    boot = json.loads(Path(str(destination)+'.bootstrap.json').read_text())
    package = json.loads(Path(str(destination)+'.package.json').read_text())
    assert set(boot) == {'source.py','freeze.py','supervise.py'}
    assert 'cipher_lab_plugins/speck_study/gpu_telemetry.py' in provider
    assert 'cipher_lab_plugins/speck_study/platform_resources/dxgi.py' in provider
    assert 'cipher_lab_plugins/speck_study/teacher_network.py' in module.CLOSURE
    assert 'cipher_lab_plugins/speck_study/native_parent.py' in module.CLOSURE
    assert set(module.CLOSURE) <= set(package)
    assert 'cipher_lab/training/application.py' in package


def test_clean_positive_root_filters_only_test_owned_copy_keeps_cache_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    from cipher_lab_plugins.speck_study.tests.clean_source import clean_source

    helper = importlib.import_module(clean_source.__module__)
    original = clean_source(tmp_path/'original')
    cache = original/'src/cipher_lab/__pycache__/fixture.pyc'
    cache.parent.mkdir()
    cache.write_bytes(b'ARTIFICIAL CACHE NEVER EXECUTED')
    before = cache.read_bytes()
    module = tool()
    with pytest.raises(ValueError, match='not source-only'):
        module.package_snapshot(original/'src', original/'plugins')
    monkeypatch.setattr(helper, 'ROOT', original)
    monkeypatch.setattr(helper, 'PLUGIN', original/'plugins/cipher_lab_plugins/speck_study')
    copied = clean_source(tmp_path/'copy')
    assert not list(copied.rglob('__pycache__'))
    assert not list(copied.rglob('*.pyc'))
    assert not list(copied.rglob('tests'))
    assert cache.read_bytes() == before  # original is never cleaned
    identity = module.package_snapshot(copied/'src', copied/'plugins')
    assert 'cipher_lab/cli.py' in identity
    assert 'cipher_lab_plugins/speck_study/codec.py' in identity


def test_duplicate_manifest_keys_not_hidden_by_json_last_wins(tmp_path: Path) -> None:
    module = tool()
    path = tmp_path/'bad.json'
    path.write_text('{"x":"first","x":"second"}')
    with pytest.raises(ValueError):
        module.read_manifest(path)

"""Namespace-init cache boundary; handwritten sources/bytes, no science."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from cipher_lab_plugins.speck_study.resources import verify_sources
from cipher_lab_plugins.speck_study.tests.test_plugin_roots import tree, wrapper


@pytest.mark.parametrize('name', [
    '__init__.pyc', '__init__.pyo',
    '__pycache__/__init__.cpython-310.pyc',
    '__pycache__/__init__.cpython-310.opt-1.pyc',
    '__pycache__/__init__.cpython-39.pyo',
])
@pytest.mark.parametrize('linked', [False, True])
def test_namespace_init_cache_rejected(tmp_path: Path, name: str, linked: bool) -> None:
    module = wrapper()
    core, plugins, manifest = tree(tmp_path, module.CLOSURE)
    cache = plugins / 'cipher_lab_plugins' / name
    cache.parent.mkdir(exist_ok=True)
    if linked:
        # A dangling link must not look like an absent optional cache.
        cache.symlink_to(tmp_path / 'absent-bytecode')
    else:
        cache.write_bytes(b'handwritten fake bytecode')
    with pytest.raises(ValueError, match='source-only|symlink'):
        module.package_snapshot(core, plugins)
    with pytest.raises(ValueError, match='source-only|symlink'):
        verify_sources(core, plugins, manifest)


@pytest.mark.parametrize('dangling', [False, True])
def test_namespace_cache_directory_symlink_rejected(tmp_path: Path, dangling: bool) -> None:
    module = wrapper()
    core, plugins, manifest = tree(tmp_path, module.CLOSURE)
    outside = tmp_path / 'outside'
    if not dangling:
        outside.mkdir()
    (plugins / 'cipher_lab_plugins/__pycache__').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        module.package_snapshot(core, plugins)
    with pytest.raises(ValueError, match='symlink'):
        verify_sources(core, plugins, manifest)


@pytest.mark.parametrize('error', [PermissionError, OSError])
def test_namespace_cache_enumeration_error_fails_closed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[OSError]) -> None:
    module = wrapper()
    core, plugins, manifest = tree(tmp_path, module.CLOSURE)
    cache = plugins / 'cipher_lab_plugins/__pycache__'
    cache.mkdir()
    original = Path.iterdir

    def iterdir(path: Path) -> Iterator[Path]:
        if path == cache:
            raise error('artificial enumeration failure')
        return original(path)

    monkeypatch.setattr(Path, 'iterdir', iterdir)
    with pytest.raises(error, match='artificial enumeration failure'):
        module.package_snapshot(core, plugins)
    with pytest.raises(error, match='artificial enumeration failure'):
        verify_sources(core, plugins, manifest)


def test_unrelated_namespace_and_sibling_caches_do_not_affect_identity(tmp_path: Path) -> None:
    module = wrapper()
    core, plugins, manifest = tree(tmp_path, module.CLOSURE)
    before = module.package_snapshot(core, plugins)
    namespace = plugins / 'cipher_lab_plugins'
    cache = namespace / '__pycache__'
    cache.mkdir()
    (cache / 'other.cpython-310.pyc').write_bytes(b'unrelated module cache')
    (namespace / 'other.pyc').write_bytes(b'unrelated legacy cache')
    sibling = namespace / 'unrelated/__pycache__'
    sibling.mkdir(parents=True)
    (sibling / '__init__.cpython-310.pyc').write_bytes(b'sibling init cache')
    (namespace / 'unrelated/linked').symlink_to(tmp_path / 'absent')
    assert module.package_snapshot(core, plugins) == before
    verify_sources(core, plugins, manifest)

"""Regression trees contain only handwritten bytes, never scientific execution."""
from __future__ import annotations

from pathlib import Path

import pytest

from cipher_lab_plugins.speck_study import paths, sources
from cipher_lab_plugins.speck_study.resources import verify_sources
from cipher_lab_plugins.speck_study.tests.test_plugin_roots import tree, wrapper


@pytest.mark.parametrize('owner', ['core', 'plugin'])
@pytest.mark.parametrize('level', ['file', 'root', 'ancestor'])
def test_supplier_rejects_lexical_import_symlinks_before_resolve(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: str, level: str) -> None:
    import cipher_lab

    module = wrapper()
    core, plugins, _ = tree(tmp_path / 'real', module.CLOSURE)
    core_file = core / 'cipher_lab/__init__.py'
    plugin_file = plugins / 'cipher_lab_plugins/speck_study/paths.py'
    original = core_file if owner == 'core' else plugin_file
    target = original if level == 'file' else (
        (core if owner == 'core' else plugins) if level == 'root' else tmp_path / 'real')
    link = tmp_path / 'import-link'
    link.symlink_to(target, target_is_directory=level != 'file')
    lexical = link if level == 'file' else link / original.relative_to(target)
    monkeypatch.setattr(cipher_lab, '__file__', str(lexical if owner == 'core' else core_file))
    monkeypatch.setattr(paths, '__file__', str(lexical if owner == 'plugin' else plugin_file))
    with pytest.raises(ValueError, match='symlink'):
        sources.source_identity()


def test_supplier_accepts_plain_absolute_import_paths(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cipher_lab

    module = wrapper()
    core, plugins, _ = tree(tmp_path, module.CLOSURE)
    monkeypatch.setattr(cipher_lab, '__file__', str(core / 'cipher_lab/__init__.py'))
    monkeypatch.setattr(paths, '__file__', str(plugins / 'cipher_lab_plugins/speck_study/paths.py'))
    assert paths.source_roots() == (core, plugins)
    assert sources.source_identity() == module.package_snapshot(core, plugins)


@pytest.mark.parametrize('location', ['unrelated', 'tests', 'nested_tests'])
@pytest.mark.parametrize('artifact', ['cache', 'pyc', 'pyo', 'file_link', 'dir_link'])
def test_unmanaged_artifacts_do_not_affect_identity_or_verify(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str, artifact: str) -> None:
    module = wrapper()
    core, plugins, manifest = tree(tmp_path, module.CLOSURE)
    monkeypatch.setattr(sources, 'source_roots', lambda: (core, plugins))
    before = sources.source_identity()
    own = plugins / 'cipher_lab_plugins/speck_study'
    base = {'unrelated': plugins / 'cipher_lab_plugins/unrelated',
            'tests': own / 'tests', 'nested_tests': own / 'runtime/tests'}[location]
    base.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'hidden.py').write_bytes(b'# artificial source\n')
    if artifact == 'dir_link':
        # Excluded trees must be pruned before even examining the linked entry.
        base.symlink_to(outside, target_is_directory=True)
    else:
        base.mkdir()
        if artifact == 'cache':
            (base / '__pycache__').mkdir()
            (base / '__pycache__/fake.pyc').write_bytes(b'fake bytecode')
        elif artifact in ('pyc', 'pyo'):
            (base / ('fake.' + artifact)).write_bytes(b'fake bytecode')
        else:
            (base / 'linked.py').symlink_to(outside / 'hidden.py')
    assert sources.source_identity() == before
    verify_sources(core, plugins, manifest)


@pytest.mark.parametrize('location', ['core', 'production', 'runtime'])
@pytest.mark.parametrize('artifact', ['cache', 'pyc', 'pyo', 'file_link', 'dir_link'])
def test_managed_artifacts_still_rejected(
        tmp_path: Path, location: str, artifact: str) -> None:
    module = wrapper()
    core, plugins, manifest = tree(tmp_path, module.CLOSURE)
    own = plugins / 'cipher_lab_plugins/speck_study'
    base = {'core': core / 'cipher_lab', 'production': own, 'runtime': own / 'runtime'}[location]
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'hidden.py').write_bytes(b'# artificial source\n')
    if artifact == 'cache':
        (base / '__pycache__').mkdir()
    elif artifact in ('pyc', 'pyo'):
        (base / ('fake.' + artifact)).write_bytes(b'fake bytecode')
    else:
        (base / ('linked.py' if artifact == 'file_link' else 'linked')).symlink_to(
            outside / 'hidden.py' if artifact == 'file_link' else outside,
            target_is_directory=artifact == 'dir_link')
    with pytest.raises(ValueError, match='symlink|source-only'):
        module.package_snapshot(core, plugins)
    with pytest.raises(ValueError, match='symlink|source-only'):
        verify_sources(core, plugins, manifest)


@pytest.mark.parametrize('owner', ['core', 'plugin'])
@pytest.mark.parametrize('level', ['root', 'ancestor'])
def test_verify_rejects_managed_lexical_root_symlinks(
        tmp_path: Path, owner: str, level: str) -> None:
    module = wrapper()
    core, plugins, manifest = tree(tmp_path / 'real', module.CLOSURE)
    original = core if owner == 'core' else plugins
    target = original if level == 'root' else tmp_path / 'real'
    link = tmp_path / 'import-link'
    link.symlink_to(target, target_is_directory=True)
    lexical = link / original.relative_to(target)
    with pytest.raises(ValueError, match='symlink|source-only'):
        verify_sources(lexical if owner == 'core' else core,
                       lexical if owner == 'plugin' else plugins, manifest)


@pytest.mark.parametrize('check', ['identity', 'verify'])
def test_namespace_initializer_symlink_rejected(tmp_path: Path, check: str) -> None:
    module = wrapper()
    core, plugins, manifest = tree(tmp_path, module.CLOSURE)
    path = plugins / 'cipher_lab_plugins/__init__.py'
    outside = tmp_path / 'same.py'
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink|source-only'):
        if check == 'identity':
            module.package_snapshot(core, plugins)
        else:
            verify_sources(core, plugins, manifest)


def test_verify_still_rejects_changed_source(tmp_path: Path) -> None:
    module = wrapper()
    core, plugins, manifest = tree(tmp_path, module.CLOSURE)
    (plugins / 'cipher_lab_plugins/speck_study/codec.py').write_bytes(b'# changed\n')
    with pytest.raises(ValueError, match='source identity mismatch'):
        verify_sources(core, plugins, manifest)

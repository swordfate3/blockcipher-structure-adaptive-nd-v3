"""Artificial filesystem-only checks of the explicit prediction source closure."""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from cipher_lab_plugins.prediction_common import source

from cipher_lab import artifacts
from cipher_lab.artifacts import ArtifactError

OWNERS = ('prediction_common', 'residual', 'feature_probe', 'frontier')


@pytest.fixture
def source_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / 'project' / 'plugins' / 'cipher_lab_plugins'
    root.mkdir(parents=True)
    (root / '__init__.py').write_text('# namespace\n')
    for owner in OWNERS:
        package = root / owner
        package.mkdir()
        (package / '__init__.py').write_text('# owner\n')
        (package / 'source.py').write_text('# artificial module\n')
    monkeypatch.setattr(source, '__file__', str(root / 'prediction_common' / 'source.py'))
    monkeypatch.setattr(artifacts, 'source_identity', lambda: {'cipher_lab/fake.py': 'a' * 64})
    return root


def test_source_closure_is_explicit_and_fresh(source_tree: Path) -> None:
    root = source_tree
    unrelated = root / 'unrelated'
    unrelated.mkdir()
    (unrelated / 'module.py').write_text('# not a dependency\n')
    for owner in OWNERS:
        tests = root / owner / 'tests'
        tests.mkdir()
        (tests / 'test_fake.py').write_text('# not production\n')
    expected = {'cipher_lab/fake.py', 'cipher_lab_plugins/__init__.py'}
    expected.update(f'cipher_lab_plugins/{owner}/{name}' for owner in OWNERS
                    for name in ('__init__.py', 'source.py'))
    before = source.source_identity()
    assert set(before) == expected
    changed = root / 'residual' / 'source.py'
    changed.write_bytes(b'# changed\n')
    after = source.source_identity()
    key = 'cipher_lab_plugins/residual/source.py'
    assert after[key] == hashlib.sha256(b'# changed\n').hexdigest()
    assert before[key] != after[key]
    before.pop(key)
    after.pop(key)
    assert before == after


@pytest.mark.parametrize('level', ['root', 'parent', 'ancestor', 'owner', 'file'])
def test_source_rejects_lexical_symlink_ancestry(source_tree: Path,
        monkeypatch: pytest.MonkeyPatch, level: str) -> None:
    root = source_tree
    selected = {'root': root, 'parent': root.parent, 'ancestor': root.parent.parent,
                'owner': root / 'prediction_common',
                'file': root / 'prediction_common' / 'source.py'}[level]
    target = selected.with_name(selected.name + '-target')
    selected.rename(target)
    selected.symlink_to(target, target_is_directory=target.is_dir())
    # Keep __file__ lexical: resolving it before validation is the original bug.
    monkeypatch.setattr(source, '__file__', str(root / 'prediction_common' / 'source.py'))
    with pytest.raises(ArtifactError):
        source.source_identity()


@pytest.mark.parametrize('owner', OWNERS)
@pytest.mark.parametrize('missing', ['directory', 'init'])
def test_source_rejects_missing_required_owner(source_tree: Path,
        owner: str, missing: str) -> None:
    package = source_tree / owner
    if missing == 'directory':
        shutil.rmtree(package)
    else:
        (package / '__init__.py').unlink()
    with pytest.raises(ArtifactError):
        source.source_identity()


def test_source_rejects_missing_namespace_init(source_tree: Path) -> None:
    (source_tree / '__init__.py').unlink()
    with pytest.raises(ArtifactError):
        source.source_identity()


@pytest.mark.parametrize('owner', OWNERS)
def test_source_rejects_scan_errors_without_partial_identity(source_tree: Path,
        monkeypatch: pytest.MonkeyPatch, owner: str) -> None:
    blocked = source_tree / owner / 'nested'
    blocked.mkdir()
    (blocked / 'module.py').write_text('# must not silently disappear\n')
    original = os.scandir

    def fail_scan(path: Any) -> Any:
        if Path(path) == blocked:
            raise PermissionError('artificial enumeration failure')
        return original(path)

    monkeypatch.setattr(os, 'scandir', fail_scan)
    with pytest.raises(ArtifactError):
        source.source_identity()


@pytest.mark.parametrize('kind', ['directory', 'file', 'dangling-directory'])
def test_source_rejects_managed_symlinks(source_tree: Path, kind: str) -> None:
    package = source_tree / 'residual'
    if kind == 'file':
        (package / 'linked.py').symlink_to(package / 'source.py')
    else:
        target = source_tree.parent / 'external'
        if kind == 'directory':
            target.mkdir()
            (target / 'module.py').write_text('# must not follow\n')
        (package / 'linked').symlink_to(target, target_is_directory=True)
    with pytest.raises(ArtifactError):
        source.source_identity()


@pytest.mark.parametrize('owner', OWNERS)
@pytest.mark.parametrize('dangling', [False, True])
def test_source_ignores_test_symlinks(source_tree: Path, owner: str, dangling: bool) -> None:
    before = source.source_identity()
    target = source_tree.parent / 'excluded-tests'
    if not dangling:
        target.mkdir()
        (target / 'test_fake.py').write_text('# excluded\n')
    (source_tree / owner / 'tests').symlink_to(target, target_is_directory=True)
    assert source.source_identity() == before


def test_source_rejects_artificial_read_error(source_tree: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_bytes

    def fail_read(path: Path) -> bytes:
        if path == source_tree / 'frontier' / 'source.py':
            raise OSError('artificial read failure')
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', fail_read)
    with pytest.raises(ArtifactError):
        source.source_identity()

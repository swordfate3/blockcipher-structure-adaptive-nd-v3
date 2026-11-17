"""Two-root freeze tests: only hand-written source bytes and fake children."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'


def wrapper(name: str = 'source') -> Any:
    path = RUNTIME / (name + '.py')
    assert path.is_file(), 'SPECK plugin runtime has not been migrated'
    spec = importlib.util.spec_from_file_location('fake_tree_' + name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree(tmp_path: Path, closure: tuple[str, ...]) -> tuple[Path, Path, dict[str, str]]:
    core, plugins = tmp_path / 'src', tmp_path / 'plugins'
    manifest = {}
    for name in closure:
        root = core if name.startswith('cipher_lab/') else plugins
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = b'# hand-written fake source\n'
        path.write_bytes(raw)
        manifest[name] = hashlib.sha256(raw).hexdigest()
    return core, plugins, manifest


@pytest.mark.parametrize('kind', ['source', 'freeze'])
def test_two_roots_exact_missing_extra_sha_symlink(tmp_path: Path, kind: str) -> None:
    module = wrapper(kind)
    core, plugins, manifest = tree(tmp_path, module.CLOSURE)
    check = module.checked if kind == 'source' else module.checked_sources
    assert set(check(core, plugins, manifest)) == set(module.CLOSURE)
    assert 'cipher_lab_plugins/__init__.py' in manifest
    assert 'cipher_lab_plugins/speck_study/__init__.py' in manifest
    assert len(module.CLOSURE) == len(set(module.CLOSURE))
    for bad in ({}, dict(manifest, extra='0'*64),
                dict(manifest, **{'cipher_lab/__init__.py': '0'*64})):
        with pytest.raises(ValueError):
            check(core, plugins, bad)
    path = plugins / 'cipher_lab_plugins/speck_study/__init__.py'
    path.unlink()
    with pytest.raises(FileNotFoundError):
        check(core, plugins, manifest)
    target = tmp_path / 'target.py'
    target.write_bytes(b'# hand-written fake source\n')
    path.symlink_to(target)
    with pytest.raises(ValueError, match='symlink'):
        check(core, plugins, manifest)


@pytest.mark.parametrize('kind', ['source', 'freeze'])
def test_loader_blocks_both_namespaces_outside_closure(tmp_path: Path, kind: str) -> None:
    module = wrapper(kind)
    core, plugins, manifest = tree(tmp_path, module.CLOSURE)
    check = module.checked if kind == 'source' else module.checked_sources
    loader = module.SourceOnly(core, plugins, check(core, plugins, manifest))
    for name in ('cipher_lab.unapproved', 'cipher_lab_plugins.other.cli'):
        with pytest.raises(ImportError):
            loader.find_spec(name)
    assert loader.find_spec('cipher_lab_plugins.speck_study') is not None
    assert loader.find_spec('json') is None


@pytest.mark.parametrize('kind', ['source', 'freeze'])
def test_duplicate_manifest_rejected(tmp_path: Path, kind: str) -> None:
    module = wrapper(kind)
    path = tmp_path / 'manifest.json'
    path.write_text('{"x":"a","x":"b"}')
    with pytest.raises(ValueError):
        module.read_manifest(path)


@pytest.mark.parametrize('kind', ['source', 'freeze'])
@pytest.mark.parametrize('namespace', ['cipher_lab','cipher_lab.loaded',
                                       'cipher_lab_plugins','cipher_lab_plugins.loaded'])
def test_preimport_plugin_rejected_before_file_access(tmp_path: Path, kind: str, namespace: str,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    module = wrapper(kind)
    monkeypatch.setattr(module, 'sys', SimpleNamespace(modules={namespace: object()}))
    with pytest.raises(RuntimeError, match='imported before freeze'):
        if kind == 'source':
            module.main(['run', str(tmp_path/'absent'), str(tmp_path/'src'),
                         str(tmp_path/'plugins'), 'run', '--plugin',
                         'cipher_lab_plugins.speck_study.cli'])
        else:
            module.teacher(tmp_path/'absent', tmp_path/'src', tmp_path/'plugins', [])


def test_fixed_teacher_child_dual_roots(tmp_path: Path) -> None:
    module = wrapper('supervise')
    calls = []
    class Job:
        def spawn(self, argv: list[str], **kwargs: Any) -> int:
            calls.append(argv)
            return 17
    spawn = module.TrustedTeacherSpawn(Job(), tmp_path/'m', tmp_path/'src', tmp_path/'plugins')
    assert spawn(['py39', '-B', '-m', 'cipher_lab_plugins.speck_study.teacher_provider',
                  '--flag']) == 17
    assert calls[0][3:7] == ['--teacher', str(tmp_path/'m'), str(tmp_path/'src'),
                             str(tmp_path/'plugins')]
    with pytest.raises(ValueError):
        spawn(['py39', '-B', '-m', 'cipher_lab.speck_study.teacher_provider'])

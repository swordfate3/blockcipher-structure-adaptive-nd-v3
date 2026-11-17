"""Stdlib-only boundary probes: no model, cipher, entropy or claim."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())


def probe(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, '-c', script, *args], cwd=PROJECT,
        capture_output=True, text=True, check=False,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join((str(PROJECT / 'src'),
                 str(PROJECT / 'plugins'))), PYTHONDONTWRITEBYTECODE='1'))


@pytest.mark.parametrize('filename', ['fourier-r6.json', 'fourier-example.json'])
def test_plugin_check_is_stdlib_no_claim_or_writes(tmp_path: Path, filename: str) -> None:
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'schema': 2, 'wall_seconds': 60, 'cpu_seconds': 60,
        'deadline_utc': '2099-01-01T00:00:00Z', 'real_blocks': 4000, 'null_words': 4000,
        'rss_stop_bytes': 1000000, 'output_bytes': 4000000,
        'optimizer_updates': 0, 'forward_rows': 0}))
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    done = probe('''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'torch', 'psutil') or fullname.startswith((
            'cipher_lab.artifacts', 'cipher_lab.execution', 'cipher_lab.ciphers',
            'cipher_lab.data.', 'cipher_lab.models.', 'cipher_lab.training.',
            'cipher_lab_plugins.fourier.data', 'cipher_lab_plugins.fourier.method',
            'cipher_lab_plugins.fourier.search', 'cipher_lab_plugins.fourier.source')):
            raise AssertionError('check loaded science/storage: ' + fullname)
        return None
sys.meta_path.insert(0, Block())
from cipher_lab_plugins.fourier.cli import main
raise SystemExit(main(sys.argv[1:]))
''', 'check', '--config', str(PROJECT / 'configs' / filename), '--policy', str(policy))
    assert done.returncode == 0, done.stderr
    output = json.loads(done.stdout)
    assert output['recipe']['method'] == 'bit-fourier-search-v1'
    assert output['estimate']['planned_optimizer_updates'] == 0
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_config_loader_strict_json_and_no_core_extension_dispatch(tmp_path: Path) -> None:
    from cipher_lab_plugins.fourier.config import FourierRecipe, load_experiment

    path = tmp_path / 'recipe.json'
    recipe = FourierRecipe()
    path.write_text(json.dumps(recipe.to_dict()))
    assert load_experiment(path) == recipe
    for value in ('{"schema":1,"schema":1}', '{"tau":NaN}', '{}'):
        path.write_text(value)
        with pytest.raises(ValueError):
            load_experiment(path)


def test_source_scope_tracks_only_core_plugin_and_package_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.fourier import source

    core = tmp_path / 'src/cipher_lab'
    plugins = tmp_path / 'plugins/cipher_lab_plugins'
    fourier = plugins / 'fourier'
    other = plugins / 'other'
    for directory in (core, fourier, other):
        directory.mkdir(parents=True)
        (directory / '__init__.py').write_text('')
    (plugins / '__init__.py').write_text('')
    monkeypatch.setattr(source, '_source_roots', lambda: (core, fourier))
    first = source.source_identity()
    assert set(first) == {'cipher_lab/__init__.py', 'cipher_lab_plugins/__init__.py',
                          'cipher_lab_plugins/fourier/__init__.py'}
    (other / '__init__.py').write_text('UNRELATED')
    assert source.source_identity() == first
    (fourier / '__init__.py').write_text('CHANGED')
    assert source.source_identity() != first
    (plugins / '__init__.py').write_text('CHANGED ROOT')
    assert source.source_identity()['cipher_lab_plugins/__init__.py'] != first[
        'cipher_lab_plugins/__init__.py']


def test_source_rejects_missing_or_symlinked_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.fourier import source

    from cipher_lab.artifacts import ArtifactError

    core = tmp_path / 'src/cipher_lab'
    plugin = tmp_path / 'plugins/cipher_lab_plugins/fourier'
    for path in (core, plugin):
        path.mkdir(parents=True)
        (path / '__init__.py').write_text('')
    monkeypatch.setattr(source, '_source_roots', lambda: (core, plugin))
    with pytest.raises(ArtifactError, match='SOURCE_UNAVAILABLE'):
        source.source_identity()  # missing package init is not omitted
    (plugin.parent / '__init__.py').write_text('')
    outside = tmp_path / 'outside.py'
    outside.write_text('')
    (plugin / 'escape.py').symlink_to(outside)
    with pytest.raises(ArtifactError, match='SOURCE_UNAVAILABLE'):
        source.source_identity()


@pytest.fixture
def artificial_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """Only literal source files and deterministic technical claim names."""
    import uuid

    from cipher_lab_plugins.fourier import source

    from cipher_lab import artifacts

    core = tmp_path / 'src/cipher_lab'
    plugin = tmp_path / 'plugins/cipher_lab_plugins/fourier'
    for path in (core, plugin):
        path.mkdir(parents=True)
        (path / '__init__.py').write_text('')
    (plugin.parent / '__init__.py').write_text('')
    (plugin / 'production.py').write_text('VERSION = 1\n')
    monkeypatch.setattr(source, '_source_roots', lambda: (core, plugin))
    counter = iter(range(1, 1000))
    monkeypatch.setattr(artifacts, 'uuid4', lambda: uuid.UUID(int=next(counter)))
    return core, plugin


@pytest.mark.parametrize('test_tree', ['bytes', 'internal-link', 'internal-file-link',
                                        'directory-link', 'scan-denied'])
def test_source_excludes_tests_before_descent_and_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    artificial_source: tuple[Path, Path], test_tree: str,
) -> None:
    from cipher_lab_plugins.fourier import source

    from cipher_lab.artifacts import RunStore

    _, plugin = artificial_source
    initial = source.source_identity()
    tests = plugin / 'tests'
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'test_outside.py').write_text('TEST_ONLY = 1\n')
    if test_tree == 'directory-link':
        tests.symlink_to(outside, target_is_directory=True)
    else:
        tests.mkdir()
        (tests / 'test_example.py').write_text('TEST_ONLY = 1\n')
        if test_tree == 'internal-link':
            (tests / 'linked').symlink_to(outside, target_is_directory=True)
        if test_tree == 'internal-file-link':
            (tests / 'linked.py').symlink_to(outside / 'test_outside.py')
        if test_tree == 'scan-denied':
            _deny_source_scan(monkeypatch, tests)
    assert source.source_identity() == initial
    store = RunStore.claim(tmp_path / 'run-root', 'artificial', output_bytes=4_000_000,
                           source_supplier=source.source_identity)
    assert store.source_snapshot() == initial
    assert store.current_source_identity() == initial
    if test_tree == 'bytes':
        (tests / 'test_example.py').write_text('TEST_ONLY = 2\n')
        assert store.current_source_identity() == initial
    (plugin / 'production.py').write_text('VERSION = 2\n')
    assert store.current_source_identity() != initial
    assert store.source_snapshot() == initial


def _deny_source_scan(monkeypatch: pytest.MonkeyPatch, denied: Path) -> None:
    # Exercise actual os.walk onerror, without relying on chmod/user privileges.
    from typing import Any

    original = os.scandir

    def scandir(path: Any) -> Any:
        if Path(path) == denied:
            raise PermissionError('synthetic non-enumerable source tree')
        return original(path)

    monkeypatch.setattr(os, 'scandir', scandir)


@pytest.mark.parametrize('damage', ['file-link', 'directory-link', 'ancestor-link',
                                     'scan-error', 'core-tests-scan-error',
                                     'read-error', 'missing-init'])
def test_source_keeps_production_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    artificial_source: tuple[Path, Path], damage: str,
) -> None:
    from cipher_lab_plugins.fourier import source

    from cipher_lab.artifacts import ArtifactError, RunStore

    core, plugin = artificial_source
    if damage == 'file-link':
        (plugin / 'linked.py').symlink_to(plugin / 'production.py')
    elif damage == 'directory-link':
        (plugin / 'linked').symlink_to(core, target_is_directory=True)
    elif damage == 'ancestor-link':
        alias = tmp_path / 'alias'
        alias.symlink_to(plugin.parent, target_is_directory=True)
        monkeypatch.setattr(source, '_source_roots', lambda: (core, alias / 'fourier'))
    elif damage == 'scan-error':
        production = plugin / 'production'
        production.mkdir()
        _deny_source_scan(monkeypatch, production)
    elif damage == 'core-tests-scan-error':
        core_tests = core / 'tests'
        core_tests.mkdir()
        _deny_source_scan(monkeypatch, core_tests)
    elif damage == 'read-error':
        original = Path.read_bytes

        def read_bytes(path: Path) -> bytes:
            if path == plugin / 'production.py':
                raise PermissionError('synthetic unreadable production source')
            return original(path)

        monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    else:
        (plugin / '__init__.py').unlink()
    with pytest.raises(ArtifactError, match='SOURCE_UNAVAILABLE'):
        source.source_identity()
    root = tmp_path / 'run-root'
    with pytest.raises(ArtifactError, match='SOURCE_UNAVAILABLE'):
        RunStore.claim(root, 'artificial', output_bytes=4_000_000,
                       source_supplier=source.source_identity)
    assert not root.exists()

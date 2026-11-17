"""Artificial source trees and sealed owners; no production inputs."""
import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from cipher_lab.artifacts import ArtifactError
from cipher_lab_plugins.integral_search import source
from cipher_lab_plugins.integral_search.method import run_integral
from cipher_lab_plugins.integral_search.tests.fakes import (
    FakeCollector,
    FakeMeter,
    FakeStore,
    recipe,
)
from cipher_lab_plugins.integral_search.tests.fakes import (
    narrow_math as narrow_math,
)


def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    core = tmp_path / 'core/cipher_lab'
    plugin = tmp_path / 'plugins/cipher_lab_plugins/integral_search'
    for root in (core, plugin, plugin.parent):
        root.mkdir(parents=True, exist_ok=True)
        (root / '__init__.py').write_text('')
    for name in ('ciphers', 'data', 'integral', 'models', 'training', 'evaluation',
                 'representations'):
        (core / name).mkdir()
        (core / name / '__init__.py').write_text('')
    monkeypatch.setattr(source, '_source_roots', lambda: (core, plugin))
    return core, plugin


def test_exact_source_closure_fresh_and_test_unrelated_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    core, plugin = tree(tmp_path, monkeypatch)
    (plugin / 'method.py').write_text('x = 1')
    (plugin / 'tests').symlink_to(tmp_path / 'absent', target_is_directory=True)
    other = plugin.parent / 'unrelated'
    other.mkdir()
    (other / 'broken.py').symlink_to(tmp_path / 'absent')
    initial = source.source_identity()
    assert set(initial) == {'cipher_lab/__init__.py',
        *(f'cipher_lab/{name}/__init__.py' for name in
          ('ciphers', 'data', 'integral', 'models', 'training', 'evaluation', 'representations')),
        'cipher_lab_plugins/__init__.py',
        'cipher_lab_plugins/integral_search/__init__.py',
        'cipher_lab_plugins/integral_search/method.py'}
    (plugin / 'method.py').write_text('x = 2')
    assert source.source_identity() != initial
    (core / 'new.py').write_text('')
    assert 'cipher_lab/new.py' in source.source_identity()


@pytest.mark.parametrize('point', ['core', 'plugin', 'namespace_init', 'core_init',
    'plugin_init', 'required_dir', 'required_init', 'source_link', 'dir_link', 'ancestor'])
def test_missing_or_lexical_symlink_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, point: str,
) -> None:
    core, plugin = tree(tmp_path, monkeypatch)
    if point in ('core', 'plugin'):
        monkeypatch.setattr(source, '_source_roots', lambda: (
            tmp_path / 'absent' if point == 'core' else core,
            tmp_path / 'absent' if point == 'plugin' else plugin))
    elif point in ('namespace_init', 'core_init', 'plugin_init', 'required_init'):
        target = {'namespace_init': plugin.parent / '__init__.py',
                  'core_init': core / '__init__.py', 'plugin_init': plugin / '__init__.py',
                  'required_init': core / 'integral/__init__.py'}[point]
        target.unlink()
    elif point == 'required_dir':
        (core / 'integral/__init__.py').unlink()
        (core / 'integral').rmdir()
    elif point == 'source_link':
        (plugin / 'x.py').symlink_to(core / '__init__.py')
    elif point == 'dir_link':
        (plugin / 'nested').symlink_to(core, target_is_directory=True)
    else:
        alias = tmp_path / 'alias'
        alias.symlink_to(plugin.parent, target_is_directory=True)
        monkeypatch.setattr(source, '_source_roots', lambda: (core, alias / plugin.name))
    with pytest.raises(ArtifactError, match='SOURCE_UNAVAILABLE'):
        source.source_identity()


def test_scan_error_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tree(tmp_path, monkeypatch)
    def broken(*args: object, **kwargs: object) -> object:
        raise PermissionError('PRIVATE')
    monkeypatch.setattr(os, 'walk', broken)
    with pytest.raises(ArtifactError, match='^SOURCE_UNAVAILABLE$'):
        source.source_identity()


@pytest.mark.parametrize('point', ['append_event', 'write_payload', 'write_snapshot'])
def test_storage_failure_owner_propagates_without_finalizing_or_retry(
    monkeypatch: pytest.MonkeyPatch, point: str,
) -> None:
    store = FakeStore()
    collector = FakeCollector()
    sealed = False
    error = ArtifactError('OUTPUT_LIMIT')
    original_snapshot = store.write_snapshot
    def fail(*args: object) -> None:
        nonlocal sealed
        assert not sealed
        sealed = True
        raise error
    def snapshot(value: Mapping[str, object]) -> None:
        if collector.calls:
            fail(value)
        original_snapshot(value)
    monkeypatch.setattr(store, point, snapshot if point == 'write_snapshot' else fail)
    monkeypatch.setattr(store, 'finalize', lambda value: pytest.fail('sealed owner finalized'))
    with pytest.raises(ArtifactError) as caught:
        run_integral(recipe((11,)), collector, FakeMeter(), store)
    assert caught.value is error and sealed
    assert len(collector.calls) == 1


def test_same_supplier_scan_failure_stops_and_finalizes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, collector = FakeStore(), FakeCollector()
    def broken() -> dict[str, str]:
        raise ArtifactError('SOURCE_UNAVAILABLE')
    def change(cell: object, domain: object) -> None:
        monkeypatch.setattr(store, 'current_source_identity', broken)
    collector.hook = change
    result = run_integral(recipe(), collector, FakeMeter(), store)
    assert result['status'] == 'STOPPED' and store.finalizations == 1
    assert len(collector.calls) == 1
    assert result['summary']['reason'] == 'IDENTITY_CHANGED'  # type: ignore[index]


@pytest.mark.parametrize('point', ['representations_dir', 'nested_init'])
def test_required_production_layout_rejects_missing_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, point: str,
) -> None:
    core, plugin = tree(tmp_path, monkeypatch)
    if point == 'representations_dir':
        (core / 'representations/__init__.py').unlink()
        (core / 'representations').rmdir()
    if point == 'nested_init':
        (plugin / 'nested').mkdir()
        (plugin / 'nested/owner.py').write_text('')
    with pytest.raises(ArtifactError, match='SOURCE_UNAVAILABLE'):
        source.source_identity()


def test_claim_and_all_guards_use_same_explicit_supplier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uuid

    from cipher_lab.artifacts import RunStore, read_saved

    source_map = {'cipher_lab_plugins/integral_search/artificial.py': '0' * 64}
    calls = 0
    def supplier() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return source_map.copy()
    monkeypatch.setattr(uuid, 'uuid4', lambda: uuid.UUID(int=1))
    store = RunStore.claim(tmp_path, 'supplier', output_bytes=20000000,
                           source_supplier=supplier)
    collector = FakeCollector()
    collector.hook = lambda cell, domain: source_map.update({
        'cipher_lab_plugins/integral_search/artificial.py': '1' * 64})
    result = run_integral(recipe(), collector, FakeMeter(), store)
    saved = read_saved(tmp_path, 'supplier')
    assert result['status'] == saved['status'] == 'STOPPED'
    assert saved['source'] == {'cipher_lab_plugins/integral_search/artificial.py': '0' * 64}
    assert len(collector.calls) == 1 and calls > 3
    assert result['summary']['reason'] == 'IDENTITY_CHANGED'  # type: ignore[index]

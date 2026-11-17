"""Opaque synthetic bytes exercise the real owner; no scientific work."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import pytest

from cipher_lab.artifacts import ArtifactError, RunStore, read_saved
from cipher_lab_plugins.frontier.tests.frontier_fakes import poison

__all__ = ['poison']


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('slot', ['best', 'A-best', 'B-best', 'C-best'])
def test_same_slot_atomic_replace(tmp_path: Path, slot: str) -> None:
    store = RunStore.claim(tmp_path, 'slots', output_bytes=1000000, needs_models=True)
    ref = store.write_model(slot, b'old weights')
    assert store.write_model(slot, b'new weights', replace=True) == ref
    assert store.read_model(ref, sha(b'new weights'), 100) == b'new weights'
    entries = cast(dict[str, Any], read_saved(tmp_path, 'slots')['artifacts'])
    assert entries[ref]['sha256'] == sha(b'new weights')
    assert [p.name for p in (tmp_path/'models/slots').iterdir()] == [slot+'.pt']


@pytest.mark.parametrize('slot', ['A-e1', 'B-e1', 'C-e1', 'arbitrary', 'best-other'])
def test_other_slot_replace_remains_rejected(tmp_path: Path, slot: str) -> None:
    store = RunStore.claim(tmp_path, 'slots', output_bytes=1000000, needs_models=True)
    ref = store.write_model(slot, b'old')
    with pytest.raises(ArtifactError, match='OCCUPIED'):
        store.write_model(slot, b'new', replace=True)
    assert store.read_model(ref, sha(b'old'), 100) == b'old'


@pytest.mark.parametrize('damage', ['cross-slot', 'unregistered', 'same-size', 'symlink'])
def test_slot_read_fail_closed(tmp_path: Path, damage: str) -> None:
    store = RunStore.claim(tmp_path, 'slots', output_bytes=1000000, needs_models=True)
    ref = store.write_model('A-best', b'AAA')
    store.write_model('B-best', b'BBB')
    if damage == 'cross-slot':
        ref = 'model/B-best.pt'  # correct bytes for another identity are not A's weights
    elif damage == 'unregistered':
        ref = 'model/C-best.pt'
    elif damage == 'same-size':
        (tmp_path/'models/slots/A-best.pt').write_bytes(b'XXX')
    else:
        path = tmp_path/'models/slots/A-best.pt'
        path.unlink()
        path.symlink_to(tmp_path/'models/slots/B-best.pt')
    with pytest.raises(ArtifactError):
        store.read_model(ref, sha(b'AAA'), 100)
    with pytest.raises(ArtifactError, match='FINALIZED'):
        store.write_model('C-best', b'CCC')


def test_failed_replace_seals_and_retains_old_bytes(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    store = RunStore.claim(tmp_path, 'slots', output_bytes=1000000, needs_models=True)
    store.write_model('C-best', b'old weights')
    atomic = store._atomic

    def fail(target: Path, raw: bytes) -> None:
        if target.name == 'C-best.pt':
            raise OSError('synthetic failure before atomic commit')
        atomic(target, raw)

    monkeypatch.setattr(store, '_atomic', fail)
    with pytest.raises(ArtifactError, match='WRITE_FAILED'):
        store.write_model('C-best', b'new weights', replace=True)
    assert (tmp_path/'models/slots/C-best.pt').read_bytes() == b'old weights'
    assert read_saved(tmp_path, 'slots')['status'] == 'FAILED'
    with pytest.raises(ArtifactError, match='FINALIZED'):
        store.finalize({'status': 'FAILED'})


def test_temporary_peak_counts_old_and_new(tmp_path: Path) -> None:
    store = RunStore.claim(tmp_path, 'slots', output_bytes=1000000, needs_models=True)
    store.write_model('A-best', b'a'*10000)
    # Enough for a final same-size model, but not old+new temporary bytes.
    resident = sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file())
    store._limit = resident + 5000
    with pytest.raises(ArtifactError, match='OUTPUT_LIMIT'):
        store.write_model('A-best', b'b'*10000, replace=True)
    assert (tmp_path/'models/slots/A-best.pt').read_bytes() == b'a'*10000
    # A full failure record can itself exceed remaining temporary space: UNKNOWN.
    assert read_saved(tmp_path, 'slots')['status'] in ('FAILED', 'RUNNING')
    with pytest.raises(ArtifactError, match='FINALIZED'):
        store.finalize({'status': 'SUCCEEDED'})


@pytest.mark.parametrize('slot', ['A-best', 'B-best', 'C-best'])
@pytest.mark.parametrize('failure', ['model-rename', 'metadata-once', 'metadata-always'])
def test_frontier_replace_media_failure_keeps_sha_truth(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, slot: str, failure: str) -> None:
    import os

    from cipher_lab.artifacts import read_binary
    store = RunStore.claim(tmp_path, 'slots', output_bytes=1000000, needs_models=True)
    ref = store.write_model(slot, b'old')
    rename = os.replace
    failed = False
    def fault(src: Any, dst: Any) -> None:
        nonlocal failed
        name = Path(dst).name
        if ((failure == 'model-rename' and name == slot+'.pt')
                or (failure.startswith('metadata') and name == 'run.json'
                    and (not failed or failure == 'metadata-always'))):
            failed = True
            raise OSError('synthetic media failure')
        rename(src, dst)
    monkeypatch.setattr(os, 'replace', fault)
    with pytest.raises(ArtifactError, match='WRITE_FAILED'):
        store.write_model(slot, b'new', replace=True)
    with pytest.raises(ArtifactError, match='FINALIZED'):
        store.finalize({'status': 'SUCCEEDED'})
    if failure == 'metadata-always':
        assert (tmp_path/f'models/slots/{slot}.pt').read_bytes() == b'new'
        with pytest.raises(ArtifactError, match='HASH_MISMATCH'):
            read_binary(tmp_path, 'slots', ref, sha(b'old'), 100)
    else:
        assert read_saved(tmp_path, 'slots')['status'] == 'FAILED'
        expected = b'new' if failure == 'metadata-once' else b'old'
        assert read_binary(tmp_path, 'slots', ref, sha(expected), 100) == expected
    assert not list(tmp_path.rglob('*.tmp'))


@pytest.mark.parametrize('damage', ['same-size', 'symlink', 'unregistered'])
def test_replace_cannot_adopt_foreign_slot(tmp_path: Path, damage: str) -> None:
    store = RunStore.claim(tmp_path, 'slots', output_bytes=1000000, needs_models=True)
    path = tmp_path/'models/slots/C-best.pt'
    if damage != 'unregistered':
        store.write_model('C-best', b'old')
    if damage == 'symlink':
        other = tmp_path/'foreign'
        other.write_bytes(b'XXX')
        path.unlink()
        path.symlink_to(other)
    else:
        path.write_bytes(b'XXX')
    with pytest.raises(ArtifactError):
        store.write_model('C-best', b'new', replace=True)
    assert path.read_bytes() == b'XXX'

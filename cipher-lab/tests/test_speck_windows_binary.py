"""Exact Windows CRT text translation emulation; no native DLL or science."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from cipher_lab import artifacts
from cipher_lab.artifacts import ArtifactError, RunStore, read_saved


@pytest.fixture(autouse=True)
def no_native_or_entropy(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError('binary regression forbids native/science/entropy')
    monkeypatch.setattr(os, 'urandom', forbidden)
    monkeypatch.setattr(secrets, 'token_bytes', forbidden)
    monkeypatch.setattr(ctypes, 'CDLL', forbidden)
    monkeypatch.setattr(ctypes, 'WinDLL', forbidden, raising=False)
    monkeypatch.setattr(artifacts, 'uuid4', lambda: UUID(int=1))


@pytest.fixture
def win_crt(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Model os.open/os.write's CRT LF expansion, preserving write return count."""
    binary = 0x8000
    native_binary = getattr(os, 'O_BINARY', 0)
    real_open, real_write = os.open, os.write
    real_close, real_sync, real_replace = os.close, os.fsync, os.replace
    modes: dict[int, bool] = {}
    observed: dict[str, Any] = {'flags': [], 'sync': 0, 'replace': 0}

    def opened(path: Any, flags: int, mode: int = 0o777) -> int:
        # Actual backing fd is always binary, even when running on Windows:
        # only this emulator, never the host CRT, performs the text translation.
        fd = real_open(path, (flags & ~binary) | native_binary, mode)
        modes[fd] = bool(flags & binary)
        observed['flags'].append(flags)
        return fd

    def written(fd: int, value: bytes) -> int:
        disk = value if modes.get(fd, True) else value.replace(b'\n', b'\r\n')
        assert real_write(fd, disk) == len(disk)
        return len(value)

    def closed(fd: int) -> None:
        modes.pop(fd, None)
        real_close(fd)

    def synced(fd: int) -> None:
        observed['sync'] += 1
        real_sync(fd)

    def replaced(source: Any, target: Any) -> None:
        observed['replace'] += 1
        real_replace(source, target)

    monkeypatch.setattr(os, 'O_BINARY', binary, raising=False)
    monkeypatch.setattr(os, 'open', opened)
    monkeypatch.setattr(os, 'write', written)
    monkeypatch.setattr(os, 'close', closed)
    monkeypatch.setattr(os, 'fsync', synced)
    monkeypatch.setattr(os, 'replace', replaced)
    monkeypatch.setattr(artifacts, 'uuid4', lambda: UUID(int=1))
    monkeypatch.setattr(artifacts, '_environment_identity', lambda: {
        'python_implementation': 'CPython', 'python_version': '3.10.16',
        'os': 'Windows', 'architecture': 'AMD64'})
    return observed


def test_windows_atomic_claim_snapshot_registered_binary_roundtrip(
    tmp_path: Path, win_crt: dict[str, Any],
) -> None:
    owner = RunStore.claim(tmp_path, 'binary', output_bytes=1_000_000, needs_models=True)
    record = tmp_path / 'data/binary/run.json'
    raw = record.read_bytes()
    assert raw.endswith(b'\n') and not raw.endswith(b'\r\n')
    # The first snapshot must still own the exact claim, not a normalized hash.
    owner.write_snapshot({'summary': {'scientific_validation': False}})
    content = b'opaque\n\r\n\x00\x1a\xff'
    ref = owner.write_model('A-best', content)
    owner.write_payload('manual', {'value': 'public'})
    owner.write_dataset('manual', content)
    digest = hashlib.sha256(content).hexdigest()
    saved = json.loads(record.read_bytes())
    assert saved['artifacts'][ref] == {'kind': 'model', 'sha256': digest, 'bytes': len(content)}
    assert (tmp_path / 'models/binary/A-best.pt').read_bytes() == content
    assert owner.read_model(ref, digest, len(content)) == content
    owner.write_model('A-best', content + b'\n', replace=True)
    owner.finalize({'status': 'SUCCEEDED'})
    saved = read_saved(tmp_path, 'binary')
    reports = {'report.md': b'public\nreport\n', 'index.html': b'<p>public</p>\n',
               'source.json': (json.dumps({'schema': 1, 'run_id': 'binary',
                   'run_ref': saved['run_ref'],
                   'run_sha256': saved['run_sha256']}) + '\n').encode()}
    owner.write_report_files(reports)
    assert all((tmp_path / 'result/binary' / name).read_bytes() == data
               for name, data in reports.items())
    assert (tmp_path / 'data/binary/dataset/manual.npz').read_bytes() == content
    assert all(flags & getattr(os, 'O_BINARY') for flags in win_crt['flags'])
    assert win_crt['sync'] == win_crt['replace'] == len(win_crt['flags'])
    assert not list(tmp_path.rglob('*.tmp'))


def test_windows_append_events_exact_lf_and_hash(
    tmp_path: Path, win_crt: dict[str, Any],
) -> None:
    owner = RunStore.claim(tmp_path, 'events', output_bytes=1_000_000)
    for _ in range(2):
        owner.append_event({'kind': 'manual'})
    raw = (tmp_path / 'data/events/events.jsonl').read_bytes()
    assert b'\r' not in raw
    assert [json.loads(line)['sequence'] for line in raw.splitlines()] == [1, 2]
    assert all(flags & getattr(os, 'O_BINARY') for flags in win_crt['flags'])
    assert win_crt['sync'] == 3 and win_crt['replace'] == 1


@pytest.mark.parametrize('fault', ['owner', 'model', 'model-rehashed', 'same-size-rehashed'])
def test_windows_binary_never_normalizes_tampered_owner_or_model(
    tmp_path: Path, win_crt: dict[str, Any], fault: str,
) -> None:
    owner = RunStore.claim(tmp_path, 'fault', output_bytes=1_000_000, needs_models=True)
    content = b'manual\n'
    ref = owner.write_model('A-best', content)
    path = tmp_path / ('data/fault/run.json' if fault == 'owner' else 'models/fault/A-best.pt')
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'manual', b'Manual') if fault == 'same-size-rehashed'
                     else raw.replace(b'\n', b'\r\n'))
    live = fault in ('model-rehashed', 'same-size-rehashed')
    digest = hashlib.sha256(path.read_bytes() if live else content).hexdigest()
    with pytest.raises(ArtifactError):
        owner.read_model(ref, digest, 1000)
    with pytest.raises(ArtifactError, match='FINALIZED'):
        owner.write_payload('no-recovery', {})


def test_posix_without_binary_flag_preserves_original_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != 'posix':
        pytest.skip('POSIX no-O_BINARY contract')
    monkeypatch.delattr(os, 'O_BINARY', raising=False)
    real_open = os.open
    observed: list[int] = []

    def opened(path: Any, flags: int, mode: int = 0o777) -> int:
        observed.append(flags)
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, 'open', opened)
    owner = RunStore.claim(tmp_path, 'posix', output_bytes=1_000_000)
    owner.append_event({'kind': 'manual'})
    owner.append_event({'kind': 'manual'})
    assert observed == [os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL,
                        os.O_WRONLY | os.O_APPEND]

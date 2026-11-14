"""Strict retained-owner read regression: opaque bytes, no scientific imports."""
from __future__ import annotations

import ctypes
import hashlib
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

from cipher_lab import artifacts
from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter, Stopped


@pytest.fixture(autouse=True)
def storage_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError('owner read tests forbid entropy/native/resource probes')
    monkeypatch.setattr(os, 'urandom', forbidden)
    monkeypatch.setattr(secrets, 'token_bytes', forbidden)
    monkeypatch.setattr(ctypes, 'CDLL', forbidden)
    monkeypatch.setattr(ctypes, 'WinDLL', forbidden, raising=False)
    monkeypatch.setattr('cipher_lab.execution.production_rss_bytes', forbidden)
    monkeypatch.setattr(artifacts, 'uuid4', lambda: UUID(int=1))


@pytest.mark.parametrize('fault', ['valid', 'wrong-sha', 'size', 'unregistered',
                                  'tamper-bytes', 'symlink', 'metadata-owner'])
def test_registered_bounded_read_and_sticky_owner(
    tmp_path: Path, fault: str,
) -> None:
    owner = RunStore.claim(tmp_path, 'opaque', output_bytes=1_000_000, needs_models=True)
    ref = owner.write_model('A-best', b'opaque-model')
    digest = hashlib.sha256(b'opaque-model').hexdigest()
    limit = len(b'opaque-model')
    path = tmp_path/'models/opaque/A-best.pt'
    if fault == 'wrong-sha':
        digest = '0'*64
    elif fault == 'size':
        limit -= 1
    elif fault == 'unregistered':
        ref = 'model/unregistered.pt'
    elif fault == 'tamper-bytes':
        path.write_bytes(b'opaque-modeX')
    elif fault == 'symlink':
        target = tmp_path/'opaque-target'
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    elif fault == 'metadata-owner':
        record = tmp_path/'data/opaque/run.json'
        record.write_bytes(record.read_bytes()+b' ')
    if fault == 'valid':
        assert owner.read_model(ref, digest, limit) == b'opaque-model'
        # HEAD behavior and the two necessary study slots remain explicit.
        for name in ('best', 'A-best', 'B-best'):
            if name != 'A-best':
                owner.write_model(name, b'first')
            owner.write_model(name, b'updated', replace=True)
        owner.write_model('C-best', b'first')
        with pytest.raises(ArtifactError, match='OCCUPIED'):
            owner.write_model('C-best', b'not-approved', replace=True)
    else:
        with pytest.raises(ArtifactError):
            owner.read_model(ref, digest, limit)
        with pytest.raises(ArtifactError, match='FINALIZED'):
            owner.write_payload('no-recovery', {})


def test_cpu_false_is_explicit_default_cpu_stop_is_unchanged() -> None:
    policy = Policy(2, 100, 1, '2099-01-01T00:00:00Z', 1, 0, 100, 1000, 0, 0)
    cpu = [0.0]
    clocks = Clocks(lambda: 1.0, lambda: cpu[0],
                    lambda: datetime(2030, 1, 1, tzinfo=timezone.utc))
    default = Meter(policy, clocks, lambda: 0)
    study = Meter(policy, clocks, lambda: 0, enforce_cpu=False)
    cpu[0] = 2.0
    with pytest.raises(Stopped, match='CPU_LIMIT'):
        default.check()
    study.check()
    assert study.snapshot()['cpu_elapsed_seconds'] == 2.0

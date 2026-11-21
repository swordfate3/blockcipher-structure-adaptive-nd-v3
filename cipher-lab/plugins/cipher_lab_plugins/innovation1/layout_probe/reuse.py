"""Read only the three pinned files, never claim or load the old run's weights."""
from __future__ import annotations

import io
import os
import re
import stat
from collections.abc import Callable
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from zipfile import ZIP_STORED, BadZipFile, ZipFile

import numpy as np
from numpy.typing import NDArray

from cipher_lab.artifacts import ArtifactError, _decode, _validate

from ..config import canonical, validate_run_id
from ..data import BlockMeter, Oracle, PrivateSource
from ..single_train.config import BACKEND, estimate
from ..single_train.data import PairBatch
from ..source import no_reparse
from .config import FIXED_MSE, SLOTS, FilePin, LayoutRecipe
from .data import FixedPoolProvider

MAX_RECORD = 1 << 20
MAX_METADATA = 1 << 20
CHUNK = 1 << 20


class PoolSourceError(ValueError):
    """Missing, unsafe or mismatched archived input; there is no alternate source."""


def _pinned(path: Path, pin: FilePin, limit: int, check: Callable[[], None]) -> io.BytesIO:
    if (type(pin.bytes) is not int or not 0 < pin.bytes <= limit
            or not re.fullmatch('[0-9a-f]{64}', pin.sha256)):
        raise PoolSourceError
    path = no_reparse(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size != pin.bytes:
        raise PoolSourceError
    digest, raw = sha256(), io.BytesIO()
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if (not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened)
                or opened.st_size != pin.bytes):
            raise PoolSourceError
        remaining = pin.bytes
        while remaining:
            check()
            chunk = stream.read(min(CHUNK, remaining))
            if not chunk:
                raise PoolSourceError
            remaining -= len(chunk)
            digest.update(chunk)
            raw.write(chunk)
        if stream.read(1) or digest.hexdigest() != pin.sha256:
            raise PoolSourceError
    if not os.path.samestat(opened, no_reparse(path).lstat()):
        raise PoolSourceError
    check()
    raw.seek(0)
    return raw


def _array(archive: ZipFile, name: str, shape: tuple[int, ...] | None,
           limit: int) -> NDArray[np.uint8]:
    member = archive.getinfo(name + '.npy')
    if member.compress_type != ZIP_STORED or not 0 < member.file_size <= limit + 1024:
        raise PoolSourceError
    # Read the header first and bound the data before allocating it. The original
    # np.savez writer emits small v1 headers and uncompressed native uint8 only.
    with archive.open(member) as stream:
        if np.lib.format.read_magic(stream) != (1, 0):
            raise PoolSourceError
        actual, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=1024)
        if (dtype != np.dtype(np.uint8) or fortran
                or (shape is None and (len(actual) != 1 or not 0 < actual[0] <= limit))
                or (shape is not None and shape != actual)):
            raise PoolSourceError
        size = actual[0] if shape is None else limit
        if stream.tell() + size != member.file_size:
            raise PoolSourceError
        raw = stream.read(size)
        if len(raw) != size or stream.read(1):
            raise PoolSourceError
    return np.frombuffer(raw, np.uint8).reshape(actual)


def _panel(path: Path, file: FilePin, groups: int, split: str, expected: dict[str, Any],
           recipe: LayoutRecipe, check: Callable[[], None]) -> PairBatch:
    with (_pinned(path, file, groups * 17 + MAX_METADATA + 8192, check) as raw,
          ZipFile(raw) as archive):
        if sorted(archive.namelist()) != ['ciphertext.npy', 'labels.npy', 'metadata.npy']:
            raise PoolSourceError
        meta = _decode(_array(archive, 'metadata', None, MAX_METADATA).tobytes())
        if canonical(meta) != canonical(expected):
            raise PoolSourceError
        check()
        x = _array(archive, 'ciphertext', (groups, 1, 2, 8), groups * 16)
        check()
        y = _array(archive, 'labels', (groups, 1), groups)
    batch = PairBatch(x, y, split, 0, recipe.namespace)
    if (batch.digest != expected['digest']
            or (split == 'dev' and (y[::2].any() or not (y[1::2] == 1).all()))):
        raise PoolSourceError
    check()
    return batch


def _load(recipe: LayoutRecipe, check: Callable[[], None]) -> tuple[PairBatch, PairBatch,
                                                                     dict[str, Any]]:
    pin = recipe.pool_source
    if pin is None or not recipe.reuse_pool:
        raise PoolSourceError
    validate_run_id(pin.run_id)
    root = Path(pin.root)
    if not root.is_absolute():
        raise PoolSourceError
    base = root / 'data' / pin.run_id
    totals = estimate(recipe)
    tr = recipe.training
    # Production's child is exact; the guarded artificial seam scales these pool
    # dimensions only. The original epoch, TEST, and eval geometry remain pinned.
    old = replace(FIXED_MSE, run_id=pin.run_id, training=replace(
        FIXED_MSE.training, batch_groups=tr.batch_groups, batches_per_epoch=tr.batches_per_epoch,
        dev_groups=tr.dev_groups))
    with _pinned(base / 'run.json', pin.record, MAX_RECORD, check) as raw:
        record = cast(dict[str, Any], _validate(_decode(raw.read()), pin.run_id))
    summary, counts = record['summary'], record['counts']
    if (record['status'] != 'RUNNING' or canonical(record['recipe']) != canonical(old.to_dict())
            or summary.get('method') != old.method or summary.get('backend_identity') != BACKEND
            or summary.get('synthetic') is not False or summary.get('pool_status') != 'READY'
            or any(type(counts.get(f'{kind}_{unit}')) is not int
                   or counts[f'{kind}_{unit}'] != expected
                   for kind in ('attempted', 'completed')
                   for unit, expected in (('real_blocks', 2 * totals['pool_groups']),
                                          ('null_words', 0)))):
        raise PoolSourceError
    evidence: dict[str, Any] = {'pool_source': {
        **asdict(pin), 'record_ref': f'data/{pin.run_id}/run.json',
        'method': old.method, 'namespace': old.namespace,
        'status': record['status'], 'pool_status': summary['pool_status'],
    }}
    panels = []
    for split, file, groups in (('train', pin.train, totals['train_groups_per_epoch']),
                                ('dev', pin.dev, tr.dev_groups)):
        ref = f'dataset/{split}-SKINNY.npz'
        info = summary.get(f'{split}_panel')
        if (record['artifacts'].get(ref) != dict(kind='dataset', **asdict(file))
                or not isinstance(info, dict) or set(info) != {'ref', 'sha256', 'digest'}
                or info['ref'] != ref or info['sha256'] != file.sha256
                or not isinstance(info['digest'], str)
                or not re.fullmatch('[0-9a-f]{64}', info['digest'])):
            raise PoolSourceError
        expected = dict(schema=old.artifact_schema, method=old.method, run_id=pin.run_id,
                        recipe=record['recipe'], source=record['source'], backend=BACKEND,
                        split=split, digest=info['digest'], encoding='packed-msb0-single-pair-v1',
                        slots=list(SLOTS))
        panels.append(_panel(base / ref, file, groups, split, expected, old, check))
        evidence[f'{split}_panel'] = dict(**info, origin='reused', bytes=file.bytes, groups=groups,
                                        source_run_id=pin.run_id)
    return panels[0], panels[1], evidence


class ReusedPoolProvider(FixedPoolProvider):
    def __init__(self, recipe: LayoutRecipe, oracle: Oracle, private: PrivateSource,
                 meter: BlockMeter) -> None:
        if not recipe.reuse_pool:
            raise PoolSourceError
        super().__init__(recipe, oracle, private, meter)
        self._provenance: dict[str, Any] | None = None

    def prepare_pool(self) -> PairBatch:
        if self._started:
            raise ValueError('POOL_ORDER')
        self._started = True
        try:
            train, dev, provenance = _load(self.recipe, self.meter.check)
        except (ArtifactError, OSError, ValueError, BadZipFile, EOFError, KeyError, SyntaxError):
            raise PoolSourceError from None
        self._dev = dev
        train = self._use_train(train)
        self._provenance = provenance
        return train

    def pool_provenance(self) -> dict[str, Any]:
        if self._provenance is None or 'dev' not in self._panels:
            raise ValueError('POOL_NOT_READY')
        return self._provenance

    def _collect(self, split: str, index: int, size: int) -> PairBatch:
        if split != 'test':
            raise ValueError('POOL_REUSE_NO_GENERATION')
        return super()._collect(split, index, size)

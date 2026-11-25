"""Read only the pinned source record and, on demand, its three compact panels."""
from __future__ import annotations

import stat
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Literal
from zipfile import ZIP_STORED, BadZipFile, ZipFile

import numpy as np

from cipher_lab.artifacts import (
    ArtifactError,
    _decode,
    _mapping,
    _path,
    _ref,
    _stream,
    _token,
    _validate,
)
from cipher_lab.execution import Meter

from . import config
from .data import Panel, Words

_RUN_BYTES = 16 * 2**20
_HEADER_BYTES = 512


@dataclass(frozen=True, slots=True)
class SavedPanel(Panel):
    raw: bytes

    def archive(self) -> bytes:
        return self.raw  # exact source artifact, no second serialization


def _source_path(path: Path) -> Path:
    checked = _path(path)
    # On Windows a directory junction is not necessarily a symlink to _path.
    if any(part.is_junction() for part in (*checked.parents, checked)):
        raise ArtifactError('UNSAFE_PATH')
    return checked


def _record(root: Path, run_id: str, digest: str) -> bytes:
    try:
        path = _source_path(root / 'data' / run_id / 'run.json')
        if not stat.S_ISREG(path.stat().st_mode):
            raise ArtifactError('INVALID_SOURCE_RUN')
        with path.open('rb') as stream:
            raw = stream.read(_RUN_BYTES + 1)
    except OSError:
        raise ArtifactError('READ_FAILED') from None
    if len(raw) > _RUN_BYTES or sha256(raw).hexdigest() != digest:
        raise ArtifactError('HASH_MISMATCH')
    return raw


def _array(raw: bytes, shape: tuple[int, ...], payload: int) -> Words:
    # The pinned producer uses NumPy's v1.0 header, little-endian words.
    stream = BytesIO(raw)
    if (np.lib.format.read_magic(stream) != (1, 0)
            or not 0 < int.from_bytes(stream.read(2), 'little') <= _HEADER_BYTES):
        raise ValueError
    stream.seek(8)
    dims, fortran, dtype = np.lib.format.read_array_header_1_0(
        stream, max_header_size=_HEADER_BYTES)
    if (dims != shape or any(type(n) is not int for n in dims) or fortran
            or dtype.str != '<u8' or len(raw) - stream.tell() != payload):
        raise ValueError
    stream.seek(0)
    value = np.load(stream, allow_pickle=False, max_header_size=_HEADER_BYTES)
    if (type(value) is not np.ndarray or value.shape != shape or value.dtype.str != '<u8'
            or not value.flags.c_contiguous):
        raise ValueError
    return value.astype(np.uint64, copy=False)


def _panel(raw: bytes, families: int, meter: Meter) -> tuple[SavedPanel, Words]:
    rows = families * config.PAIRS_PER_FAMILY
    shapes = {'plaintext.npy': (rows, 2), 'ciphertext.npy': (rows, 2), 'family_ids.npy': (rows,)}
    arrays: dict[str, Words] = {}
    try:
        with ZipFile(BytesIO(raw)) as archive:
            entries = archive.infolist()
            if len(entries) != len(shapes) or {item.filename for item in entries} != shapes.keys():
                raise ValueError
            for item in entries:
                shape = shapes[item.filename]
                payload = rows * (16 if len(shape) == 2 else 8)
                if (item.orig_filename != item.filename or item.flag_bits & 1
                        or item.compress_type != ZIP_STORED or item.compress_size != item.file_size
                        or not payload + 10 < item.file_size <= payload + 10 + _HEADER_BYTES):
                    raise ValueError
                meter.check()
                arrays[item.filename] = _array(archive.read(item), shape, payload)
                meter.check()
    except (BadZipFile, ValueError, EOFError, OverflowError, NotImplementedError, RuntimeError):
        raise ValueError('INVALID_SOURCE_PANEL') from None
    panel = SavedPanel(arrays['plaintext.npy'], arrays['ciphertext.npy'],
                       arrays['family_ids.npy'], raw)
    left, right = panel.plaintext.T
    mask = np.uint64(config.MASK)
    # CubePanels fixes the high active bit in the left endpoint to zero. With
    # distinct left words there are at most eight valid rows per family. Exact
    # rows + family count therefore require all eight complementary pairs.
    if (not np.all(right == (left ^ mask)) or np.any(left & np.uint64(1 << config.ACTIVE_BITS[-1]))
            or not np.all(panel.family_ids == (left & ~mask)) or len(np.unique(left)) != rows):
        raise ValueError('INVALID_SOURCE_PANEL')
    bases = np.unique(panel.family_ids)
    if len(bases) != families:
        raise ValueError('INVALID_SOURCE_PANEL')
    meter.check()
    return panel, bases


class _PinnedSource:
    """An explicit run identity and its allowed dataset references only."""

    def __init__(self, root: Path, meter: Meter, *, run_id: str, run_sha256: str,
                 pins: Mapping[str, tuple[int, str]]) -> None:
        self._root, self._meter = root.absolute(), meter
        self._run_id, self._sha = _token(run_id), run_sha256
        meter.check()
        # Identity precedes interpretation. Never open any score/model/payload.
        record = _validate(_decode(_record(self._root, self._run_id, self._sha)), run_id)
        if record['status'] != 'SUCCEEDED':
            raise ArtifactError('INVALID_SOURCE_RUN')
        artifacts = _mapping(record['artifacts'])
        self._pins: dict[str, dict[str, object]] = {}
        for ref, (size, digest) in pins.items():
            if not _ref(ref).startswith('dataset/') or artifacts.get(ref) != {
                'kind': 'dataset', 'bytes': size, 'sha256': digest,
            }:
                raise ArtifactError('INVALID_SOURCE_RUN')
            self._pins[ref] = {'kind': 'dataset', 'bytes': size, 'sha256': digest}
        meter.check()

    def panel(self, ref: str, families: int) -> tuple[SavedPanel, Words]:
        meter = self._meter
        meter.check()
        _record(self._root, self._run_id, self._sha)
        meter.check()
        try:
            path = _source_path(self._root / 'data' / self._run_id / ref)
            raw = _stream(path, self._pins[ref], collect=True)
        except OSError:
            raise ArtifactError('READ_FAILED') from None
        meter.check()
        _record(self._root, self._run_id, self._sha)
        meter.check()
        return _panel(raw, families, meter)


def _check_disjoint(bases: Words, earlier: list[Words], meter: Meter) -> None:
    if any(np.intersect1d(bases, other, assume_unique=True).size for other in earlier):
        raise ArtifactError('SOURCE_SPLIT_OVERLAP')
    meter.check()


class SavedPanels:
    """Single read-only provider. TEST can only be requested as the last split."""

    def __init__(self, source_root: Path, meter: Meter) -> None:
        self._meter = meter
        self._next = 0
        self._failed = False
        self._bases: list[Words] = []
        self._source = _PinnedSource(
            source_root, meter, run_id=config.SOURCE_RUN_ID, run_sha256=config.SOURCE_RUN_SHA256,
            pins={f'dataset/{split}.npz': pin for split, pin in
                  zip(config.SPLITS, config.SOURCE_SPLITS, strict=True)},
        )

    @property
    def provenance(self) -> dict[str, object]:
        return config.source_recipe()

    def build(self, split: Literal['train', 'dev']) -> SavedPanel:
        if split not in ('train', 'dev'):
            raise ValueError('invalid split order; use label_test')
        return self._build(split)

    def label_test(self) -> SavedPanel:
        """Call only after both newly persisted checkpoints have been restored."""
        return self._build('test')

    def build_plaintext(self, split: Literal['train', 'dev']) -> tuple[Words, Words]:
        panel = self.build(split)
        return panel.plaintext, panel.family_ids

    def label_test_plaintext(self) -> tuple[Words, Words]:
        """Project the validated source TEST only after the caller's reload gate."""
        panel = self.label_test()
        return panel.plaintext, panel.family_ids

    def _build(self, split: str) -> SavedPanel:
        idx = self._next
        if self._failed or idx >= len(config.SPLITS) or config.SPLITS[idx] != split:
            raise ValueError('invalid split order')
        self._failed = True
        panel, bases = self._source.panel(f'dataset/{split}.npz', config.FAMILIES[idx])
        _check_disjoint(bases, self._bases, self._meter)
        self._bases.append(bases)
        self._next += 1
        self._failed = False
        return panel

"""Read only the pinned source record and, on demand, its three compact panels."""
from __future__ import annotations

import stat
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Literal
from zipfile import ZIP_STORED, BadZipFile, ZipFile

import numpy as np

from cipher_lab.artifacts import ArtifactError, _decode, _mapping, _path, _validate, read_binary
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


def _record(root: Path) -> bytes:
    try:
        path = _source_path(root / 'data' / config.SOURCE_RUN_ID / 'run.json')
        if not stat.S_ISREG(path.stat().st_mode):
            raise ArtifactError('INVALID_SOURCE_RUN')
        with path.open('rb') as stream:
            raw = stream.read(_RUN_BYTES + 1)
    except OSError:
        raise ArtifactError('READ_FAILED') from None
    if len(raw) > _RUN_BYTES or sha256(raw).hexdigest() != config.SOURCE_RUN_SHA256:
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


class SavedPanels:
    """Single read-only provider. TEST can only be requested as the last split."""

    def __init__(self, source_root: Path, meter: Meter) -> None:
        self._root, self._meter = source_root.absolute(), meter
        self._next = 0
        self._failed = False
        self._bases: list[Words] = []
        meter.check()
        # Identity precedes JSON interpretation. Only artifact declarations are
        # consulted; no old score, event, checkpoint or data content is consumed.
        record = _validate(_decode(_record(self._root)), config.SOURCE_RUN_ID)
        if record['status'] != 'SUCCEEDED':
            raise ArtifactError('INVALID_SOURCE_RUN')
        artifacts = _mapping(record['artifacts'])
        for split, (size, digest) in zip(config.SPLITS, config.SOURCE_SPLITS, strict=True):
            if artifacts.get(f'dataset/{split}.npz') != {
                'kind': 'dataset', 'bytes': size, 'sha256': digest,
            }:
                raise ArtifactError('INVALID_SOURCE_RUN')
        meter.check()

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
        meter = self._meter
        meter.check()
        _record(self._root)
        meter.check()
        size, digest = config.SOURCE_SPLITS[idx]
        ref = f'dataset/{split}.npz'
        try:
            _source_path(self._root / 'data' / config.SOURCE_RUN_ID / ref)
        except OSError:
            raise ArtifactError('READ_FAILED') from None
        raw = read_binary(self._root, config.SOURCE_RUN_ID, ref, digest, size)
        meter.check()
        _record(self._root)
        if len(raw) != size:
            raise ArtifactError('HASH_MISMATCH')
        meter.check()
        panel, bases = _panel(raw, config.FAMILIES[idx], meter)
        if any(np.intersect1d(bases, other, assume_unique=True).size for other in self._bases):
            raise ArtifactError('SOURCE_SPLIT_OVERLAP')
        meter.check()
        self._bases.append(bases)
        self._next += 1
        self._failed = False
        return panel

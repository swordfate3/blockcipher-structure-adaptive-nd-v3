"""Strict public experiment identity and model-only checkpoint serialization."""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import ENCODING, INPUT_ADAPTER, MASKS, METHOD


def _sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


@dataclass(frozen=True, slots=True)
class Identity:
    experiment_id: str
    fit_id: str
    train_hash: str
    dev_hash: str
    target_train_hash: str
    target_dev_hash: str
    target_version: str
    masks: tuple[int, ...] = MASKS
    encoding: str = ENCODING
    input_adapter: str = INPUT_ADAPTER
    method: str = METHOD

    def __post_init__(self) -> None:
        if (not isinstance(self.experiment_id, str) or not self.experiment_id
                or len(self.experiment_id) > 128
                or self.fit_id not in ('F1', 'F2', 'F3', 'F4', 'F5')
                or not all(_sha(v) for v in (self.train_hash, self.dev_hash,
                                            self.target_train_hash, self.target_dev_hash))
                or self.target_version != ('row-permutation-v1' if self.fit_id == 'F5'
                                           else 'real-v1')
                or type(self.masks) is not tuple or self.masks != MASKS
                or self.encoding != ENCODING or self.input_adapter != INPUT_ADAPTER
                or self.method != METHOD):
            raise ValueError('INVALID_CHECKPOINT_IDENTITY')

    def to_dict(self) -> dict[str, object]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _valid_tensor(value: object) -> bool:
    return (isinstance(value, torch.Tensor) and value.layout == torch.strided
            and not value.is_quantized and not value.is_meta
            and bool(torch.isfinite(value).all()))


def save_checkpoint(path: Path, model: nn.Module, identity: Identity) -> str:
    if not isinstance(identity, Identity):
        raise ValueError('INVALID_CHECKPOINT_IDENTITY')
    state = model.state_dict()
    if not all(isinstance(k, str) and _valid_tensor(v) for k, v in state.items()):
        raise ValueError('INVALID_STATE_DICT')
    payload = {'schema': 1, 'identity': identity.to_dict(),
               'state_dict': {k: v.detach().cpu().clone() for k, v in state.items()}}
    # Slot ownership and durable replacement remain the caller's RunStore responsibility.
    with path.open('wb') as stream:
        torch.save(payload, stream)
    return _file_sha(path)


def load_checkpoint(path: Path, model: nn.Module, expected: Identity, *,
                    expected_sha256: str | None = None) -> str:
    try:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and (not _sha(expected_sha256) or digest != expected_sha256):
            raise ValueError
        # Hash and deserialize the exact same immutable bytes, not two path opens.
        payload: Any = torch.load(io.BytesIO(raw), weights_only=True, map_location='cpu')
        if (type(payload) is not dict or set(payload) != {'schema', 'identity', 'state_dict'}
                or type(payload['schema']) is not int or payload['schema'] != 1
                or type(payload['identity']) is not dict
                or payload['identity'] != expected.to_dict()):
            raise ValueError
        # Python equality equates True and 1; canonical serialized public identity must not.
        if json.dumps(payload['identity'], sort_keys=True) != json.dumps(
                expected.to_dict(), sort_keys=True):
            raise ValueError
        state = payload['state_dict']
        reference = model.state_dict()
        if type(state) is not dict or set(state) != set(reference):
            raise ValueError
        for name, target in reference.items():
            source = state[name]
            if (not _valid_tensor(source) or source.shape != target.shape
                    or source.dtype != target.dtype):
                raise ValueError
        if _file_sha(path) != digest:
            raise ValueError
        model.load_state_dict(state, strict=True)
        return digest
    except Exception:
        raise ValueError('CHECKPOINT_SCHEMA_OR_IDENTITY_MISMATCH') from None


@dataclass(frozen=True, slots=True)
class SavedDataset:
    """Read-only decoded arrays for evaluation; no private/generation dependencies."""

    train: np.ndarray[Any, np.dtype[np.uint64]]
    dev: np.ndarray[Any, np.dtype[np.uint64]]
    test: np.ndarray[Any, np.dtype[np.uint64]]
    train_labels: np.ndarray[Any, np.dtype[np.uint8]]
    dev_labels: np.ndarray[Any, np.dtype[np.uint8]]
    null_train: np.ndarray[Any, np.dtype[np.uint8]] | None
    null_dev: np.ndarray[Any, np.dtype[np.uint8]] | None
    test_labels: np.ndarray[Any, np.dtype[np.uint8]] | None
    metadata: dict[str, Any]


def _array_digest(a: np.ndarray[Any, Any]) -> str:
    a = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder('<'))
    h = hashlib.sha256()
    h.update(json.dumps([a.dtype.str, list(a.shape)], separators=(',', ':')).encode('ascii'))
    h.update(a.tobytes())
    return h.hexdigest()


def load_saved_dataset(path: Path, *, experiment_id: str) -> SavedDataset:
    """Decode an existing npz only: no data module, RNG, cipher or label generation.

    Content hashes bind actual arrays; callers additionally verify original run's
    file SHA and selected checkpoint SHA. SHA is identity, not authentication.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError
            arrays = {name: archive[name] for name in archive.files}
        raw = arrays.pop('metadata')
        if raw.shape != () or raw.dtype.kind != 'U':
            raise ValueError
        m = json.loads(str(raw))
        if (type(m) is not dict or m.get('experiment_id') != experiment_id
                or not experiment_id or type(m.get('schema')) is not int or m['schema'] != 1
                or m.get('encoding') != ENCODING or m.get('masks') != list(MASKS)
                or m.get('real_target_version') != 'real-v1'):
            raise ValueError
        metadata_keys = {'schema', 'experiment_id', 'encoding', 'masks', 'train_small',
                         'rejected_duplicates', 'train_hash', 'dev_hash', 'test_orbits_hash',
                         'real_target_version', 'real_train_target_hash', 'real_dev_target_hash'}
        array_keys = {'train', 'dev', 'test', 'train_labels', 'dev_labels'}
        has_null = 'null_version' in m
        has_test = 'test_hash' in m
        if has_null:
            metadata_keys |= {'null_version', 'null_train_hash', 'null_dev_hash',
                              'null_train_record', 'null_dev_record'}
            array_keys |= {'null_train', 'null_dev'}
        if has_test:
            metadata_keys |= {'test_hash', 'selected_sha256'}
            array_keys.add('test_labels')
        if set(arrays) != array_keys or set(m) != metadata_keys:
            raise ValueError
        for name in ('train', 'dev', 'test'):
            a = arrays[name]
            if (a.dtype != np.uint64 or a.ndim != 1 or len(a) == 0 or len(a) % 4
                    or np.any(a & np.uint64(64))):
                raise ValueError
        joined = np.concatenate([arrays[n] for n in ('train', 'dev', 'test')])
        if len(np.unique(joined)) != len(joined):
            raise ValueError
        if (type(m['train_small']) is not int or not 0 < m['train_small'] <= len(arrays['train'])
                or m['train_small'] % 4 or type(m['rejected_duplicates']) is not int
                or m['rejected_duplicates'] < 0):
            raise ValueError
        for split in ('train', 'dev', 'test'):
            name = split + '_labels'
            if name not in arrays:
                continue
            y = arrays[name]
            if y.dtype != np.uint8 or y.shape != (len(arrays[split]), 4) or np.any(y > 1):
                raise ValueError
            digest = hashlib.sha256((_array_digest(arrays[split]) + _array_digest(y))
                                    .encode('ascii')).hexdigest()
            if digest != m[split + '_hash']:
                raise ValueError
        if _array_digest(arrays['test']) != m['test_orbits_hash']:
            raise ValueError
        for split in ('train', 'dev'):
            real = arrays[split + '_labels']
            if _array_digest(real) != m['real_' + split + '_target_hash']:
                raise ValueError
            if has_null:
                target = arrays['null_' + split]
                record = m['null_' + split + '_record']
                if (m['null_version'] != 'row-permutation-v1' or target.dtype != np.uint8
                        or target.shape != real.shape or np.any(target > 1)
                        or _array_digest(target) != m['null_' + split + '_hash']):
                    raise ValueError
                w = np.array([1, 2, 4, 8], dtype=np.uint8)
                if not np.array_equal(np.bincount(real @ w, minlength=16),
                                      np.bincount(target @ w, minlength=16)):
                    raise ValueError
                if (type(record) is not dict
                        or set(record) != {'seed', 'permutation_sha256',
                                           'fixed_rows', 'coincident_rows'}
                        or type(record['seed']) is not int or not _sha(record['permutation_sha256'])
                        or type(record['fixed_rows']) is not int
                        or type(record['coincident_rows']) is not int
                        or not 0 <= record['fixed_rows'] <= record['coincident_rows'] <= len(real)
                        or record['coincident_rows']
                        != int(np.sum(np.all(real == target, axis=1)))):
                    raise ValueError
        if has_test:
            selected = m['selected_sha256']
            if (type(selected) is not dict or set(selected) != {f'F{i}' for i in range(1, 6)}
                    or not all(_sha(v) for v in selected.values())
                    or np.all(arrays['train_labels'] == arrays['train_labels'][0])):
                raise ValueError
        for a in arrays.values():
            a.setflags(write=False)
        return SavedDataset(arrays['train'], arrays['dev'], arrays['test'],
                            arrays['train_labels'], arrays['dev_labels'], arrays.get('null_train'),
                            arrays.get('null_dev'), arrays.get('test_labels'), m)
    except Exception:
        raise ValueError('DATA_SCHEMA_OR_HASH_MISMATCH') from None

"""Private label boundary, canonical orbit encoding and compact verified caches."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .config import ENCODING, MASKS, Recipe

Array = NDArray[Any]
U64 = NDArray[np.uint64]
U8 = NDArray[np.uint8]
F32 = NDArray[np.float32]
_FITS = frozenset(f'F{i}' for i in range(1, 6))


def _sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _experiment(value: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError('INVALID_EXPERIMENT_ID')


def canonical_orbit(p: int) -> int:
    if type(p) is not int or not 0 <= p < 1 << 64:
        raise ValueError('INVALID_WORD')
    return min(p, p ^ 64)


def _orbits(values: Array, *, unique: bool = True) -> U64:
    if (not isinstance(values, np.ndarray) or values.dtype != np.uint64 or values.ndim != 1
            or len(values) == 0 or np.any(values & np.uint64(64))):
        raise ValueError('INVALID_CANONICAL_ORBITS')
    if unique and len(np.unique(values)) != len(values):
        raise ValueError('DUPLICATE_ORBIT')
    return values


def _labels(values: Array, rows: int) -> U8:
    if (not isinstance(values, np.ndarray) or values.dtype != np.uint8
            or values.shape != (rows, 4) or np.any(values > 1)):
        raise ValueError('INVALID_LABELS')
    return values


def _readonly(values: Array) -> Array:
    result = values.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SplitSizes:
    train_small: int
    train_large: int
    dev: int
    test: int

    def __post_init__(self) -> None:
        if (any(type(v) is not int or v <= 0 or v % 4 for v in
                (self.train_small, self.train_large, self.dev, self.test))
                or self.train_small > self.train_large):
            raise ValueError('INVALID_SPLIT_SIZES')

    @classmethod
    def from_recipe(cls, recipe: Recipe) -> SplitSizes:
        return cls(recipe.train_small, recipe.train_large, recipe.dev, recipe.test)


@dataclass(frozen=True, slots=True)
class OrbitSplits:
    train: U64
    dev: U64
    test: U64
    train_small: int
    rejected_duplicates: int = 0

    def __post_init__(self) -> None:
        SplitSizes(self.train_small, len(self.train), len(self.dev), len(self.test))
        if type(self.rejected_duplicates) is not int or self.rejected_duplicates < 0:
            raise ValueError('INVALID_REJECTION_COUNT')
        for name in ('train', 'dev', 'test'):
            object.__setattr__(self, name, _readonly(_orbits(getattr(self, name))))
        joined = np.concatenate((self.train, self.dev, self.test))
        if len(np.unique(joined)) != len(joined):
            raise ValueError('CROSS_SPLIT_ORBIT')

    @property
    def small(self) -> U64:
        return self.train[:self.train_small]


def sample_orbits(sizes: SplitSizes, *, seed: int = 20261116) -> OrbitSplits:
    rng = np.random.default_rng(seed)  # Public sampling ONLY. Never used for the private key.
    n = sizes.train_large + sizes.dev + sizes.test
    seen: set[int] = set()
    result: list[int] = []
    rejected = 0
    while len(result) < n:
        words = rng.bit_generator.random_raw(min(n - len(result), 65536))
        for raw in words:
            p = int(raw) & ~64
            if p in seen:
                rejected += 1
            else:
                seen.add(p)
                result.append(p)
    a = np.asarray(result, dtype=np.uint64)
    end = sizes.train_large + sizes.dev
    return OrbitSplits(a[:sizes.train_large], a[sizes.train_large:end], a[end:],
                       sizes.train_small, rejected)


def encode_pairs(orbits: U64) -> F32:
    _orbits(orbits)
    if len(orbits) % 4:
        raise ValueError('INCOMPLETE_GROUP')
    pairs = np.stack((orbits, orbits ^ np.uint64(64)), axis=1)
    shifts = np.arange(63, -1, -1, dtype=np.uint64)
    return ((pairs[..., None] >> shifts) & np.uint64(1)).astype(np.float32).reshape(-1, 4, 2, 64)


def relation_labels(differences: U64) -> U8:
    if differences.dtype != np.uint64 or differences.ndim != 1:
        raise ValueError('INVALID_DIFFERENCES')
    result = np.zeros((len(differences), 4), dtype=np.uint8)
    for j, mask in enumerate(MASKS):
        for bit in range(64):
            if mask >> bit & 1:
                result[:, j] ^= ((differences >> np.uint64(bit)) & np.uint64(1)).astype(np.uint8)
    return result


def array_sha256(array: Array) -> str:
    if array.dtype.hasobject:
        raise ValueError('OBJECT_ARRAY_FORBIDDEN')
    # Normalize byte order for Windows/Linux portable identities.
    a = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder('<'))
    h = hashlib.sha256()
    h.update(json.dumps([a.dtype.str, list(a.shape)], separators=(',', ':')).encode('ascii'))
    h.update(a.tobytes())
    return h.hexdigest()


def content_sha256(orbits: U64, labels: U8) -> str:
    _orbits(orbits)
    _labels(labels, len(orbits))
    return hashlib.sha256((array_sha256(orbits) + array_sha256(labels)).encode('ascii')).hexdigest()


class Oracle(Protocol):
    def __call__(self, block: int, key: object) -> int: ...


class BatchOracle(Protocol):
    def __call__(self, blocks: tuple[int, ...], key: object) -> tuple[int, ...]: ...


class BlockMeter(Protocol):
    def reserve(self, kind: Literal['real_blocks'], amount: int) -> None: ...
    def complete(self, kind: Literal['real_blocks'], amount: int) -> None: ...


@dataclass(frozen=True, slots=True)
class FrozenFits:
    experiment_id: str
    selected_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        _experiment(self.experiment_id)
        if (set(self.selected_sha256) != _FITS
                or not all(_sha(v) for v in self.selected_sha256.values())):
            raise ValueError('ALL_FIVE_FITS_MUST_BE_FROZEN')
        object.__setattr__(self, 'selected_sha256', MappingProxyType(dict(self.selected_sha256)))


@dataclass(frozen=True, slots=True)
class PreparedData:
    experiment_id: str
    splits: OrbitSplits
    train_labels: U8
    dev_labels: U8

    def __post_init__(self) -> None:
        _experiment(self.experiment_id)
        object.__setattr__(self, 'train_labels',
                           _readonly(_labels(self.train_labels, len(self.splits.train))))
        object.__setattr__(self, 'dev_labels',
                           _readonly(_labels(self.dev_labels, len(self.splits.dev))))

    @property
    def train_hash(self) -> str:
        return content_sha256(self.splits.train, self.train_labels)

    @property
    def dev_hash(self) -> str:
        return content_sha256(self.splits.dev, self.dev_labels)

    @property
    def all_train_constant(self) -> bool:
        return bool(np.all(self.train_labels == self.train_labels[0]))


class DataSession:
    """Label-side private state only; caller supplies an independently generated key.

    An oracle failure poisons the session: there is no automatic retry. Meter
    exceptions retain their public stop code; private oracle errors are redacted.
    """

    def __init__(self, experiment_id: str, splits: OrbitSplits, *, key: object,
                 meter: BlockMeter, oracle: Oracle | None = None,
                 batch_oracle: BatchOracle | None = None) -> None:
        _experiment(experiment_id)
        if (oracle is None) == (batch_oracle is None):
            raise ValueError('EXACTLY_ONE_ORACLE_REQUIRED')
        self._batch_oracle = batch_oracle
        self.experiment_id = experiment_id
        self.splits = splits
        self._key = key
        self._oracle = oracle
        self._meter = meter
        self._active = True
        self._prepared: PreparedData | None = None
        self._test: U8 | None = None
        self._frozen: FrozenFits | None = None

    def _generate(self, orbits: U64) -> U8:
        if not self._active:
            raise ValueError('PRIVATE_STATE_UNAVAILABLE')
        differences = np.empty(len(orbits), dtype=np.uint64)
        try:
            for start in range(0, len(orbits), 1024):
                chunk = orbits[start:start + 1024]
                if self._batch_oracle is not None:
                    for _ in chunk:
                        self._meter.reserve('real_blocks', 2)
                    blocks = tuple(v for p in chunk for v in (int(p), int(p) ^ 64))
                    try:
                        outputs = self._batch_oracle(blocks, self._key)
                        if (type(outputs) is not tuple or len(outputs) != len(blocks)
                                or any(type(v) is not int or not 0 <= v < 1 << 64
                                       for v in outputs)):
                            raise ValueError
                    except Exception:
                        raise ValueError('ORACLE_FAILED') from None
                    for offset in range(len(chunk)):
                        self._meter.complete('real_blocks', 2)
                        differences[start + offset] = outputs[2 * offset] ^ outputs[2 * offset + 1]
                else:
                    assert self._oracle is not None
                    for offset, p in enumerate(chunk):
                        self._meter.reserve('real_blocks', 2)
                        try:
                            c = self._oracle(int(p), self._key)
                            other = self._oracle(int(p) ^ 64, self._key)
                            if any(type(v) is not int or not 0 <= v < 1 << 64 for v in (c, other)):
                                raise ValueError
                        except Exception:
                            raise ValueError('ORACLE_FAILED') from None
                        self._meter.complete('real_blocks', 2)
                        differences[start + offset] = c ^ other
        except BaseException:
            self.release_key()
            raise
        result = relation_labels(differences)
        result.setflags(write=False)
        return result

    def prepare(self) -> PreparedData:
        if self._prepared is None:
            self._prepared = PreparedData(self.experiment_id, self.splits,
                                         self._generate(self.splits.train),
                                         self._generate(self.splits.dev))
        return self._prepared

    def test_labels(self, token: FrozenFits) -> U8:
        if not isinstance(token, FrozenFits) or token.experiment_id != self.experiment_id:
            raise ValueError('FROZEN_IDENTITY_MISMATCH')
        if self._prepared is None:
            raise ValueError('TRAIN_DEV_REQUIRED')
        if self._prepared.all_train_constant:
            raise ValueError('ALL_TRAIN_RELATIONS_CONSTANT')
        if self._frozen is not None and self._frozen != token:
            raise ValueError('FROZEN_SELECTION_CHANGED')
        if self._test is None:
            self._test = self._generate(self.splits.test)
            self._frozen = token
        return self._test

    def release_key(self) -> None:
        self._key = None
        self._active = False


@dataclass(frozen=True, slots=True)
class NullTargets:
    train: U8
    dev: U8
    train_hash: str
    dev_hash: str
    train_record: dict[str, object]
    dev_record: dict[str, object]
    version: str = 'row-permutation-v1'


def make_null_targets(train_labels: U8, dev_labels: U8, *, train_seed: int = 41,
                      dev_seed: int = 43) -> NullTargets:
    def permute(labels: U8, seed: int) -> tuple[U8, dict[str, object]]:
        _labels(labels, len(labels))
        order = np.random.default_rng(seed).permutation(len(labels)).astype(np.int64)
        result = labels[order].copy()
        result.setflags(write=False)
        return result, {'seed': seed, 'permutation_sha256': array_sha256(order),
                        'fixed_rows': int(np.sum(order == np.arange(len(labels)))),
                        'coincident_rows': int(np.sum(np.all(result == labels, axis=1)))}
    train, tr = permute(train_labels, train_seed)
    dev, dr = permute(dev_labels, dev_seed)
    return NullTargets(train, dev, array_sha256(train), array_sha256(dev), tr, dr)


def _metadata(prepared: PreparedData, null: NullTargets | None, test: U8 | None,
              frozen: FrozenFits | None) -> dict[str, object]:
    p = prepared
    meta: dict[str, object] = {
        'schema': 1, 'experiment_id': p.experiment_id, 'encoding': ENCODING,
        'masks': list(MASKS), 'train_small': p.splits.train_small,
        'rejected_duplicates': p.splits.rejected_duplicates,
        'train_hash': p.train_hash, 'dev_hash': p.dev_hash,
        'test_orbits_hash': array_sha256(p.splits.test),
        'real_target_version': 'real-v1',
        'real_train_target_hash': array_sha256(p.train_labels),
        'real_dev_target_hash': array_sha256(p.dev_labels),
    }
    if null is not None:
        if null.version != 'row-permutation-v1':
            raise ValueError('NULL_VERSION_MISMATCH')
        for real, target, digest, record in (
                (p.train_labels, null.train, null.train_hash, null.train_record),
                (p.dev_labels, null.dev, null.dev_hash, null.dev_record)):
            _labels(target, len(real))
            if digest != array_sha256(target):
                raise ValueError('NULL_HASH_MISMATCH')
            weights = np.array([1, 2, 4, 8], dtype=np.uint8)
            if not np.array_equal(np.bincount(real @ weights, minlength=16),
                                  np.bincount(target @ weights, minlength=16)):
                raise ValueError('NULL_JOINT_FREQUENCIES_MISMATCH')
            if (set(record) != {'seed', 'permutation_sha256', 'fixed_rows', 'coincident_rows'}
                    or type(record['seed']) is not int
                    or not _sha(record['permutation_sha256'])
                    or type(record['fixed_rows']) is not int
                    or type(record['coincident_rows']) is not int
                    or not 0 <= record['fixed_rows'] <= record['coincident_rows'] <= len(real)
                    or record['coincident_rows'] != int(np.sum(np.all(real == target, axis=1)))):
                raise ValueError('NULL_RECORD_MISMATCH')
        meta.update(null_version=null.version, null_train_hash=null.train_hash,
                    null_dev_hash=null.dev_hash, null_train_record=null.train_record,
                    null_dev_record=null.dev_record)
    if test is not None:
        if (frozen is None or frozen.experiment_id != p.experiment_id
                or p.all_train_constant):
            raise ValueError('TEST_REQUIRES_FROZEN_FITS')
        _labels(test, len(p.splits.test))
        meta.update(test_hash=content_sha256(p.splits.test, test),
                    selected_sha256=dict(frozen.selected_sha256))
    elif frozen is not None:
        raise ValueError('FROZEN_WITHOUT_TEST')
    return meta


def save_dataset(path: Path, prepared: PreparedData, *, null: NullTargets | None = None,
                 test_labels: U8 | None = None,
                 frozen: FrozenFits | None = None) -> dict[str, object]:
    meta = _metadata(prepared, null, test_labels, frozen)
    arrays: dict[str, Array] = {
        'train': prepared.splits.train, 'dev': prepared.splits.dev, 'test': prepared.splits.test,
        'train_labels': prepared.train_labels, 'dev_labels': prepared.dev_labels,
        'metadata': np.asarray(json.dumps(meta, sort_keys=True, separators=(',', ':'))),
    }
    if null is not None:
        arrays.update(null_train=null.train, null_dev=null.dev)
    if test_labels is not None:
        arrays['test_labels'] = test_labels
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
    return meta


def load_dataset(path: Path, *, experiment_id: str
                 ) -> tuple[PreparedData, NullTargets | None, U8 | None, dict[str, object]]:
    # Training convenience wrapper; evaluate imports checkpoint.load_saved_dataset directly.
    from .checkpoint import load_saved_dataset

    saved = load_saved_dataset(path, experiment_id=experiment_id)
    m = saved.metadata
    splits = OrbitSplits(saved.train, saved.dev, saved.test, m['train_small'],
                         m['rejected_duplicates'])
    p = PreparedData(experiment_id, splits, saved.train_labels, saved.dev_labels)
    null = None
    if saved.null_train is not None and saved.null_dev is not None:
        null = NullTargets(saved.null_train, saved.null_dev, m['null_train_hash'],
                           m['null_dev_hash'], m['null_train_record'], m['null_dev_record'],
                           m['null_version'])
    return p, null, saved.test_labels, m

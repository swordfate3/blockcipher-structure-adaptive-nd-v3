"""Fresh balanced single-pair ciphertext batches. No key/plaintext leaves the provider."""
from __future__ import annotations

import io
import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..config import Recipe, Task, Training, canonical
from ..data import BlockMeter, DataError, Oracle, PrivateSource, derive_seed
from .config import TASK, SingleRecipe, estimate

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, eq=False)
class PairBatch:
    ciphertext: NDArray[np.uint8]
    labels: NDArray[np.uint8]
    split: str
    index: int
    namespace: str

    def __post_init__(self) -> None:
        x, y = self.ciphertext, self.labels
        if (x.ndim != 4 or x.shape[1:] != (1, 2, 8) or not x.shape[0] or x.shape[0] % 4
                or x.dtype != np.uint8 or y.shape != (x.shape[0], 1) or y.dtype != np.uint8
                or int(y.sum()) != x.shape[0] // 2 or not np.isin(y, (0, 1)).all()
                or self.split not in ('train', 'dev', 'test')
                or type(self.index) is not int or self.index < 0):
            raise ValueError('PAIR_BATCH')
        for name, array in (('ciphertext', x), ('labels', y)):
            object.__setattr__(self, name, np.frombuffer(array.tobytes(), np.uint8).reshape(
                array.shape))

    @property
    def groups(self) -> int:
        return int(self.labels.shape[0])

    @property
    def digest(self) -> str:
        h = sha256(canonical([self.namespace, self.split, self.index, self.groups]).encode('ascii'))
        h.update(self.ciphertext)
        h.update(self.labels)
        return h.hexdigest()

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        import torch
        bits = np.unpackbits(self.ciphertext, axis=-1, bitorder='big').astype(np.float32)
        return torch.from_numpy(bits), torch.from_numpy(self.labels.astype(np.float32))

    def slice(self, start: int, stop: int) -> PairBatch:
        if not 0 <= start < stop <= self.groups:
            raise ValueError('PAIR_SLICE')
        return PairBatch(self.ciphertext[start:stop], self.labels[start:stop],
                         self.split, self.index, self.namespace)


class Provider:
    def __init__(self, recipe: SingleRecipe, oracle: Oracle, private: PrivateSource,
                 meter: BlockMeter, *, streams: Recipe, task: Task = TASK,
                 progress: Callable[[int], None] | None = None) -> None:
        self.recipe, self.oracle, self.private, self.meter = recipe, oracle, private, meter
        self._streams = streams
        self.task = task
        self._progress = progress
        self._cycle = 0
        self._panels: set[str] = set()
        self._stopped = False

    def train(self, cycle: int) -> PairBatch:
        tr = self.recipe.training
        if (type(cycle) is not int or cycle != self._cycle
                or cycle >= tr.epochs * tr.batches_per_epoch):
            raise ValueError('TRAIN_ORDER')
        batch = self._collect('train', cycle, tr.batch_groups)
        self._cycle += 1
        return batch

    def panel(self, split: str) -> PairBatch:
        tr = self.recipe.training
        if split not in ('dev', 'test') or split in self._panels:
            raise ValueError('PANEL_ORDER')
        self._panels.add(split)
        size = tr.dev_groups if split == 'dev' else tr.test_groups
        parts = []
        for index in range(size // tr.eval_batch_groups):
            parts.append(self._collect(split, index, tr.eval_batch_groups))
            if self._progress is not None:
                self._progress(index + 1)
        return PairBatch(np.concatenate([b.ciphertext for b in parts]),
                         np.concatenate([b.labels for b in parts]), split, 0,
                         self._streams.namespace)

    def _collect(self, split: str, index: int, size: int) -> PairBatch:
        if self._stopped:
            raise DataError('PROVIDER_STOPPED')
        try:
            return self._generate(split, index, size)
        except BaseException:
            self._stopped = True
            raise

    def _generate(self, split: str, index: int, size: int) -> PairBatch:
        public = random.Random(derive_seed(self._streams, self.task, split, index, 'sample'))
        shuffle = random.Random(derive_seed(self._streams, self.task, split, index, 'shuffle'))
        labels = [0] * (size // 2) + [1] * (size // 2)
        shuffle.shuffle(labels)
        packed = np.empty((size, 1, 2, 8), dtype=np.uint8)
        for row, label in enumerate(labels):
            first = public.getrandbits(64)
            second = first ^ self.task.difference if label else public.getrandbits(64)
            self.meter.reserve('real_blocks', 2)
            try:
                raw = self.private(self.task.key_bytes)
                if type(raw) is not bytes or len(raw) != self.task.key_bytes:
                    raise ValueError
                integer = int.from_bytes(raw, 'big')
                key = tuple((integer >> shift) & 0xFFFFFFFF
                            for shift in range(0, self.task.key_bytes * 8, 32))
            except Exception:
                raise DataError('PRIVATE_SOURCE_FAILED') from None
            try:
                words = self.oracle((first, second), key, self.task.rounds)
            except Exception:
                raise DataError('ORACLE_FAILED') from None
            try:
                a, b = words
                if any(type(w) is not int or not 0 <= w < 2**64 for w in (a, b)):
                    raise ValueError
                packed[row, 0] = np.frombuffer(a.to_bytes(8, 'big') + b.to_bytes(8, 'big'),
                                               dtype=np.uint8).reshape(2, 8)
            except Exception:
                raise DataError('ORACLE_RESPONSE') from None
            self.meter.complete('real_blocks', 2)
            self.meter.check()
        return PairBatch(packed, np.asarray(labels, dtype=np.uint8).reshape(-1, 1), split, index,
                         self._streams.namespace)


class FixedPairReplay:
    """The existing per-class epoch shuffle, without a sampler or oracle."""

    def __init__(self, train: PairBatch, training: Training, streams: Recipe, task: Task,
                 shuffle_namespace: str, check: Callable[[], None]) -> None:
        self._train, self._training, self._task, self._check = train, training, task, check
        self._streams = replace(streams, namespace=shuffle_namespace)
        self._classes = (np.flatnonzero(train.labels == 0), np.flatnonzero(train.labels == 1))
        check()
        self._cycle = 0
        self._epoch: tuple[np.random.Generator, NDArray[np.intp], NDArray[np.intp]] | None = None

    def train(self, cycle: int) -> PairBatch:
        tr = self._training
        if (type(cycle) is not int or cycle != self._cycle
                or cycle >= tr.epochs * tr.batches_per_epoch):
            raise ValueError('TRAIN_ORDER')
        epoch, offset = divmod(cycle, tr.batches_per_epoch)
        if offset == 0:
            # Shuffle row indices only, independently within each class.
            rng = np.random.default_rng(derive_seed(
                self._streams, self._task, 'train', epoch, 'shuffle'))
            self._epoch = rng, rng.permutation(self._classes[0]), rng.permutation(self._classes[1])
            self._check()
        assert self._epoch is not None
        rng, zeros, ones = self._epoch
        half = tr.batch_groups // 2
        start = offset * half
        rows = np.concatenate((zeros[start:start + half], ones[start:start + half]))
        rng.shuffle(rows)
        batch = PairBatch(self._train.ciphertext[rows], self._train.labels[rows],
                          'train', cycle, self._train.namespace)
        self._cycle += 1
        return batch


class FixedPairPool(Provider):
    """Collect the whole balanced pool once, then replay fixed training rows."""

    def __init__(self, recipe: SingleRecipe, oracle: Oracle, private: PrivateSource,
                 meter: BlockMeter, *, streams: Recipe, task: Task,
                 shuffle_namespace: str, progress: Callable[[int], None] | None = None) -> None:
        if not recipe.fixed_pool:
            raise ValueError('POOL_RECIPE')
        super().__init__(recipe, oracle, private, meter, streams=streams, task=task,
                         progress=progress)
        self._shuffle_namespace = shuffle_namespace
        self._started = self._ready = False
        self._replay: FixedPairReplay | None = None
        self._dev: PairBatch | None = None

    def prepare_pool(self) -> PairBatch:
        if self._started or self.recipe.reuse_pool:
            raise ValueError('POOL_ORDER')
        self._started = True
        totals = estimate(self.recipe)
        size = self.recipe.training.batch_groups
        groups = totals['pool_groups']
        ns = self._streams.namespace
        x = np.empty((groups, 1, 2, 8), dtype=np.uint8)
        y = np.empty((groups, 1), dtype=np.uint8)
        for i in range(groups // size):
            part = self._collect('train', i, size)
            if (part.split, part.index, part.groups, part.namespace) != ('train', i, size, ns):
                raise ValueError('POOL_BATCH_IDENTITY')
            start = i * size
            x[start:start + size], y[start:start + size] = part.ciphertext, part.labels
            self.meter.check()
            if self._progress is not None:
                self._progress(i + 1)
        # One whole balanced chunk per ten to DEV. No split before collection ends.
        x_blocks = x.reshape(-1, 10, size, 1, 2, 8)
        y_blocks = y.reshape(-1, 10, size, 1)
        train = PairBatch(x_blocks[:, :9].reshape(-1, 1, 2, 8),
                          y_blocks[:, :9].reshape(-1, 1), 'train', 0, ns)
        self.meter.check()
        dev_y = y_blocks[:, 9].reshape(-1, 1)
        dev_order = np.column_stack((np.flatnonzero(dev_y == 0),
                                     np.flatnonzero(dev_y == 1))).ravel()
        self._dev = PairBatch(x_blocks[:, 9].reshape(-1, 1, 2, 8)[dev_order],
                              dev_y[dev_order], 'dev', 0, ns)
        # DEV now satisfies PairBatch quotas for every even-aligned slice,
        # including the 64-row tail of the production 512-row evaluation loop.
        return self._use_train(train)

    def _use_train(self, train: PairBatch) -> PairBatch:
        self._replay = FixedPairReplay(train, self.recipe.training, self._streams, self.task,
                                       self._shuffle_namespace, self.meter.check)
        return train

    def activate(self) -> None:
        """The loop calls this only after committing both split artifacts and READY."""
        if self._replay is None or 'dev' not in self._panels or self._ready:
            raise ValueError('POOL_ORDER')
        self._ready = True

    def panel(self, split: str) -> PairBatch:
        if split == 'dev':
            if self._dev is None or 'dev' in self._panels:
                raise ValueError('PANEL_ORDER')
            self._panels.add(split)
            return self._dev
        if not self._ready:
            raise ValueError('POOL_NOT_READY')
        return super().panel(split)

    def train(self, cycle: int) -> PairBatch:
        if not self._ready or self._replay is None:
            raise ValueError('POOL_NOT_READY')
        return self._replay.train(cycle)


def pack_panel(batch: PairBatch, recipe: SingleRecipe, metadata: dict[str, Any],
               slots: tuple[str, str], columns: tuple[str, str], namespace: str, *,
               logits: dict[str, Any] | None, weights: dict[str, Any] | None) -> bytes:
    has_scores = batch.split == 'test'
    size = recipe.training.test_groups if has_scores else recipe.training.dev_groups
    allowed: tuple[str, ...] = ('dev', 'test')
    if recipe.fixed_pool and batch.split == 'train':
        allowed = ('train',)
        size = recipe.training.batches_per_epoch * recipe.training.batch_groups
    if (batch.split not in allowed or batch.index != 0 or batch.groups != size
            or batch.namespace != namespace
            or (logits is not None) != has_scores or (weights is not None) != has_scores):
        raise ValueError('PANEL_IDENTITY')
    stream = io.BytesIO()
    arrays: dict[str, Any] = dict(ciphertext=batch.ciphertext, labels=batch.labels)
    if logits is not None and weights is not None:
        if set(logits) != set(slots) or set(weights) != set(slots):
            raise ValueError('PANEL_BINDING')
        metadata = {**metadata, 'weights': weights}
        arrays.update({column: logits[slot] for slot, column in zip(slots, columns, strict=True)})
    np.savez(stream, metadata=np.frombuffer(canonical(metadata).encode('ascii'), dtype=np.uint8),
             **arrays)
    return stream.getvalue()

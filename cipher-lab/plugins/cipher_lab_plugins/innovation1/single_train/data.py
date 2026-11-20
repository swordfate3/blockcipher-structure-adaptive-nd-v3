"""Fresh balanced single-pair ciphertext batches. No key/plaintext leaves the provider."""
from __future__ import annotations

import io
import random
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..config import Recipe, canonical
from ..data import BlockMeter, DataError, Oracle, PrivateSource, derive_seed
from .config import TASK, SingleRecipe

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
                 meter: BlockMeter, *, streams: Recipe) -> None:
        self.recipe, self.oracle, self.private, self.meter = recipe, oracle, private, meter
        self._streams = streams
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
        parts = [self._collect(split, index, tr.eval_batch_groups)
                 for index in range(size // tr.eval_batch_groups)]
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
        public = random.Random(derive_seed(self._streams, TASK, split, index, 'sample'))
        shuffle = random.Random(derive_seed(self._streams, TASK, split, index, 'shuffle'))
        labels = [0] * (size // 2) + [1] * (size // 2)
        shuffle.shuffle(labels)
        packed = np.empty((size, 1, 2, 8), dtype=np.uint8)
        for row, label in enumerate(labels):
            first = public.getrandbits(64)
            second = first ^ TASK.difference if label else public.getrandbits(64)
            self.meter.reserve('real_blocks', 2)
            try:
                raw = self.private(TASK.key_bytes)
                if type(raw) is not bytes or len(raw) != 8:
                    raise ValueError
                integer = int.from_bytes(raw, 'big')
                key = (integer & 0xFFFFFFFF, integer >> 32)
            except Exception:
                raise DataError('PRIVATE_SOURCE_FAILED') from None
            try:
                words = self.oracle((first, second), key, TASK.rounds)
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


def pack_panel(batch: PairBatch, recipe: SingleRecipe, metadata: dict[str, Any],
               slots: tuple[str, str], columns: tuple[str, str], namespace: str, *,
               logits: dict[str, Any] | None, weights: dict[str, Any] | None) -> bytes:
    has_scores = batch.split == 'test'
    size = recipe.training.test_groups if has_scores else recipe.training.dev_groups
    if (batch.split not in ('dev', 'test') or batch.index != 0 or batch.groups != size
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

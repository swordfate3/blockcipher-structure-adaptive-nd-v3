"""Explicit layout sampling stream and saved-panel identity; no new sampler."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..data import BlockMeter, Oracle, PrivateSource, derive_seed
from ..single_train.config import BACKEND, TASK, SingleRecipe, estimate
from ..single_train.data import PairBatch, pack_panel
from ..single_train.data import Provider as SingleProvider
from .config import PUBLIC_STREAMS, SLOTS, LayoutRecipe, official


class Provider(SingleProvider):
    recipe: LayoutRecipe

    def __init__(self, recipe: LayoutRecipe, oracle: Oracle, private: PrivateSource,
                 meter: BlockMeter) -> None:
        super().__init__(recipe, oracle, private, meter,
                         streams=replace(PUBLIC_STREAMS, namespace=recipe.namespace))


class FixedPoolProvider(Provider):
    """Collect the whole balanced pool once, then replay fixed training rows."""

    def __init__(self, recipe: LayoutRecipe, oracle: Oracle, private: PrivateSource,
                 meter: BlockMeter) -> None:
        if not recipe.fixed_pool:
            raise ValueError('POOL_RECIPE')
        super().__init__(recipe, oracle, private, meter)
        self._started = self._ready = False
        self._train_pool: PairBatch | None = None
        self._dev: PairBatch | None = None
        self._classes: tuple[NDArray[np.intp], NDArray[np.intp]] | None = None
        self._epoch: tuple[np.random.Generator, NDArray[np.intp], NDArray[np.intp]] | None = None

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
        self._classes = (np.flatnonzero(train.labels == 0), np.flatnonzero(train.labels == 1))
        self.meter.check()
        self._train_pool = train
        return train

    def activate(self) -> None:
        """The loop calls this only after committing both split artifacts and READY."""
        if self._train_pool is None or 'dev' not in self._panels or self._ready:
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
        train, classes = self._train_pool, self._classes
        if not self._ready or train is None or classes is None:
            raise ValueError('POOL_NOT_READY')
        tr = self.recipe.training
        if (type(cycle) is not int or cycle != self._cycle
                or cycle >= tr.epochs * tr.batches_per_epoch):
            raise ValueError('TRAIN_ORDER')
        epoch, offset = divmod(cycle, tr.batches_per_epoch)
        if offset == 0:
            # Shuffle row indices only, independently within each class.
            streams = replace(self._streams, namespace=self.recipe.shuffle_namespace)
            rng = np.random.default_rng(derive_seed(streams, TASK, 'train', epoch, 'shuffle'))
            self._epoch = rng, rng.permutation(classes[0]), rng.permutation(classes[1])
            self.meter.check()
        assert self._epoch is not None
        rng, zeros, ones = self._epoch
        half = tr.batch_groups // 2
        start = offset * half
        rows = np.concatenate((zeros[start:start + half], ones[start:start + half]))
        rng.shuffle(rows)
        batch = PairBatch(train.ciphertext[rows], train.labels[rows],
                          'train', cycle, train.namespace)
        self._cycle += 1
        return batch


def encode_panel(batch: PairBatch, recipe: SingleRecipe, source: dict[str, str],
                 backend: str, *, logits: dict[str, Any] | None = None,
                 weights: dict[str, Any] | None = None,
                 reuse_provenance: dict[str, Any] | None = None) -> bytes:
    if type(recipe) is not LayoutRecipe or recipe.run_id is None:
        raise ValueError('PANEL_IDENTITY')
    if backend == BACKEND:
        official(recipe, recipe.run_id)
    metadata = dict(schema=recipe.artifact_schema, method=recipe.method,
                    run_id=recipe.run_id, recipe=recipe.to_dict(), source=source, backend=backend,
                    split=batch.split, digest=batch.digest, encoding='packed-msb0-single-pair-v1',
                    slots=list(SLOTS))
    if recipe.reuse_pool:
        if batch.split != 'test' or reuse_provenance is None:
            raise ValueError('PANEL_IDENTITY')
        metadata.update(namespace=recipe.namespace, reused_pool=reuse_provenance)
    return pack_panel(batch, recipe, metadata, SLOTS, ('a_logits', 'a_prime_logits'),
                      recipe.namespace, logits=logits, weights=weights)

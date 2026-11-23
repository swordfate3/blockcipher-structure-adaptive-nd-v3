"""Bind the old pool without rewriting it; encode only public single-pair artifacts."""
from __future__ import annotations

import io
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import Task, canonical
from ..data import BlockMeter, Oracle, PrivateSource
from ..layout_probe.config import FIXED_BCE, LayoutRecipe
from ..layout_probe.reuse import PoolSourceError, ReusedPoolProvider
from ..single_train.data import FixedPairPool, PairBatch, Provider
from .config import PANEL_OVERHEAD, FourSingleRecipe, ReplayRecipe


def skinny_pool_recipe(recipe: FourSingleRecipe) -> LayoutRecipe:
    if recipe.skinny_source is None:
        raise PoolSourceError
    # Adapt only split geometry (for guarded fixtures), not archive identity.
    t, old = recipe.training, FIXED_BCE.training
    return replace(FIXED_BCE, pool_source=recipe.skinny_source,
                   training=replace(old, batch_groups=t.batch_groups,
                                    batches_per_epoch=t.batches_per_epoch,
                                    dev_groups=t.dev_groups))


class Pools:
    def __init__(self, recipe: FourSingleRecipe, oracles: Mapping[str, Oracle],
                 private: PrivateSource, meter: BlockMeter) -> None:
        self.recipe, self.oracles, self.private, self.meter = recipe, oracles, private, meter
        self.reused = ReusedPoolProvider(skinny_pool_recipe(recipe), oracles['SKINNY'],
                                         private, meter)
        self.fixed: dict[str, FixedPairPool] = {'SKINNY': self.reused}
        self._test: set[str] = set()

    def bind_reused(self) -> tuple[PairBatch, dict[str, Any]]:
        self.reused.prepare_pool()
        panel = self.reused.panel('dev')
        return panel, self.reused.pool_provenance()

    def collect(self, task: Task, progress: Callable[[int], None]) -> tuple[PairBatch, PairBatch]:
        if task not in self.recipe.tasks[:-1] or task.cipher in self.fixed:
            raise ValueError('POOL_ORDER')
        pool = FixedPairPool(self.recipe, self.oracles[task.cipher], self.private, self.meter,
                             streams=self.recipe.streams, task=task,
                             shuffle_namespace=self.recipe.namespace, progress=progress)
        self.fixed[task.cipher] = pool
        train = pool.prepare_pool()
        return train, pool.panel('dev')

    def activate(self) -> None:
        for pool in self.fixed.values():
            pool.activate()

    def train(self, task: Task, cycle: int) -> PairBatch:
        return self.fixed[task.cipher].train(cycle)

    def release_train(self) -> None:
        # No more replay, and do not retain duplicate TEST data across tasks.
        self.fixed.clear()
        del self.reused

    def test(self, task: Task, progress: Callable[[int], None]) -> PairBatch:
        if self.fixed or task.cipher in self._test:
            raise ValueError('TEST_ORDER')
        self._test.add(task.cipher)
        provider = Provider(self.recipe, self.oracles[task.cipher], self.private, self.meter,
                            streams=self.recipe.streams, task=task, progress=progress)
        return provider.panel('test')


def encode_panel(batch: PairBatch, task: Task, recipe: FourSingleRecipe, source: dict[str, str],
                 backend: str, *, logits: NDArray[np.float64] | None = None,
                 best: dict[str, Any] | None = None,
                 reused_input: dict[str, Any] | None = None) -> bytes:
    tr = recipe.training
    counts = dict(train=tr.batches_per_epoch * tr.batch_groups, dev=tr.dev_groups,
                  test=tr.test_groups)
    scored = batch.split == 'test'
    replay = isinstance(recipe, ReplayRecipe)
    if (batch.index != 0 or batch.namespace != recipe.namespace or task not in recipe.tasks
            or batch.groups != counts[batch.split] or (scored != (logits is not None))
            or scored != (best is not None) or replay != (reused_input is not None)
            or (replay and not scored)
            or (task.cipher == 'SKINNY' and not scored)):
        raise ValueError('PANEL_IDENTITY')
    meta = dict(schema=1, method=recipe.method, run_id=recipe.run_id, recipe=recipe.to_dict(),
                backend=backend, source=source, cipher=task.cipher, split=batch.split,
                namespace=batch.namespace, digest=batch.digest, rows=batch.groups,
                encoding='packed-msb0-single-pair-v1')
    if reused_input is not None:
        meta['reused_input'] = reused_input
    arrays: dict[str, Any] = dict(ciphertext=batch.ciphertext, labels=batch.labels)
    if logits is not None:
        if (logits.shape != batch.labels.shape or logits.dtype != np.float64
                or not np.isfinite(logits).all()):
            raise ValueError('TEST_LOGITS')
        arrays['logits'] = logits
        meta['best'] = best
    stream = io.BytesIO()
    np.savez(stream, metadata=np.frombuffer(canonical(meta).encode('ascii'), dtype=np.uint8),
             **arrays)
    raw = stream.getvalue()
    if len(raw) > batch.groups * (25 if scored else 17) + PANEL_OVERHEAD:
        raise ValueError('PANEL_SIZE')
    return raw

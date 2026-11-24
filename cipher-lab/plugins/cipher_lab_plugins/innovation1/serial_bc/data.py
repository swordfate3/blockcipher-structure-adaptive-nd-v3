"""Deterministic balanced index subsets of verified original splits."""
from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import Task
from ..data import derive_seed
from ..four_single.config import ReplayRecipe
from ..four_single.data import _panel_bytes
from ..four_single.replay import ReplayPools
from ..single_train.config import BACKEND
from ..single_train.data import FixedPairReplay, PairBatch
from .config import SerialRecipe, estimate


def index_subset(panel: PairBatch, groups: int, seed: int,
                 namespace: str) -> tuple[PairBatch, NDArray[np.int64]]:
    if (panel.split not in ('train', 'dev') or panel.index != 0 or groups <= 0
            or groups % 4 or groups > panel.groups):
        raise ValueError('SUBSET_GEOMETRY')
    rng = np.random.Generator(np.random.PCG64(seed))
    rows = np.empty(groups, dtype='<i8')
    for label in (0, 1):
        candidates = np.flatnonzero(panel.labels[:, 0] == label)
        rows[label::2] = rng.permutation(candidates)[:groups // 2]
    batch = PairBatch(panel.ciphertext[rows], panel.labels[rows], panel.split, 0, namespace)
    rows.flags.writeable = False
    return batch, rows


class SerialPools(ReplayPools):
    def __init__(self, recipe: SerialRecipe, check: Callable[[], None], *,
                 archive: ReplayRecipe | None = None) -> None:
        # The optional scaled reader exists solely for guarded artificial fixtures.
        super().__init__(recipe.archive_reader_recipe() if archive is None else archive, check)
        self.serial = recipe

    def bind(self, visit: Callable[[str], None]) -> tuple[dict[str, PairBatch], dict[str, Any]]:
        recipe = self.serial
        totals = estimate(recipe)
        dev: dict[str, PairBatch] = {}
        bindings: dict[str, Any] = {}

        def subset(panel: PairBatch, groups: int, task: Task,
                   evidence: dict[str, Any]) -> PairBatch:
            self._check()
            seed = derive_seed(recipe.streams, task, panel.split, 0, 'sample')
            batch, rows = index_subset(panel, groups, seed, recipe.namespace)
            bindings[task.cipher][panel.split] = dict(
                groups=groups, digest=batch.digest, index_sha256=sha256(rows).hexdigest(),
                index_encoding='original-zero-based-le-i64-in-subset-order',
                seed=seed, source_digest=panel.digest, source_groups=panel.groups,
                source_namespace=panel.namespace, source=evidence)
            self._check()
            return batch

        for task, train, valid, evidence in self._read_pools(visit):
            bindings[task.cipher] = {}
            # Whole SKINNY evidence is the original two-panel chain, not a
            # parallel set of substitute pins.
            skinny = task.cipher == 'SKINNY'
            fixed = subset(train, totals['fixed_train_rows_each'], task,
                           evidence if skinny else evidence['train'])
            dev[task.cipher] = subset(valid, totals['fixed_dev_rows_each'], task,
                                      evidence if skinny else evidence['dev'])
            self._fixed[task.cipher] = FixedPairReplay(fixed, recipe.training, recipe.streams,
                                                     task, recipe.namespace, self._check)
            del train, valid, fixed
        self.activate()
        return dev, bindings


def encode_test(batch: PairBatch, task: Task, recipe: SerialRecipe, source: dict[str, str],
                *, logits: NDArray[np.float64], final: dict[str, Any],
                reused_input: dict[str, Any]) -> bytes:
    if ((batch.split, batch.index, batch.groups, batch.namespace) != (
            'test', 0, recipe.training.test_groups, recipe.archive_reader_recipe().namespace)
            or task not in recipe.tasks or final.get('stage') != 4):
        raise ValueError('TEST_PANEL_IDENTITY')
    return _panel_bytes(batch, task, recipe, source, BACKEND, logits=logits, best=final,
                        reused_input=reused_input)

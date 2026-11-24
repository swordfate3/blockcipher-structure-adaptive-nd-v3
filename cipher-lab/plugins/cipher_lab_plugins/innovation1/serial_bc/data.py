"""Deterministic balanced index subsets of verified original splits."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from hashlib import sha256
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import Task
from ..data import derive_seed
from ..four_single.config import ReplayRecipe
from ..four_single.data import _panel_bytes, skinny_pool_recipe
from ..four_single.replay import ReplayPools
from ..layout_probe.reuse import PoolSourceError, load_reused_pool
from ..single_train.config import BACKEND
from ..single_train.data import FixedPairReplay, PairBatch
from .config import IndependentRecipe, SerialRecipe, SubsetPin, VolumeRecipe, estimate


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
        # The guarded fixture seam may scale reader geometry, but it cannot
        # detach the opened archive or SKINNY chain from the fit's recipe.
        reader = recipe.archive_reader_recipe()
        if archive is not None and archive != replace(reader, training=archive.training,
                                                       run_id=archive.run_id):
            raise ValueError('ARCHIVE_RECIPE_IDENTITY')
        super().__init__(reader if archive is None else archive, check)
        self.serial = recipe
        self._target = recipe.tasks[0] if isinstance(recipe, IndependentRecipe) else None

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

        panels = (self._read_pools(visit) if self._target is None
                  else self._read_pools(visit, only=self._target))
        for task, train, valid, evidence in panels:
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

    def load_tests(self, visit: Callable[[str], None], *, only: Task | None = None
                   ) -> dict[str, tuple[PairBatch, dict[str, Any]]]:
        if only is not None and only != self._target:
            raise ValueError('TEST_ORDER')
        if self._target is None:
            return super().load_tests(visit)
        return super().load_tests(visit, only=self._target)

class VolumePools:
    """TRAIN/DEV-only view of the original pinned SKINNY MSE archive."""

    def __init__(self, recipe: VolumeRecipe, check: Callable[[], None]) -> None:
        self._recipe, self._check = recipe, check
        self._fixed: FixedPairReplay | None = None
        self._started = False

    def bind(self, visit: Callable[[str], None]) -> tuple[dict[str, PairBatch], dict[str, Any]]:
        if self._started:
            raise ValueError('POOL_ORDER')
        self._started = True
        recipe = self._recipe
        task, = recipe.tasks
        visit(task.cipher)
        self._check()
        train, valid, source = load_reused_pool(
            skinny_pool_recipe(recipe.archive_reader_recipe()), self._check)
        original_digest = train.digest
        if original_digest != recipe.source_train_digest:
            raise PoolSourceError
        self._check()

        def trail(panel: PairBatch, digest: str) -> dict[str, Any]:
            return dict(source_digest=digest, source_groups=panel.groups,
                        source_namespace=panel.namespace, source=source)

        def subset(panel: PairBatch, groups: int, pin: SubsetPin,
                   digest: str) -> tuple[PairBatch, dict[str, Any]]:
            self._check()
            seed = derive_seed(recipe.streams, task, panel.split, 0, 'sample')
            batch, rows = index_subset(panel, groups, seed, recipe.namespace)
            computed = SubsetPin(batch.digest, sha256(rows).hexdigest())
            if computed != pin:
                raise PoolSourceError
            self._check()
            return batch, dict(groups=batch.groups, **asdict(computed), seed=seed,
                               index_encoding='original-zero-based-le-i64-in-subset-order',
                               **trail(panel, digest))

        tr = recipe.training
        groups = tr.batches_per_epoch * tr.batch_groups
        if recipe.train_scope == 'prior-small' and groups < train.groups:
            train, train_trail = subset(train, groups, recipe.small_train, original_digest)
        elif recipe.train_scope == 'full-original' and train.groups == groups:
            # The original order is the shuffle base; identity is the whole source digest.
            train_trail = dict(groups=train.groups, digest=original_digest, index_sha256=None,
                               index_encoding='full-source-in-original-order',
                               **trail(train, original_digest))
        else:
            raise PoolSourceError
        if tr.dev_groups > valid.groups:
            raise PoolSourceError
        dev, dev_trail = subset(valid, tr.dev_groups, recipe.common_dev, valid.digest)
        del valid
        self._fixed = FixedPairReplay(train, tr, recipe.streams, task, recipe.namespace,
                                     self._check)
        return {task.cipher: dev}, {task.cipher: dict(train=train_trail, dev=dev_trail,
                                                   prior_small=recipe.prior_small())}

    def train(self, task: Task, cycle: int) -> PairBatch:
        if self._fixed is None:
            raise ValueError('POOL_NOT_READY')
        if task != self._recipe.tasks[0]:
            raise ValueError('POOL_TASK')
        batch = self._fixed.train(cycle)
        # Large keeps the original panel object; only its small output batches
        # need to be rebound to the same serial ledger namespace as the small arm.
        if batch.namespace != self._recipe.namespace:
            return replace(batch, namespace=self._recipe.namespace)
        return batch

    def release_train(self) -> None:
        self._fixed = None


def encode_test(batch: PairBatch, task: Task, recipe: SerialRecipe, source: dict[str, str],
                *, logits: NDArray[np.float64], final: dict[str, Any],
                reused_input: dict[str, Any]) -> bytes:
    if ((batch.split, batch.index, batch.groups, batch.namespace) != (
            'test', 0, recipe.training.test_groups, recipe.archive_reader_recipe().namespace)
            or task not in recipe.tasks or final.get('stage') != len(recipe.stage_slots)
            or (isinstance(recipe, IndependentRecipe)
                and (final.get('task') != task.cipher
                     or final.get('routing_mode') != recipe.routing_mode))):
        raise ValueError('TEST_PANEL_IDENTITY')
    return _panel_bytes(batch, task, recipe, source, BACKEND, logits=logits, best=final,
                        reused_input=reused_input)

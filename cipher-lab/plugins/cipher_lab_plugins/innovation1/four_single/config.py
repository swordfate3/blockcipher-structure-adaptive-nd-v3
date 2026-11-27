"""Frozen four-task contract; importable on stdlib-only check."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import PurePath, PureWindowsPath
from typing import Any

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks
from cipher_lab.training.early_stopping import Settings

from ..config import TASKS, Recipe, Resources, Task, Training, canonical, validate_run_id
from ..contracts import MAX_CHECKPOINT
from ..layout_probe.config import FIXED_BCE, FIXED_MSE, FilePin, PoolSource
from ..layout_probe.config import SEEDS as SKINNY_SEEDS
from ..single_train.config import ARCHITECTURE, BACKEND, DEVICE, SingleRecipe
from ..single_train.config import estimate as single_estimate

METHOD = 'innovation1-four-task-single-pair-v1'
NAMESPACE = 'innovation1-four-task-single-pair-v1-20261122'
REPLAY_METHOD = 'innovation1-four-task-single-pair-50epoch-v1'
REPLAY_ROOT = r'G:\lxy\shared-m-cipher-nd-runs\20261123-four-task-single-pair-50epoch-01'
REPLAY_ID = 'innovation1-four-task-single-pair-50epoch-20261123-01'
SLOT = 'M-best'
PARAMETERS = 1_087_509
BATCH_LEDGER_ROWS = 256
ROW_BYTES_LIMIT = 1024
PANEL_OVERHEAD = 1 << 20
RECORD_REPORT_RESERVE = 32 << 20
PROGRESS_CHUNKS = 32


@dataclass(frozen=True)
class FourSingleRecipe(SingleRecipe):
    method: str = METHOD
    namespace: str = NAMESPACE
    run_id: str | None = None
    tasks: tuple[Task, ...] = TASKS
    training: Training = replace(SingleRecipe.training, epochs=10, batches_per_epoch=1800,
                       batch_groups=5000, dev_groups=1_000_000)
    resources: Resources = replace(SingleRecipe.resources, wall_seconds=43200)
    skinny_source: PoolSource | None = FIXED_BCE.pool_source

    @property
    def fixed_pool(self) -> bool:
        return True

    @property
    def streams(self) -> Recipe:
        return replace(ARCHITECTURE, namespace=self.namespace, tasks=self.tasks)

    def to_dict(self) -> dict[str, Any]:
        tr = self.training
        if self.skinny_source is None:
            raise ValueError('POOL_SOURCE_REQUIRED')
        model = asdict(ARCHITECTURE.model)
        del model['dedicated']
        model['suffix_kernels'] = list(model['suffix_kernels'])
        model.update(parameters=PARAMETERS, readout='single-pair',
                     skinny_nibble='column-if-selected')
        optimizer = asdict(ARCHITECTURE.optimizer)
        optimizer['betas'] = list(optimizer['betas'])
        seeds = asdict(ARCHITECTURE.seeds)
        del seeds['bootstrap']
        return dict(method=self.method, namespace=self.namespace, run_id=self.run_id,
                    tasks=[asdict(t) for t in self.tasks], training=asdict(tr),
                    resources=asdict(self.resources), backend=BACKEND, device=DEVICE,
                    model=model, optimizer=optimizer, seeds=seeds,
                    selection=dict(metric='equal-macro-dev-bce', ties='earliest',
                                   interval='epoch-end', early_stopping=False,
                                   test='after-one-persisted-strict-best-reload'),
                    data={**{t.cipher: dict(protocol='fresh-fixed-stratified-90-10-v1',
                              namespace=self.namespace) for t in self.tasks[:-1]},
                          'SKINNY': dict(protocol='reuse-fixed-stratified-90-10-v1',
                              source=asdict(self.skinny_source), namespace=FIXED_MSE.namespace,
                              shuffle_namespace=FIXED_BCE.shuffle_namespace,
                              shuffle_seed=SKINNY_SEEDS.shuffle)})


FOUR_SINGLE = FourSingleRecipe()


@dataclass(frozen=True)
class ReplaySource:
    root: str = r'G:\lxy\shared-m-cipher-nd-runs\20261122-four-task-single-pair-01'
    run_id: str = 'innovation1-four-task-single-pair-20261122-01'
    record: FilePin = FilePin(77847,
                             '7a4a441ede8144b462cf15e5975240efd84eba1757a78f040c7eec3a3a665ddf')


@dataclass(frozen=True)
class ReplayRecipe(FourSingleRecipe):
    method: str = REPLAY_METHOD
    run_id: str = REPLAY_ID
    training: Training = replace(FOUR_SINGLE.training, epochs=50)
    resources: Resources = replace(FOUR_SINGLE.resources, wall_seconds=28 * 3600)
    archive: ReplaySource = ReplaySource()
    early_stopping: Settings = Settings(warmup_epochs=24, patience=12, min_delta=1e-5)
    cosine_epochs: int = FOUR_SINGLE.training.epochs

    @property
    def reuse_pool(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        recipe = super().to_dict()
        recipe['optimizer'].update(schedule='cycle-cosine-then-constant',
                                   cosine_epochs=self.cosine_epochs)
        recipe['selection']['early_stopping'] = asdict(self.early_stopping)
        recipe['output_root'] = REPLAY_ROOT
        recipe['data'].update({t.cipher: dict(protocol='reuse-fixed-stratified-90-10-v1',
            namespace=self.namespace) for t in self.tasks[:-1]})
        recipe['data'].update(archive=asdict(self.archive), test='reuse-prior-fixed-test')
        return recipe


FOUR_SINGLE_50 = ReplayRecipe()


def validate_replay_root(root: PurePath) -> None:
    # Require the native absolute sibling, not a relative spelling on POSIX or
    # a different Windows path (including device/UNC and drive-relative aliases).
    # CLI applies the existing no-reparse guard before acquiring any write owner.
    if not root.is_absolute() or PureWindowsPath(root) != PureWindowsPath(REPLAY_ROOT):
        raise ValueError('REPLAY_OUTPUT_ROOT')


def official(recipe: FourSingleRecipe, run_id: str) -> FourSingleRecipe:
    from .interleave_config import InterleaveRecipe
    from .interleave_config import official as official_interleave

    validate_run_id(run_id)
    if isinstance(recipe, InterleaveRecipe):
        return official_interleave(recipe, run_id)
    if isinstance(recipe, ReplayRecipe):
        if (run_id != REPLAY_ID or type(recipe) is not ReplayRecipe or recipe != FOUR_SINGLE_50
                or canonical(recipe.to_dict()) != canonical(FOUR_SINGLE_50.to_dict())):
            raise ValueError('CONFIG_EXACT')
        return FOUR_SINGLE_50
    pin = FOUR_SINGLE.skinny_source
    expected = replace(FOUR_SINGLE, run_id=run_id)
    if (pin is None or run_id.casefold() == pin.run_id.casefold()
            or type(recipe) is not FourSingleRecipe or recipe != expected
            or canonical(recipe.to_dict()) != canonical(expected.to_dict())):
        raise ValueError('CONFIG_EXACT')
    return expected


def estimate(recipe: FourSingleRecipe) -> dict[str, int]:
    n = single_estimate(recipe)  # shared geometry validation (including stratified 90/10)
    if recipe.tasks != TASKS or recipe.skinny_source is None:
        raise ValueError('POOL_RECIPE')
    cycles = n['cycles']
    tr = recipe.training
    replay = isinstance(recipe, ReplayRecipe)
    tasks, fresh = len(TASKS), 0 if replay else len(TASKS) - 1
    pool = n['pool_groups']
    fresh_blocks, test_blocks = 2 * fresh * pool, 2 * tasks * tr.test_groups
    if replay:
        test_blocks = 0
    updates = tasks * cycles
    # Includes all immutable data, bounded JSON ledgers, both copies of the
    # one atomic-replacement model and shared record/report/temp overhead.
    panels = fresh * pool * 17 + tasks * tr.test_groups * (17 + 8)
    # Group flushing can happen at each epoch boundary in addition to full chunks.
    ledger = updates * ROW_BYTES_LIMIT + (updates // BATCH_LEDGER_ROWS + tr.epochs) * 2
    retained_panel_bound = panels + (2 * fresh + tasks) * PANEL_OVERHEAD
    output = retained_panel_bound + ledger + 2 * MAX_CHECKPOINT + RECORD_REPORT_RESERVE
    forward_rows = tasks * (n['train_groups'] + tr.epochs * tr.dev_groups + tr.test_groups)
    return dict(fits=1, retries=0, cycles=cycles, optimizer_updates=updates,
                updates_each=cycles, train_rows=tasks * n['train_groups'],
                fixed_train_rows_each=n['train_groups_per_epoch'],
                fixed_dev_rows_each=tr.dev_groups,
                test_rows_each=tr.test_groups, dev_tail_rows=tr.dev_groups % tr.eval_batch_groups,
                fresh_pool_real_blocks=fresh_blocks, test_real_blocks=test_blocks,
                real_blocks=fresh_blocks + test_blocks, null_words=0,
                forward_rows=forward_rows,
                panel_bound_bytes=retained_panel_bound, batch_ledger_bound_bytes=ledger,
                artifact_peak_bound_bytes=output)


def make_policy(recipe: FourSingleRecipe, clocks: Clocks) -> Policy:
    n, limits = estimate(recipe), recipe.resources
    deadline = clocks.utc() + timedelta(seconds=limits.wall_seconds)
    return Policy(2, limits.wall_seconds, limits.cpu_seconds, deadline.isoformat(),
                  n['real_blocks'], 0, limits.rss_bytes, limits.artifact_bytes,
                  n['optimizer_updates'], n['forward_rows'])

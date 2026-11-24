"""Exact stdlib settings for serial and independent B/C, not execution permission."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import PurePath, PureWindowsPath
from typing import Any, Literal

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Stopped

from ..config import TASKS, Resources, Task, Training, canonical
from ..contracts import MAX_CHECKPOINT
from ..four_single.config import (
    BATCH_LEDGER_ROWS,
    FOUR_SINGLE,
    PANEL_OVERHEAD,
    RECORD_REPORT_RESERVE,
    ROW_BYTES_LIMIT,
    SLOT,
    FourSingleRecipe,
    ReplayRecipe,
    ReplaySource,
)
from ..single_train.config import estimate as estimate_pair

METHOD = 'innovation1-four-task-serial-bc-v1'
NAMESPACE = 'innovation1-four-task-serial-bc-v1-20261124'
ROOT = r'G:\lxy\shared-m-cipher-nd-runs\20261124-four-task-serial-bc-01'
INDEPENDENT_METHOD = 'innovation1-independent-bc-30epoch-v1'
INDEPENDENT_ROOT = r'G:\lxy\shared-m-cipher-nd-runs\20261124-independent-bc-30epoch-01'
VOLUME_METHOD = 'innovation1-skinny-train-volume-v1'
VOLUME_ROOT = r'G:\lxy\shared-m-cipher-nd-runs\20261124-skinny-train-volume-01'
VOLUME_GPU_UUID = 'GPU-171dfbbe-c1a7-d93b-549d-8824cb8a18c7'
DEV_OBSERVATION_BYTES = 4096
DEADLINE_ENV = 'CIPHER_LAB_SERIAL_DEADLINE_UTC'
STAGE_SLOTS = ('M-stage1', 'M-stage2', 'M-stage3', SLOT)
RoutingMode = Literal['public-card', 'constant-no-card']


@dataclass(frozen=True)
class SerialRecipe(FourSingleRecipe):
    method: str = METHOD
    namespace: str = NAMESPACE
    run_id: str = 'innovation1-four-task-serial-b-20261124-01'
    training: Training = replace(FOUR_SINGLE.training, batches_per_epoch=180, dev_groups=100_000)
    resources: Resources = replace(FOUR_SINGLE.resources, wall_seconds=5400)
    archive: ReplaySource = ReplaySource()
    routing_mode: RoutingMode = 'public-card'

    @property
    def reuse_pool(self) -> bool:
        return True

    @property
    def stage_slots(self) -> tuple[str, ...]:
        return STAGE_SLOTS

    def archive_reader_recipe(self) -> ReplayRecipe:
        # Reader geometry is that of the ORIGINAL run, not our subsets.
        return replace(ReplayRecipe(), archive=self.archive, skinny_source=self.skinny_source)

    def to_dict(self) -> dict[str, Any]:
        recipe = super().to_dict()
        recipe.update(output_root=ROOT, routing=dict(mode=self.routing_mode,
                      constant=[.5] * 13 if self.routing_mode == 'constant-no-card' else None),
                      selection=dict(rule='fourth-stage-only', interval='stage-end',
                                     early_stopping=False, test='after-strict-final-reload'),
                      stages=dict(order=[t.cipher for t in self.tasks], adam='continuous',
                                  slots=list(self.stage_slots)))
        recipe['optimizer'].update(schedule='task-local-cycle-cosine')
        recipe['data'] = dict(archive=asdict(self.archive), skinny_source=asdict(self.skinny_source)
                              if self.skinny_source is not None else None,
                              protocol='split-local-stratified-index-pcg64-v1',
                              order='balanced-fixed-pair-replay-v1',
                              test='same-four-prior-fixed-panels-rescored')
        return recipe


SERIAL_B = SerialRecipe()
SERIAL_C = replace(SERIAL_B, run_id='innovation1-four-task-serial-c-20261124-01',
                   routing_mode='constant-no-card')


@dataclass(frozen=True)
class IndependentRecipe(SerialRecipe):
    method: str = INDEPENDENT_METHOD
    run_id: str = 'innovation1-independent-simeck-b-30epoch-20261124-01'
    tasks: tuple[Task, ...] = (TASKS[0],)
    training: Training = replace(SerialRecipe.training, epochs=30)

    @property
    def stage_slots(self) -> tuple[str, ...]:
        return (SLOT,)

    def to_dict(self) -> dict[str, Any]:
        recipe = super().to_dict()
        recipe['output_root'] = INDEPENDENT_ROOT
        recipe['selection'].update(rule='last-epoch-only', interval='fit-end')
        recipe['stages']['adam'] = 'fresh'
        recipe['data']['test'] = 'same-task-prior-fixed-panel-rescored'
        return recipe


_ARMS: tuple[tuple[str, RoutingMode], ...] = (('b', 'public-card'), ('c', 'constant-no-card'))
INDEPENDENT_RECIPES: tuple[IndependentRecipe, ...] = tuple(
    IndependentRecipe(tasks=(task,), routing_mode=mode,
        run_id=f'innovation1-independent-{task.cipher.lower()}-{arm}-30epoch-20261124-01')
    for task in TASKS for arm, mode in _ARMS
)


@dataclass(frozen=True)
class SubsetPin:
    digest: str
    index_sha256: str


_PRIOR_SKINNY_B = INDEPENDENT_RECIPES[-2]


@dataclass(frozen=True)
class VolumeRecipe(IndependentRecipe):
    method: str = VOLUME_METHOD
    run_id: str = 'innovation1-skinny-train-volume-small-20261124-01'
    tasks: tuple[Task, ...] = _PRIOR_SKINNY_B.tasks
    training: Training = replace(_PRIOR_SKINNY_B.training, test_groups=0)
    resources: Resources = replace(_PRIOR_SKINNY_B.resources, wall_seconds=3600)
    train_scope: Literal['prior-small', 'full-original'] = 'prior-small'
    dev_updates: tuple[int, ...] = (1800, 3600, 5400)
    prior_record_sha256: str = '3843f9a08b9b40c3b0e1a5ad2f66f316b549229d9845d1b93044a1eb852193e0'
    source_train_digest: str = 'f2fa2651f899a04ee079458fa026cedaff75c1f1f979afc12362d9065603c39d'
    small_train: SubsetPin = SubsetPin(
        '471c7276d81be465dc45db88f6922884a08133cd29d51053501764b15645ff8e',
        '5c8137d9db18f90d38910e81041062b7b5d8e6c43196a947b2d59a9a5752a52f')
    common_dev: SubsetPin = SubsetPin(
        '22d91a93a58e381788c49856585062e3d5a59b7f763bda53bd627b1935fc7bf2',
        '6c0e924d75810df8dc91c5bf7c0737f107bd62f11dfb56138710fdb5cc3d1c19')

    @property
    def stage_slots(self) -> tuple[str, ...]:
        return ()

    def prior_small(self) -> dict[str, Any]:
        return dict(run_id=_PRIOR_SKINNY_B.run_id, record_sha256=self.prior_record_sha256,
                    train=asdict(self.small_train), dev=asdict(self.common_dev))

    def to_dict(self) -> dict[str, Any]:
        recipe = super().to_dict()
        recipe['output_root'] = VOLUME_ROOT
        recipe['gpu_uuid'] = VOLUME_GPU_UUID
        recipe['selection'] = dict(rule='none', interval='fixed-completed-updates',
                                   dev_updates=list(self.dev_updates), primary='final-dev-bce',
                                   early_stopping=False, test='never')
        recipe['data'].pop('archive')  # Prior run identity is evidence, never a reader input.
        recipe['data'].update(test='never', train_scope=self.train_scope,
                              source_train_digest=self.source_train_digest,
                              prior_small=self.prior_small())
        return recipe


_VOLUME_SMALL = VolumeRecipe()
VOLUME_RECIPES = (
    _VOLUME_SMALL,
    replace(_VOLUME_SMALL, run_id='innovation1-skinny-train-volume-large-20261124-01',
            train_scope='full-original',
            training=replace(_VOLUME_SMALL.training, epochs=3, batches_per_epoch=1800)),
)

def validate_root(root: PurePath, *, expected: str = ROOT) -> None:
    if not root.is_absolute() or PureWindowsPath(root) != PureWindowsPath(expected):
        raise ValueError('SERIAL_OUTPUT_ROOT')


def official(recipe: SerialRecipe, run_id: str) -> None:
    if (recipe.run_id != run_id or not any(
            type(recipe) is type(exact) and recipe == exact
            and canonical(recipe.to_dict()) == canonical(exact.to_dict())
            for exact in (SERIAL_B, SERIAL_C, *INDEPENDENT_RECIPES, *VOLUME_RECIPES))):
        raise ValueError('CONFIG_EXACT')


def estimate(recipe: SerialRecipe) -> dict[str, int]:
    if isinstance(recipe, VolumeRecipe):
        return estimate_volume(recipe)
    pair = estimate_pair(recipe)
    tr, tasks = recipe.training, len(recipe.tasks)
    if not tasks or tasks != len(recipe.stage_slots):
        raise ValueError('SERIAL_GEOMETRY')
    updates = pair['cycles'] * tasks
    panels = tasks * (tr.test_groups * 25 + PANEL_OVERHEAD)
    ledger = updates * ROW_BYTES_LIMIT + 2 * (updates // BATCH_LEDGER_ROWS + tr.epochs * tasks)
    return dict(fits=1, retries=0, stages=tasks, cycles=pair['cycles'],
                updates_each=pair['cycles'], optimizer_updates=updates,
                train_rows=updates * tr.batch_groups, dev_rows=tasks**2 * tr.dev_groups,
                test_rows=tasks * tr.test_groups,
                fixed_train_rows_each=pair['train_groups_per_epoch'],
                fixed_dev_rows_each=tr.dev_groups, test_rows_each=tr.test_groups,
                dev_tail_rows=tr.dev_groups % tr.eval_batch_groups,
                real_blocks=0, null_words=0, forward_rows=updates * tr.batch_groups
                + tasks**2 * tr.dev_groups + tasks * tr.test_groups,
                panel_bound_bytes=panels, batch_ledger_bound_bytes=ledger,
                artifact_peak_bound_bytes=panels + ledger + tasks * MAX_CHECKPOINT
                + RECORD_REPORT_RESERVE)


def estimate_volume(recipe: VolumeRecipe) -> dict[str, int]:
    tr, points = recipe.training, recipe.dev_updates
    if (recipe.tasks != (TASKS[-1],) or recipe.routing_mode != 'public-card'
            or any(type(n) is not int or n <= 0 for n in (tr.epochs, tr.batches_per_epoch,
                   tr.batch_groups, tr.dev_groups, tr.eval_batch_groups))
            or any(n % 4 for n in (tr.batch_groups, tr.dev_groups, tr.eval_batch_groups))
            or (tr.test_groups, tr.pairs_per_group, tr.gradient_accumulation, tr.retries)
            != (0, 1, 1, 0)):
        raise ValueError('VOLUME_GEOMETRY')
    updates = tr.epochs * tr.batches_per_epoch
    if (len(points) != 3 or any(type(n) is not int or n <= 0
                               or n % tr.batches_per_epoch for n in points)
            or not points[0] < points[1] < points[2] == updates):
        raise ValueError('VOLUME_CADENCE')
    train, dev = updates * tr.batch_groups, len(points) * tr.dev_groups
    ledger = updates * ROW_BYTES_LIMIT + 2 * (updates // BATCH_LEDGER_ROWS + tr.epochs)
    return dict(fits=1, retries=0, stages=1, cycles=updates, updates_each=updates,
                optimizer_updates=updates, train_rows=train, dev_rows=dev, test_rows=0,
                fixed_train_rows_each=tr.batches_per_epoch * tr.batch_groups,
                fixed_dev_rows_each=tr.dev_groups, test_rows_each=0,
                dev_tail_rows=tr.dev_groups % tr.eval_batch_groups, real_blocks=0, null_words=0,
                forward_rows=train + dev, panel_bound_bytes=0, batch_ledger_bound_bytes=ledger,
                artifact_peak_bound_bytes=ledger + len(points) * DEV_OBSERVATION_BYTES
                + RECORD_REPORT_RESERVE)

def make_policy(recipe: SerialRecipe, clocks: Clocks, inherited: str | None = None) -> Policy:
    if isinstance(recipe, IndependentRecipe) and inherited is None:
        raise Stopped('DEADLINE')
    limits, n = recipe.resources, estimate(recipe)
    now = clocks.utc()
    own = now + timedelta(seconds=limits.wall_seconds)
    try:
        # Policy validates the precise absolute UTC spelling even if it is later
        # than our own deadline, so an invalid value never silently gets ignored.
        request = Policy(2, limits.wall_seconds, limits.cpu_seconds,
                         own.isoformat() if inherited is None else inherited,
                         0, 0, limits.rss_bytes, limits.artifact_bytes,
                         n['optimizer_updates'], n['forward_rows'])
        deadline = datetime.fromisoformat(request.deadline_utc)
    except ValueError:
        raise Stopped('DEADLINE') from None
    if deadline <= now:
        raise Stopped('DEADLINE')
    return request if deadline <= own else replace(request, deadline_utc=own.isoformat())

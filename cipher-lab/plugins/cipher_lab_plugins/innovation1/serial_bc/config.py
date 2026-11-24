"""Exact stdlib settings for the two serial runs, not an execution permission."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import PurePath, PureWindowsPath
from typing import Any, Literal

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Stopped

from ..config import Resources, Training, canonical
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
                                  slots=list(STAGE_SLOTS)))
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


def validate_root(root: PurePath) -> None:
    if not root.is_absolute() or PureWindowsPath(root) != PureWindowsPath(ROOT):
        raise ValueError('SERIAL_OUTPUT_ROOT')


def official(recipe: SerialRecipe, run_id: str) -> None:
    if (type(recipe) is not SerialRecipe or recipe.run_id != run_id
            or not any(recipe == exact and canonical(recipe.to_dict()) == canonical(exact.to_dict())
                       for exact in (SERIAL_B, SERIAL_C))):
        raise ValueError('CONFIG_EXACT')


def estimate(recipe: SerialRecipe) -> dict[str, int]:
    pair = estimate_pair(recipe)
    tr, tasks = recipe.training, len(STAGE_SLOTS)
    if tasks != len(recipe.tasks):
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


def make_policy(recipe: SerialRecipe, clocks: Clocks, inherited: str | None = None) -> Policy:
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

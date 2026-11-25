"""One exact stdlib-only recipe and its paid-work and output reservations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import PurePath
from typing import Any

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Stopped
from cipher_lab.training.early_stopping import Settings

from ..config import Resources, Task, Training, canonical
from ..contracts import MAX_CHECKPOINT
from ..four_single.config import (
    BATCH_LEDGER_ROWS,
    FOUR_SINGLE,
    PANEL_OVERHEAD,
    RECORD_REPORT_RESERVE,
    FourSingleRecipe,
    ReplayRecipe,
    ReplaySource,
)
from ..serial_bc.config import DEADLINE_ENV as DEADLINE_ENV
from ..serial_bc.config import validate_root as validate_serial_root
from ..single_train.config import TASK
from ..single_train.config import estimate as estimate_pair

METHOD = 'innovation1-skinny-full-pool-card-bc-v1'
NAMESPACE = 'innovation1-skinny-full-pool-card-bc-v1-20261125'
RUN_ID = 'innovation1-skinny-full-pool-card-bc-20261125-01'
ROOT = r'G:\lxy\shared-m-cipher-nd-runs\20261125-skinny-full-pool-card-bc-01'
GPU_UUID = 'GPU-171dfbbe-c1a7-d93b-549d-8824cb8a18c7'
SLOTS = ('B-card-best', 'C-no-card-best')
ROW_BYTES = 448


@dataclass(frozen=True)
class CardRecipe(FourSingleRecipe):
    method: str = METHOD
    namespace: str = NAMESPACE
    run_id: str = RUN_ID
    tasks: tuple[Task, ...] = (TASK,)
    training: Training = replace(FOUR_SINGLE.training, epochs=50)
    resources: Resources = replace(FOUR_SINGLE.resources, wall_seconds=50400)
    archive: ReplaySource = ReplaySource()
    test_sha256: str = '3a845204b3ccf3afb77b73625219fcd78b107155a9526406fe3693935f1f3c27'
    test_digest: str = 'abb5a3a9d6c0fa4303ac353e2ae9117ba8e7263507e60b8fdf322fd061119bb6'
    early_stopping: Settings = Settings(24, 12, 1e-5)
    cosine_epochs: int = 24

    @property
    def reuse_pool(self) -> bool:
        return True

    def archive_reader_recipe(self) -> ReplayRecipe:
        return replace(ReplayRecipe(), archive=self.archive, skinny_source=self.skinny_source,
                       training=self.training)

    def to_dict(self) -> dict[str, Any]:
        value = super().to_dict()
        value.update(output_root=ROOT, gpu_uuid=GPU_UUID, deadline_env=DEADLINE_ENV,
                     routing={SLOTS[0]: dict(mode='public-card', constant=None),
                              SLOTS[1]: dict(mode='constant-no-card', constant=[.5] * 13)},
                     selection=dict(metric='per-model-dev-bce', ties='earliest',
                                    interval='epoch-end',
                                    early_stopping=asdict(self.early_stopping),
                                    stop='all-models-stale-or-epoch-ceiling',
                                    test='after-two-persisted-strict-best-reloads'))
        value['data']['test'] = dict(protocol='reuse-original-four-single-skinny-v1',
                                     archive=asdict(self.archive), ref='dataset/test-SKINNY.npz',
                                     sha256=self.test_sha256, digest=self.test_digest)
        value['optimizer'].update(schedule='cycle-cosine-then-constant',
                                  cosine_epochs=self.cosine_epochs)
        return value


CARD = CardRecipe()


def official(recipe: CardRecipe, run_id: str) -> None:
    if (run_id != RUN_ID or type(recipe) is not CardRecipe or recipe != CARD
            or canonical(recipe.to_dict()) != canonical(CARD.to_dict())):
        raise ValueError('CONFIG_EXACT')


def validate_root(root: PurePath) -> None:
    validate_serial_root(root, expected=ROOT)


def estimate(recipe: CardRecipe) -> dict[str, int]:
    n = estimate_pair(recipe)
    tr = recipe.training
    if recipe.tasks != (TASK,) or recipe.skinny_source is None:
        raise ValueError('POOL_RECIPE')
    # Archive TEST needs no generation. A paired ledger stores one row per cycle.
    ledger = n['cycles'] * ROW_BYTES + 2 * (n['cycles'] // BATCH_LEDGER_ROWS + tr.epochs)
    panel = tr.test_groups * (17 + 2 * 8) + PANEL_OVERHEAD
    # Two registered weights plus one full replacement temporary; extra full
    # TEST temporary and record/report/progress temporaries are conservatively charged.
    peak = ledger + 2 * panel + 3 * MAX_CHECKPOINT + RECORD_REPORT_RESERVE
    if peak > recipe.resources.artifact_bytes:
        raise ValueError('ARTIFACT_BOUND')
    return dict(n, real_blocks=0, panel_bound_bytes=panel, batch_ledger_bound_bytes=ledger,
                artifact_peak_bound_bytes=peak)


def make_policy(recipe: CardRecipe, clocks: Clocks, inherited: str | None) -> Policy:
    n, limits = estimate(recipe), recipe.resources
    now = clocks.utc()
    if not inherited:
        raise Stopped('DEADLINE')
    try:
        policy = Policy(2, limits.wall_seconds, limits.cpu_seconds, inherited, 0, 0,
                        limits.rss_bytes, limits.artifact_bytes,
                        n['optimizer_updates'], n['forward_rows'])
        deadline = datetime.fromisoformat(policy.deadline_utc)
    except ValueError:
        raise Stopped('DEADLINE') from None
    # Never default, rebase or silently change the controller's shared UTC cutoff.
    if not now < deadline <= now + timedelta(seconds=limits.wall_seconds):
        raise Stopped('DEADLINE')
    return policy

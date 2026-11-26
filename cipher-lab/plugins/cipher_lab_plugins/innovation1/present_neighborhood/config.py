"""One exact stdlib-only recipe, with no production overrides or fallback."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import PurePath
from typing import Any

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Stopped

from ..config import TASKS, Resources, Task, canonical
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
from ..single_train.config import estimate as estimate_pair
from .layout import DEFAULT_SHA256, STRUCTURE_SHA256

METHOD = 'innovation1-present-neighborhood-v1'
NAMESPACE = 'innovation1-present-neighborhood-v1-20261126'
RUN_ID = 'innovation1-present-neighborhood-20261126-01'
ROOT = r'G:\lxy\shared-m-cipher-nd-runs\20261126-present-neighborhood-01'
GPU_UUID = 'GPU-171dfbbe-c1a7-d93b-549d-8824cb8a18c7'
TASK = TASKS[1]
SLOTS = ('A-default-best', 'B-structure-best')
TEST_REF = f'dataset/test-{TASK.cipher}.npz'
ROW_BYTES = 448
ROUTE = (('nibble', .5), ('word', .5))
NIBBLE_BUFFER = 'shared.stems.nibble.indices'
TABLE_SHA256 = (DEFAULT_SHA256, STRUCTURE_SHA256)


@dataclass(frozen=True)
class NeighborhoodRecipe(FourSingleRecipe):
    method: str = METHOD
    namespace: str = NAMESPACE
    run_id: str = RUN_ID
    tasks: tuple[Task, ...] = (TASK,)
    resources: Resources = replace(FOUR_SINGLE.resources, wall_seconds=10800)
    archive: ReplaySource = ReplaySource()
    test_sha256: str = '6e8ea523326fddebf61d997e91c75c72517fbc1edcbd63621553f85e93b05143'
    test_digest: str = 'c270e4a121e6945ba283c12f93f30cf0e663b968a62a609d7c9a270bae261ffd'

    @property
    def reuse_pool(self) -> bool:
        return True

    def archive_reader_recipe(self) -> ReplayRecipe:
        # The reader describes the original four-task format and shuffle domain;
        # the provider below exposes only the PRESENT view of that archive.
        return replace(ReplayRecipe(), archive=self.archive, skinny_source=self.skinny_source,
                       training=self.training)

    def to_dict(self) -> dict[str, Any]:
        value = super().to_dict()
        del value['model']['skinny_nibble']
        value.update(output_root=ROOT, gpu_uuid=GPU_UUID, deadline_env=DEADLINE_ENV,
                     slots=list(SLOTS), routing=dict(mode='fixed-no-trained-router',
                         weights=dict(ROUTE), public_card='PRESENT-preprocessing-only'),
                     geometry=dict(buffer=NIBBLE_BUFFER, shape=[64, 9], center_slot=4,
                         sentinel=64, encoding='msb0-int64-le-row-major',
                         sha256=dict(zip(SLOTS, TABLE_SHA256, strict=True))),
                     selection=dict(metric='per-model-dev-bce', ties='earliest',
                         interval='epoch-end', early_stopping=False, stop='epoch-ceiling',
                         test='after-two-persisted-strict-best-reloads'))
        value['optimizer'].update(schedule='cycle-cosine', cosine_epochs=self.training.epochs)
        value['seeds']['init_domain'] = 'same-seed-as-independent-present-b'
        value['data'] = dict(protocol='reuse-original-four-single-present-v1',
            archive=asdict(self.archive), archive_recipe_skinny_source=value['data']['SKINNY'][
                'source'], namespace=FOUR_SINGLE.namespace, order='balanced-fixed-pair-replay-v1',
            test=dict(ref=TEST_REF, sha256=self.test_sha256, digest=self.test_digest))
        return value


NEIGHBORHOOD = NeighborhoodRecipe()


def official(recipe: NeighborhoodRecipe, run_id: str) -> None:
    if (run_id != RUN_ID or type(recipe) is not NeighborhoodRecipe or recipe != NEIGHBORHOOD
            or canonical(recipe.to_dict()) != canonical(NEIGHBORHOOD.to_dict())):
        raise ValueError('CONFIG_EXACT')


def validate_root(root: PurePath) -> None:
    validate_serial_root(root, expected=ROOT)


def estimate(recipe: NeighborhoodRecipe) -> dict[str, int]:
    n = estimate_pair(recipe)
    tr = recipe.training
    if recipe.tasks != (TASK,) or recipe.skinny_source is None:
        raise ValueError('POOL_RECIPE')
    ledger = n['cycles'] * ROW_BYTES + 2 * (n['cycles'] // BATCH_LEDGER_ROWS + tr.epochs)
    panel = tr.test_groups * 33 + PANEL_OVERHEAD
    peak = ledger + 2 * panel + 3 * MAX_CHECKPOINT + RECORD_REPORT_RESERVE
    if peak > recipe.resources.artifact_bytes:
        raise ValueError('ARTIFACT_BOUND')
    return dict(n, real_blocks=0, panel_bound_bytes=panel, batch_ledger_bound_bytes=ledger,
                artifact_peak_bound_bytes=peak)


def make_policy(recipe: NeighborhoodRecipe, clocks: Clocks, inherited: str | None) -> Policy:
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
    if not now < deadline <= now + timedelta(seconds=limits.wall_seconds):
        raise Stopped('DEADLINE')
    return policy

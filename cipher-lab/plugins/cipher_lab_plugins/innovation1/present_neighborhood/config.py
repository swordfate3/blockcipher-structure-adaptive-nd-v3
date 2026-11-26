"""One exact stdlib-only recipe, with no production overrides or fallback."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import PurePath
from typing import Any

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Stopped
from cipher_lab.present_public import PRESENT_P_LAYER, PLayer

from ..config import TASKS, Resources, Task, Training, canonical
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
from .layout import DEFAULT_SHA256, NIBBLE_RULE, STRUCTURE_SHA256, GatherRule, verified_card

METHOD = 'innovation1-present-neighborhood-v1'
NAMESPACE = 'innovation1-present-neighborhood-v1-20261126'
RUN_ID = 'innovation1-present-neighborhood-20261126-01'
ROOT = r'G:\lxy\shared-m-cipher-nd-runs\20261126-present-neighborhood-01'
CARD_METHOD = 'innovation1-present-card-30epoch-v1'
CARD_RUN_ID = 'innovation1-present-card-30epoch-20261126-01'
CARD_ROOT = r'G:\lxy\shared-m-cipher-nd-runs\20261126-present-card-30epoch-01'
CARD_SLOT = 'M-structure-best'
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

    @property
    def slots(self) -> tuple[str] | tuple[str, str]:
        return SLOTS

    @property
    def table_hashes(self) -> tuple[str, ...]:
        return TABLE_SHA256

    @property
    def output_root(self) -> str:
        return ROOT

    @property
    def panel_row_bytes(self) -> int:
        return 17 + 8 * len(self.slots)  # two 64-bit ciphertexts, label, logits per slot

    def archive_reader_recipe(self) -> ReplayRecipe:
        # The reader describes the original four-task format and shuffle domain;
        # the provider below exposes only the PRESENT view of that archive.
        return replace(ReplayRecipe(), archive=self.archive, skinny_source=self.skinny_source,
                       training=self.training)

    def to_dict(self) -> dict[str, Any]:
        value = super().to_dict()
        del value['model']['skinny_nibble']
        value.update(output_root=self.output_root, gpu_uuid=GPU_UUID, deadline_env=DEADLINE_ENV,
                     slots=list(self.slots), routing=dict(mode='fixed-no-trained-router',
                         weights=dict(ROUTE), public_card='PRESENT-preprocessing-only'),
                     geometry=dict(buffer=NIBBLE_BUFFER, shape=[64, 9], center_slot=4,
                         sentinel=64, encoding='msb0-int64-le-row-major',
                         sha256=dict(zip(self.slots, self.table_hashes, strict=True))),
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


@dataclass(frozen=True)
class PresentCardRecipe(NeighborhoodRecipe):
    method: str = CARD_METHOD
    namespace: str = 'innovation1-present-card-30epoch-v1-20261126'
    run_id: str = CARD_RUN_ID
    training: Training = replace(FOUR_SINGLE.training, epochs=30)
    resources: Resources = replace(FOUR_SINGLE.resources, wall_seconds=14400)
    p_layer: PLayer = PRESENT_P_LAYER
    gather_rule: GatherRule = NIBBLE_RULE

    @property
    def slots(self) -> tuple[str]:
        return (CARD_SLOT,)

    @property
    def table_hashes(self) -> tuple[str, ...]:
        return (STRUCTURE_SHA256,)

    @property
    def output_root(self) -> str:
        return CARD_ROOT

    def to_dict(self) -> dict[str, Any]:
        value = super().to_dict()
        p_layer, rule = asdict(self.p_layer), asdict(self.gather_rule)
        p_layer['source_to_destination'] = list(self.p_layer.source_to_destination)
        rule.update(block_offsets=list(self.gather_rule.block_offsets),
                    bit_offsets=list(self.gather_rule.bit_offsets))
        value['structure_card'] = dict(p_layer=p_layer, gather_rule=rule)
        value['routing']['public_card'] = 'PRESENT-structure-card-preprocessing-only'
        value['selection']['test'] = 'after-one-persisted-strict-best-reload'
        return value


NEIGHBORHOOD = NeighborhoodRecipe()
PRESENT_CARD = PresentCardRecipe()


def official(recipe: NeighborhoodRecipe, run_id: str) -> None:
    fixed = PRESENT_CARD if type(recipe) is PresentCardRecipe else NEIGHBORHOOD
    if (run_id != fixed.run_id or type(recipe) is not type(fixed) or recipe != fixed
            or canonical(recipe.to_dict()) != canonical(fixed.to_dict())):
        raise ValueError('CONFIG_EXACT')


def validate_root(root: PurePath, *, recipe: NeighborhoodRecipe = NEIGHBORHOOD) -> None:
    validate_serial_root(root, expected=recipe.output_root)


def estimate(recipe: NeighborhoodRecipe) -> dict[str, int]:
    n = estimate_pair(recipe)
    tr = recipe.training
    if recipe.tasks != (TASK,) or recipe.skinny_source is None:
        raise ValueError('POOL_RECIPE')
    ledger = n['cycles'] * ROW_BYTES + 2 * (n['cycles'] // BATCH_LEDGER_ROWS + tr.epochs)
    panel = tr.test_groups * recipe.panel_row_bytes + PANEL_OVERHEAD
    peak = ledger + 2 * panel + 3 * MAX_CHECKPOINT + RECORD_REPORT_RESERVE
    if peak > recipe.resources.artifact_bytes:
        raise ValueError('ARTIFACT_BOUND')
    if isinstance(recipe, PresentCardRecipe):
        verified_card(recipe.p_layer, recipe.gather_rule)
        n.update(fits=1, optimizer_updates=n['cycles'],
                 forward_rows=n['train_groups'] + tr.epochs * tr.dev_groups + tr.test_groups)
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

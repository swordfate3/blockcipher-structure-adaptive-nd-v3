"""Exact shared-M replay contract. Safe to import during static check."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import PurePath
from typing import Any

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Stopped
from cipher_lab.training.early_stopping import Settings

from ..config import Resources, Training, canonical
from ..present_neighborhood.config import DEADLINE_ENV, PRESENT_CARD
from ..present_neighborhood.config import GPU_UUID as GPU_UUID
from ..present_neighborhood.layout import DEFAULT_SHA256, STRUCTURE_SHA256, verified_card
from ..serial_bc.config import SERIAL_B, SerialRecipe, validate_root
from .config import FOUR_SINGLE, SLOT, ReplayRecipe, estimate

METHOD = 'innovation1-four-task-card-1m-v1'
RUN_ID = 'innovation1-four-task-card-1m-20261127-01'
ROOT = r'G:\lxy\shared-m-cipher-nd-runs\20261127-four-task-card-1m-01'
PRESENT_BUFFER = 'present_structure'
NIBBLE_BUFFERS = dict(SIMECK='shared.stems.nibble.indices', PRESENT=PRESENT_BUFFER,
                      SIMON='shared.stems.nibble.indices', SKINNY='skinny_column')


@dataclass(frozen=True)
class InterleaveRecipe(ReplayRecipe):
    method: str = METHOD
    namespace: str = 'innovation1-four-task-card-1m-v1-20261127'
    run_id: str = RUN_ID
    training: Training = replace(SERIAL_B.training, epochs=30)
    resources: Resources = replace(FOUR_SINGLE.resources, wall_seconds=7200)
    early_stopping: Settings = Settings(warmup_epochs=10, patience=8, min_delta=1e-5)
    cosine_epochs: int = 30

    def subset_recipe(self) -> SerialRecipe:
        return replace(SERIAL_B, training=self.training, archive=self.archive,
                       skinny_source=self.skinny_source)

    def nibble_lookup(self, routes: dict[str, dict[str, float]]) -> dict[str, str | None]:
        return {t.cipher: NIBBLE_BUFFERS[t.cipher] if 'nibble' in routes[t.cipher] else None
                for t in self.tasks}

    def to_dict(self) -> dict[str, Any]:
        value = super().to_dict()
        data = self.subset_recipe().to_dict()['data']
        original = self.subset_recipe().archive_reader_recipe()
        data.update(subset_namespace=SERIAL_B.namespace, shuffle_namespace=SERIAL_B.namespace,
                    archive_namespace=original.namespace, test_namespace=original.namespace,
                    archive_train_rows=original.training.batches_per_epoch
                    * original.training.batch_groups, archive_dev_rows=original.training.dev_groups)
        value.update(output_root=ROOT, gpu_uuid=GPU_UUID, deadline_env=DEADLINE_ENV,
                     slots=[SLOT], routing=dict(mode='public-card', features=13, top_k=2),
                     geometry=dict(nibble_buffers=dict(NIBBLE_BUFFERS), shape=[64, 9],
                                   encoding='msb0-int64-le-row-major',
                                   sha256={PRESENT_BUFFER: STRUCTURE_SHA256,
                                           NIBBLE_BUFFERS['SIMECK']: DEFAULT_SHA256}),
                     structure_card=PRESENT_CARD.to_dict()['structure_card'], data=data)
        value['optimizer'].update(schedule='cycle-cosine')
        value['model'].update(present_nibble='p-layer-if-selected')
        return value


INTERLEAVE = InterleaveRecipe()


def official(recipe: InterleaveRecipe, run_id: str) -> InterleaveRecipe:
    if (run_id != RUN_ID or type(recipe) is not InterleaveRecipe or recipe != INTERLEAVE
            or canonical(recipe.to_dict()) != canonical(INTERLEAVE.to_dict())):
        raise ValueError('CONFIG_EXACT')
    verified_card(PRESENT_CARD.p_layer, PRESENT_CARD.gather_rule)
    return INTERLEAVE


def validate_launch(root: PurePath, gpu_uuid: str) -> None:
    validate_root(root, expected=ROOT)
    if gpu_uuid.casefold() != GPU_UUID.casefold():
        raise ValueError('INTERLEAVE_GPU_UUID')


def make_policy(recipe: InterleaveRecipe, clocks: Clocks, inherited: str | None) -> Policy:
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

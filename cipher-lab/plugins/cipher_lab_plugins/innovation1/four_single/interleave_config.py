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
from .config import FOUR_SINGLE, NAMESPACE, SLOT, ReplayRecipe, estimate
from .structure_cards import (
    CARDS,
    COLUMN_SHA256,
    WORD_HASHES,
    WORD_ROTATIONS,
    verified_tables,
    verified_word_tables,
    word_offsets,
)

METHOD = 'innovation1-four-task-card-1m-v1'
RUN_ID = 'innovation1-four-task-card-1m-20261127-01'
ROOT = r'G:\lxy\shared-m-cipher-nd-runs\20261127-four-task-card-1m-01'
STRUCTURE_METHOD = 'innovation1-four-task-card-1m-v2'
STRUCTURE_ID = 'innovation1-four-task-card-1m-20261127-02'
STRUCTURE_ROOT = r'G:\lxy\shared-m-cipher-nd-runs\20261127-four-task-card-1m-02'
FULL_METHOD = 'innovation1-four-task-card-10m-v1'
FULL_ID = 'innovation1-four-task-card-10m-20261127-01'
FULL_ROOT = r'G:\lxy\shared-m-cipher-nd-runs\20261127-four-task-card-10m-01'
WORD_METHOD = 'innovation1-feistel-word-tables-10m-v1'
WORD_ID = 'innovation1-feistel-word-tables-10m-20261128-01'
WORD_ROOT = r'G:\lxy\shared-m-cipher-nd-runs\20261128-feistel-word-tables-10m-01'
INTERLEAVE_METHODS = frozenset({METHOD, STRUCTURE_METHOD, FULL_METHOD, WORD_METHOD})
DEFAULT_WORD_BUFFER = 'shared.stems.word.indices'
WORD_BUFFERS = {'SIMECK': 'simeck_word', 'SIMON': 'simon_word'}
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

    @property
    def output_root(self) -> str:
        return ROOT

    def subset_recipe(self) -> SerialRecipe:
        return replace(SERIAL_B, training=self.training, archive=self.archive,
                       skinny_source=self.skinny_source)

    def archive_reader_recipe(self) -> ReplayRecipe:
        return self.subset_recipe().archive_reader_recipe()

    def nibble_lookup(self, routes: dict[str, dict[str, float]]) -> dict[str, str | None]:
        return {t.cipher: NIBBLE_BUFFERS[t.cipher] if 'nibble' in routes[t.cipher] else None
                for t in self.tasks}

    def to_dict(self) -> dict[str, Any]:
        value = super().to_dict()
        data = self.subset_recipe().to_dict()['data']
        original = self.archive_reader_recipe()
        data.update(subset_namespace=SERIAL_B.namespace, shuffle_namespace=SERIAL_B.namespace,
                    archive_namespace=original.namespace, test_namespace=original.namespace,
                    archive_train_rows=original.training.batches_per_epoch
                    * original.training.batch_groups, archive_dev_rows=original.training.dev_groups)
        value.update(output_root=self.output_root, gpu_uuid=GPU_UUID, deadline_env=DEADLINE_ENV,
                     slots=[SLOT], routing=dict(mode='public-card', features=13, top_k=2),
                     geometry=dict(nibble_buffers=dict(NIBBLE_BUFFERS), shape=[64, 9],
                                   encoding='msb0-int64-le-row-major',
                                   sha256={PRESENT_BUFFER: STRUCTURE_SHA256,
                                           NIBBLE_BUFFERS['SIMECK']: DEFAULT_SHA256}),
                     structure_card=PRESENT_CARD.to_dict()['structure_card'], data=data)
        value['optimizer'].update(schedule='cycle-cosine')
        value['model'].update(present_nibble='p-layer-if-selected')
        return value


@dataclass(frozen=True)
class StructureInterleaveRecipe(InterleaveRecipe):
    method: str = STRUCTURE_METHOD
    namespace: str = 'innovation1-four-task-card-1m-v2-20261127'
    run_id: str = STRUCTURE_ID

    @property
    def output_root(self) -> str:
        return STRUCTURE_ROOT

    def to_dict(self) -> dict[str, Any]:
        value = super().to_dict()
        value['structure_card'] = {cipher: card.to_dict() for cipher, card in CARDS.items()}
        value['geometry']['sha256'][NIBBLE_BUFFERS['SKINNY']] = COLUMN_SHA256
        return value


@dataclass(frozen=True)
class FullStructureInterleaveRecipe(StructureInterleaveRecipe):
    method: str = FULL_METHOD
    namespace: str = 'innovation1-four-task-card-10m-v1-20261127'
    run_id: str = FULL_ID
    training: Training = replace(FOUR_SINGLE.training, epochs=30)
    resources: Resources = replace(FOUR_SINGLE.resources, wall_seconds=16 * 3600)

    @property
    def output_root(self) -> str:
        return FULL_ROOT

    def archive_reader_recipe(self) -> ReplayRecipe:
        return replace(ReplayRecipe(), namespace=NAMESPACE, archive=self.archive,
                       skinny_source=self.skinny_source, training=self.training)

    def to_dict(self) -> dict[str, Any]:
        value = super().to_dict()
        data = value['data']
        del data['subset_namespace']
        data.update(protocol='reuse-original-90-10-full-v1', shuffle_namespace=NAMESPACE,
                    skinny_shuffle_namespace=self.archive_reader_recipe().to_dict()['data'][
                        'SKINNY']['shuffle_namespace'])
        return value


@dataclass(frozen=True)
class FeistelWordRecipe(FullStructureInterleaveRecipe):
    method: str = WORD_METHOD
    namespace: str = 'innovation1-feistel-word-tables-10m-v1-20261128'
    run_id: str = WORD_ID

    @property
    def output_root(self) -> str:
        return WORD_ROOT

    def word_lookup(self, routes: dict[str, dict[str, float]]) -> dict[str, str | None]:
        return {t.cipher: WORD_BUFFERS.get(t.cipher, DEFAULT_WORD_BUFFER)
                if 'word' in routes[t.cipher] else None for t in self.tasks}

    def to_dict(self) -> dict[str, Any]:
        value = super().to_dict()
        value['model']['feistel_word'] = 'rotation-first-if-selected'
        value['geometry']['word_buffers'] = {
            t.cipher: WORD_BUFFERS.get(t.cipher, DEFAULT_WORD_BUFFER) for t in self.tasks}
        for cipher, name in WORD_BUFFERS.items():
            value['geometry']['sha256'][name] = WORD_HASHES[cipher]
            value['structure_card'][cipher]['word'] = dict(
                input_order='MSB0', word_bits=32, rotations=list(WORD_ROTATIONS[cipher]),
                offsets=list(word_offsets(cipher)), center_slot=0, other_word_slot=8,
                word_boundary='wrap')
        return value


@dataclass(frozen=True)
class FeistelWordRetryRecipe(FeistelWordRecipe):
    run_id: str = 'innovation1-feistel-word-tables-10m-20261128-02'

    @property
    def output_root(self) -> str:
        return WORD_ROOT.removesuffix('-01') + '-02'


INTERLEAVE = InterleaveRecipe()
STRUCTURE_INTERLEAVE = StructureInterleaveRecipe()
FULL_INTERLEAVE = FullStructureInterleaveRecipe()
WORD_INTERLEAVE = FeistelWordRecipe()
WORD_RETRY = FeistelWordRetryRecipe()


def official(recipe: InterleaveRecipe, run_id: str) -> InterleaveRecipe:
    if type(recipe) is FeistelWordRetryRecipe:
        expected: InterleaveRecipe = WORD_RETRY
    elif type(recipe) is FeistelWordRecipe:
        expected = WORD_INTERLEAVE
    elif type(recipe) is FullStructureInterleaveRecipe:
        expected = FULL_INTERLEAVE
    else:
        expected = STRUCTURE_INTERLEAVE if type(recipe) is StructureInterleaveRecipe else INTERLEAVE
    if (run_id != expected.run_id or type(recipe) is not type(expected) or recipe != expected
            or canonical(recipe.to_dict()) != canonical(expected.to_dict())):
        raise ValueError('CONFIG_EXACT')
    if isinstance(expected, FeistelWordRecipe):
        verified_word_tables()
    if isinstance(expected, StructureInterleaveRecipe):
        verified_tables()
    else:
        verified_card(PRESENT_CARD.p_layer, PRESENT_CARD.gather_rule)
    return expected


def validate_launch(root: PurePath, gpu_uuid: str, *,
                    recipe: InterleaveRecipe = INTERLEAVE) -> None:
    validate_root(root, expected=recipe.output_root)
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

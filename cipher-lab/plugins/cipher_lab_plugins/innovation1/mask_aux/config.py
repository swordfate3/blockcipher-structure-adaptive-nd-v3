"""Frozen public recipe, with no numerical imports."""
from __future__ import annotations

from ..prefix_bridge import config as bridge

BATCH_SIZE = bridge.BATCH_SIZE
CHECKPOINT_BYTES = bridge.CHECKPOINT_BYTES
DEV_ROWS = bridge.DEV_ROWS
EPOCHS = bridge.EPOCHS
MIN_DELTA = bridge.MIN_DELTA
MIN_EPOCHS = bridge.MIN_EPOCHS
ORDER_SEED = bridge.ORDER_SEED
OUTPUT_BYTES = bridge.OUTPUT_BYTES
PATIENCE = bridge.PATIENCE
RSS_BYTES = bridge.RSS_BYTES
TEST_ROWS = bridge.TEST_ROWS
TRAIN_ROWS = bridge.TRAIN_ROWS
WARMUP = bridge.WARMUP

METHOD = 'innovation1-present-mask-aux-v1'
ROUNDS = (7,)
SLOTS = BASELINE, AUXILIARY = ('r7-baseline', 'r7-mask-aux')
PANEL = 'r7-cipher'
KAT_BLOCKS = 4
KAT_NAME = 'PRESENT-80/official-zero-r31+public-nonzero-r7/scalar-batch'
WALL_SECONDS, CPU_SECONDS = 3600, 14400
MASK_SEED, MASK_SITES = 20261127, 16
AUX_WEIGHT = .1
TARGETS_PER_ROW = bridge.PAIRS_PER_GROUP * MASK_SITES
AUX_PARAMETERS = 2 * (bridge.CHANNELS + 1)


def slot_input(slot: str) -> tuple[int, int]:
    if slot not in SLOTS:
        raise ValueError('INVALID_SLOT')
    return 7, 56


def blocks(rows: int) -> int:
    return rows * bridge.PAIRS_PER_GROUP * 2


def counts(a: int, b: int) -> bridge.Counts:
    if any(type(e) is not int or not MIN_EPOCHS <= e <= EPOCHS for e in (a, b)):
        raise ValueError('invalid completed epochs')
    return bridge.Counts(
        real_blocks=KAT_BLOCKS + blocks(TRAIN_ROWS + DEV_ROWS + TEST_ROWS), null_words=0,
        optimizer_updates=TRAIN_ROWS // BATCH_SIZE * (a + b),
        forward_rows=(TRAIN_ROWS + DEV_ROWS) * (a + b) + TRAIN_ROWS * b + 2 * TEST_ROWS,
    )


CAPS = counts(EPOCHS, EPOCHS)


def recipe() -> dict[str, object]:
    value = bridge.recipe()
    del value['s2_boundary']
    trajectories = blocks(TRAIN_ROWS + DEV_ROWS + TEST_ROWS)
    value.update(method=METHOD, rounds=list(ROUNDS), slots=list(SLOTS), fits=len(SLOTS),
                 panel=PANEL, mask_seed=MASK_SEED, mask_sites_per_pair=MASK_SITES,
                 mask_schedule='public-seed/one-based-step/uniform-sites-and-endpoints',
                 masked_channels=['flag-left', 'flag-right', 'flag-xor', 'left', 'right', 'xor'],
                 aux_targets_per_group=TARGETS_PER_ROW, aux_weight=AUX_WEIGHT,
                 aux_head=[bridge.CHANNELS, 2], kat=KAT_NAME, kat_real_blocks=KAT_BLOCKS,
                 roundstateoutputs=trajectories, trajectories=trajectories,
                 count_caps=dict(CAPS), wall_seconds=WALL_SECONDS, cpu_seconds=CPU_SECONDS)
    return value

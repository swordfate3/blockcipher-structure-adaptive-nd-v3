"""Frozen pair protocol; the exact check path uses only the standard library."""
from __future__ import annotations

from ..mask_aux import config as group
from ..prefix_bridge import config as bridge

METHOD = 'innovation1-present-single-pair-mask-10m-v1'
SLOTS = BASELINE, AUXILIARY = ('r7-pair-baseline', 'r7-pair-mask-aux')
PANEL = group.PANEL
ROUNDS = group.ROUNDS
TRAIN_ROWS, DEV_ROWS, TEST_ROWS, BATCH_SIZE = 5_000_000, 32_768, 131_072, 512
EPOCHS = 30
ORDER_SEED = bridge.ORDER_SEED
DATA_SEED = bridge.DATA_SEED
DIFFERENCE = bridge.DIFFERENCE
CHECKPOINT_BYTES, OUTPUT_BYTES, RSS_BYTES = (
    group.CHECKPOINT_BYTES, group.OUTPUT_BYTES, group.RSS_BYTES)
WALL_SECONDS, CPU_SECONDS = 43_200, 172_800
KAT_BLOCKS, KAT_NAME = group.KAT_BLOCKS, group.KAT_NAME
TARGETS_PER_ROW = group.MASK_SITES
AUX_PARAMETERS = group.AUX_PARAMETERS


def slot_input(slot: str) -> tuple[int, int]:
    if slot not in SLOTS:
        raise ValueError('INVALID_SLOT')
    return 7, 56


def blocks(rows: int) -> int:
    return 2 * rows


def steps_per_epoch() -> int:
    return (TRAIN_ROWS + BATCH_SIZE - 1) // BATCH_SIZE


def counts(a: int, b: int) -> bridge.Counts:
    if any(type(e) is not int or e != EPOCHS for e in (a, b)):
        raise ValueError('invalid completed epochs')
    return bridge.Counts(
        real_blocks=KAT_BLOCKS + blocks(TRAIN_ROWS + DEV_ROWS + TEST_ROWS), null_words=0,
        optimizer_updates=steps_per_epoch() * (a + b),
        forward_rows=(TRAIN_ROWS + DEV_ROWS) * (a + b) + TRAIN_ROWS * b + 2 * TEST_ROWS,
    )


CAPS = counts(EPOCHS, EPOCHS)


def recipe() -> dict[str, object]:
    value = group.recipe()
    del value['pairs_per_group'], value['aux_targets_per_group']
    trajectories = blocks(TRAIN_ROWS + DEV_ROWS + TEST_ROWS)
    totals = counts(EPOCHS, EPOCHS)
    value.update(
        method=METHOD, slots=list(SLOTS), unit='pair', train_rows=TRAIN_ROWS,
        train_ciphertexts=blocks(TRAIN_ROWS), dev_rows=DEV_ROWS, test_rows=TEST_ROWS,
        batch_size=BATCH_SIZE, steps_per_epoch=steps_per_epoch(),
        train_tail_rows=TRAIN_ROWS % BATCH_SIZE, max_epochs=EPOCHS, min_epochs=EPOCHS,
        selection='pair-DEV-class-BCE/earliest-min', early_stop=None,
        input_layout='pair,1,flag-left/flag-right/flag-xor/left/right/xor,MSB0',
        data_schedule='SeedSequence([data_seed,split_id,stream])/train=0,dev=1,test=2/'
                      'balanced-permutation=0,endpoints=1',
        order_schedule='SeedSequence([order_seed,zero-based-epoch])',
        mask_schedule='SeedSequence([mask_seed,one-based-arm-step])/pair-sites-and-endpoints',
        cross_split='exact-unordered-pair/fail-closed', aux_targets_per_pair=TARGETS_PER_ROW,
        roundstateoutputs=trajectories, trajectories=trajectories, count_caps=dict(totals),
        wall_seconds=WALL_SECONDS, cpu_seconds=CPU_SECONDS, primary='correct-pairs/TEST-pairs',
    )
    return value

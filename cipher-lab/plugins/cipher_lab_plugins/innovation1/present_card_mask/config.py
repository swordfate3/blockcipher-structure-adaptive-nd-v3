"""Frozen one-fit masked-card recipe."""
from __future__ import annotations

from ..mask_aux import config as masking
from ..prefix_bridge.config import Counts
from ..present_card_fresh import config as pure
from ..present_card_fresh.config import (
    BATCH_SIZE,
    CHECKPOINT_BYTES,
    CPU_SECONDS,
    DATA_SEED,
    DEV_ROWS,
    DIFFERENCE,
    EPOCHS,
    KAT_BLOCKS,
    KAT_NAME,
    MIN_DELTA,
    MIN_EPOCHS,
    ORDER_SEED,
    OUTPUT_BYTES,
    PATIENCE,
    ROUNDS,
    RSS_BYTES,
    TEST_ROWS,
    TRAIN_ROWS,
    WALL_SECONDS,
    WARMUP,
)

__all__ = [
    'BATCH_SIZE', 'CHECKPOINT_BYTES', 'CPU_SECONDS', 'DATA_SEED', 'DEV_ROWS',
    'DIFFERENCE', 'EPOCHS', 'KAT_BLOCKS', 'KAT_NAME', 'METHOD', 'MIN_DELTA', 'MIN_EPOCHS',
    'ORDER_SEED', 'OUTPUT_BYTES', 'PARAMETERS', 'PATIENCE', 'ROUNDS', 'RSS_BYTES',
    'SLOT', 'TEST_ROWS', 'TRAIN_ROWS', 'WALL_SECONDS', 'WARMUP', 'counts',
]

METHOD = 'innovation1-present-card-mask-1m-v1'
SLOT = 'r7-present-card-mask'
SLOTS = (SLOT,)
MASK_SEED, MASK_SITES, AUX_WEIGHT = masking.MASK_SEED, masking.MASK_SITES, masking.AUX_WEIGHT
HEADS = 2
WIDTH = 96
FLAG_WIDTH = 27
PARAMETERS = pure.PARAMETERS + 2 * FLAG_WIDTH * WIDTH + (WIDTH + 1) * HEADS


def counts(epochs: int, *, test: bool = True) -> Counts:
    base = pure.counts(epochs, test=test)
    return {**base, 'forward_rows': base['forward_rows'] + epochs * TRAIN_ROWS}


CAPS = counts(EPOCHS)


def recipe() -> dict[str, object]:
    return {**pure.recipe(), 'method': METHOD, 'slots': list(SLOTS), 'auxiliary': True,
            'parameters': PARAMETERS, 'count_caps': dict(CAPS), 'mask_seed': MASK_SEED,
            'mask_sites_per_pair': MASK_SITES, 'aux_weight': AUX_WEIGHT,
            'mask_schedule': 'SeedSequence([mask_seed,one-based-update])/sites-and-endpoints',
            'masked_view': ['flag-left', 'flag-right', 'flag-xor', 'left', 'right', 'xor'],
            'aux_head': [WIDTH, HEADS], 'flag_projection': [2, FLAG_WIDTH, WIDTH],
            'max_auxiliary_targets': EPOCHS * TRAIN_ROWS * MASK_SITES}

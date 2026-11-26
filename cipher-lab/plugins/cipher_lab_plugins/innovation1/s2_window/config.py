"""Frozen public recipe, stdlib only; no private protocol choices or overrides."""
from __future__ import annotations

from ..prefix_bridge import config as parent
from ..prefix_bridge.config import Counts

TRAIN_ROWS, DEV_ROWS, TEST_ROWS = parent.TRAIN_ROWS, parent.DEV_ROWS, parent.TEST_ROWS
BATCH_SIZE, PAIRS_PER_GROUP = parent.BATCH_SIZE, parent.PAIRS_PER_GROUP
DIFFERENCE, S2_BOUNDARY = parent.DIFFERENCE, parent.S2_BOUNDARY
DATA_SEED, ORDER_SEED = parent.DATA_SEED, parent.ORDER_SEED
EPOCHS, MIN_EPOCHS = parent.EPOCHS, parent.MIN_EPOCHS
WARMUP, PATIENCE, MIN_DELTA = parent.WARMUP, parent.PATIENCE, parent.MIN_DELTA
GPU_DEVICE, GPU_UUID = parent.GPU_DEVICE, parent.GPU_UUID
RSS_BYTES, OUTPUT_BYTES = parent.RSS_BYTES, parent.OUTPUT_BYTES
CHECKPOINT_BYTES = parent.CHECKPOINT_BYTES

METHOD = 'innovation1-present-s2-window-v1'
SLOT = 'r7-cipher'
SLOTS = (SLOT,)
ROUNDS = (2, 7)
MASK = 0b111000
PARAMETERS = 1_095_285
KAT_BLOCKS = 5
WALL_SECONDS, CPU_SECONDS = 3600, 14400
KAT_NAME = 'public-PRESENT-80-whitened-S2-roundtrip-total-7-reference'


def trajectories(rows: int) -> int:
    return rows * PAIRS_PER_GROUP * 2


def counts(epochs: int, *, test: bool = True) -> Counts:
    if type(epochs) is not int or not MIN_EPOCHS <= epochs <= EPOCHS:
        raise ValueError('invalid completed epochs')
    return Counts(real_blocks=KAT_BLOCKS + 3 * trajectories(
                      TRAIN_ROWS + DEV_ROWS + (TEST_ROWS if test else 0)),
                  null_words=0, optimizer_updates=(TRAIN_ROWS // BATCH_SIZE) * epochs,
                  forward_rows=(TRAIN_ROWS + DEV_ROWS) * epochs + (TEST_ROWS if test else 0))


CAPS = counts(EPOCHS)


def recipe() -> dict[str, object]:
    return {
        'method': METHOD, 'cipher': 'PRESENT-80', 'total_rounds': ROUNDS[-1],
        'rounds': list(ROUNDS), 'slots': list(SLOTS), 'fits': 1, 'fresh': True,
        's2_boundary': S2_BOUNDARY, 'difference': DIFFERENCE, 'difference_at_round': ROUNDS[0],
        'sampling': ('balanced-per-split/S2-positive-xor/S2-negative-independent/'
                     'both-invert-then-full-forward'),
        'cross_split_pairs': 'disjoint-unordered-S2-pairs/no-retry',
        'pairs_per_group': PAIRS_PER_GROUP, 'label_columns': parent.LABEL_COLUMNS,
        'train_rows': TRAIN_ROWS, 'dev_rows': DEV_ROWS, 'test_rows': TEST_ROWS,
        'secret_key_draw_bits': 80, 'secret_key_draws': 1,
        'data_seed': DATA_SEED, 'order_seed': ORDER_SEED, 'init_seed': parent.INIT_SEED,
        'input': 'C7-only/canonical-left-right-xor', 'mask': MASK, 'parameters': PARAMETERS,
        'stem_inputs': parent.STEM_INPUTS, 'channels': parent.CHANNELS,
        'norm_groups': parent.NORM_GROUPS, 'public_features': parent.PUBLIC_FEATURES,
        'router_top_k': parent.ROUTER_TOP_K, 'head_hidden': parent.HEAD_HIDDEN,
        'router_hidden': parent.ROUTER_HIDDEN, 'suffix_kernels': list(parent.SUFFIX_KERNELS),
        'batch_size': BATCH_SIZE, 'max_epochs': EPOCHS, 'min_epochs': MIN_EPOCHS,
        'optimizer': 'Adam', 'lr': parent.LR, 'betas': list(parent.BETAS), 'eps': parent.EPS,
        'clip': parent.CLIP, 'weight_decay': parent.WEIGHT_DECAY,
        'dtype': 'FP32', 'amp': False, 'tf32': False, 'selection': 'group-DEV-BCE/earliest-min',
        'early_stop': {'warmup_epochs': WARMUP, 'patience': PATIENCE, 'min_delta': MIN_DELTA},
        'gate': KAT_NAME, 'kat_real_blocks': KAT_BLOCKS,
        'trajectories': trajectories(TRAIN_ROWS + DEV_ROWS + TEST_ROWS),
        'per_trajectory': {'inverse_outputs': 1, 'full_forward_outputs': 2},
        'count_caps': dict(CAPS), 'wall_seconds': WALL_SECONDS, 'cpu_seconds': CPU_SECONDS,
        'rss_bytes': RSS_BYTES, 'output_bytes': OUTPUT_BYTES, 'checkpoint_bytes': CHECKPOINT_BYTES,
        'gpu_device': GPU_DEVICE, 'gpu_uuid': GPU_UUID,
        'primary': 'correct-groups/TEST-groups', 'reference_accuracy': .5, 'retry_count': 0,
    }

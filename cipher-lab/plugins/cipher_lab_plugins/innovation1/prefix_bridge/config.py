"""Frozen stdlib-only public recipe and successful-run accounting."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

METHOD = 'innovation1-present-prefix2-bridge-v1'
ROUNDS = (2, 3, 4, 5, 6, 7)
SLOTS = ('s2-only', *(f'r{r}-{mode}' for r in ROUNDS for mode in ('cipher', 'joint')))
FITS = len(SLOTS)
TRAIN_ROWS, DEV_ROWS, TEST_ROWS, BATCH_SIZE = 32768, 8192, 32768, 128
PAIRS_PER_GROUP = 4
DIFFERENCE = 0xD00000
DATA_SEED, ORDER_SEED, INIT_SEED = 20261125, 20261126, 31
STEM_INPUTS, CHANNELS, NORM_GROUPS = 54, 96, 8
PUBLIC_FEATURES, ROUTER_TOP_K = 13, 2
HEAD_HIDDEN, ROUTER_HIDDEN = 64, 16
SUFFIX_KERNELS = (3, 5, 7, 9, 11)
LABEL_COLUMNS = 1
S2_BOUNDARY = 'PRESENT-80/r2/with-pLayer/with-K3-final-whitening'
EPOCHS, WARMUP, PATIENCE, MIN_DELTA = 30, 10, 8, 1e-5
MIN_EPOCHS = WARMUP + PATIENCE
LR, BETAS, EPS, CLIP, WEIGHT_DECAY = .001, (.9, .999), 1e-8, 1., 0.
KAT_BLOCKS = 14
TRAJECTORIES = (TRAIN_ROWS + DEV_ROWS + TEST_ROWS) * PAIRS_PER_GROUP * 2
ROUNDSTATEOUTPUTS = TRAJECTORIES * len(ROUNDS)
WALL_SECONDS, CPU_SECONDS = 14400, 57600
RSS_BYTES, OUTPUT_BYTES, CHECKPOINT_BYTES = 8 * 2**30, 512 * 2**20, 16 * 2**20
GPU_UUID = 'GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5'
GPU_DEVICE = 'cuda:0'


class Counts(TypedDict):
    real_blocks: int
    null_words: int
    optimizer_updates: int
    forward_rows: int


def counts(epochs13: Sequence[int]) -> Counts:
    """Expected totals only after 13 successful fits and complete shared TEST."""
    if (len(epochs13) != FITS
            or any(type(e) is not int or not MIN_EPOCHS <= e <= EPOCHS for e in epochs13)):
        raise ValueError('invalid completed epochs')
    epochs = sum(epochs13)
    return Counts(real_blocks=KAT_BLOCKS + ROUNDSTATEOUTPUTS, null_words=0,
                  optimizer_updates=(TRAIN_ROWS // BATCH_SIZE) * epochs,
                  forward_rows=(TRAIN_ROWS + DEV_ROWS) * epochs + TEST_ROWS * FITS)


CAPS = counts((EPOCHS,) * FITS)


def recipe() -> dict[str, object]:
    return {
        'method': METHOD, 'cipher': 'PRESENT-80', 'rounds': list(ROUNDS),
        'slots': list(SLOTS), 'fits': FITS, 's2_boundary': S2_BOUNDARY,
        'pairs_per_group': PAIRS_PER_GROUP, 'label_columns': LABEL_COLUMNS,
        'train_rows': TRAIN_ROWS, 'dev_rows': DEV_ROWS, 'test_rows': TEST_ROWS,
        'difference': DIFFERENCE, 'secret_key_draw_bits': 80, 'secret_key_draws': 1,
        'sampling': 'balanced-per-split/positive-xor/negative-independent/both-encrypted',
        'data_seed': DATA_SEED, 'order_seed': ORDER_SEED, 'init_seed': INIT_SEED,
        'stem_inputs': STEM_INPUTS, 'channels': CHANNELS, 'norm_groups': NORM_GROUPS,
        'public_features': PUBLIC_FEATURES, 'router_top_k': ROUTER_TOP_K,
        'head_hidden': HEAD_HIDDEN, 'router_hidden': ROUTER_HIDDEN,
        'suffix_kernels': list(SUFFIX_KERNELS), 'batch_size': BATCH_SIZE,
        'max_epochs': EPOCHS, 'min_epochs': MIN_EPOCHS, 'optimizer': 'Adam',
        'lr': LR, 'betas': list(BETAS), 'eps': EPS, 'clip': CLIP, 'weight_decay': WEIGHT_DECAY,
        'dtype': 'FP32', 'amp': False, 'tf32': False, 'selection': 'group-DEV-BCE/earliest-min',
        'early_stop': {'warmup_epochs': WARMUP, 'patience': PATIENCE, 'min_delta': MIN_DELTA},
        'kat_real_blocks': KAT_BLOCKS, 'roundstateoutputs': ROUNDSTATEOUTPUTS,
        'trajectories': TRAJECTORIES, 'count_caps': dict(CAPS),
        'wall_seconds': WALL_SECONDS, 'cpu_seconds': CPU_SECONDS, 'rss_bytes': RSS_BYTES,
        'output_bytes': OUTPUT_BYTES, 'checkpoint_bytes': CHECKPOINT_BYTES,
        'gpu_device': GPU_DEVICE, 'gpu_uuid': GPU_UUID,
        'primary': 'correct-groups/TEST-groups', 'retry_count': 0,
    }


def slot_input(slot: str) -> tuple[int | None, int]:
    """Public domain: final-round identity (absent for S2) and six channel bits."""
    if slot == 's2-only':
        return None, 0b000111
    for r in ROUNDS:
        if slot == f'r{r}-cipher':
            return r, 0b111000
        if slot == f'r{r}-joint':
            return r, 0b111111
    raise ValueError('INVALID_SLOT')

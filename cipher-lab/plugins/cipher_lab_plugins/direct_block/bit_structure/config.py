"""Frozen single-bit comparison and complete-work counters."""
from __future__ import annotations

from collections.abc import Sequence

from ..cube_compare.config import BETAS, CLIP, EPS, GPU_DEVICE, GPU_UUID, LR, WEIGHT_DECAY
from ..split8.config import Counts

METHOD = 'direct-block-simon32-bit-structure-v1'
TRAIN, DEV, TEST, BATCH = 65_536, 8_192, 32_768, 256
REPEATS, EPOCHS = 2, 30
WARMUP, PATIENCE, MIN_DELTA = 10, 8, 1e-5
MIN_EPOCHS = WARMUP + PATIENCE
DATA_SEED, ORDER_SEED, INIT_SEEDS = 20261128, 20261129, (71, 73)
ARMS = ('mlp', 'correct', 'shuffled')
SLOTS = tuple(f'r{r + 1}-{arm}' for r in range(REPEATS) for arm in ARMS)
PARAMS = {'mlp': 21_255, 'correct': 21_239, 'shuffled': 21_239}
KAT_BLOCKS = 2
WALL_SECONDS, CPU_SECONDS = 5_400, 21_600
RSS_BYTES, OUTPUT_BYTES, CHECKPOINT_BYTES = 8 * 2**30, 256 * 2**20, 16 * 2**20
ROUNDS, TARGET_MSB0 = 3, 0
SHUFFLE_SHA256 = 'db685b4b3dbafcbcf33b92ed6a7a1840d692afbfd2456c0cb74381856e10b3eb'


def counts(epochs: Sequence[int], *, test: bool = True) -> Counts:
    if len(epochs) > len(SLOTS) or (test and len(epochs) != len(SLOTS)):
        raise ValueError('INVALID_COMPLETED_FITS')
    if any(type(e) is not int or not MIN_EPOCHS <= e <= EPOCHS for e in epochs):
        raise ValueError('INVALID_COMPLETED_EPOCHS')
    total = sum(epochs)
    return Counts(real_blocks=KAT_BLOCKS + REPEATS * (TRAIN + DEV + (TEST if test else 0)),
                  null_words=0, optimizer_updates=total * (TRAIN // BATCH),
                  forward_rows=total * (TRAIN + DEV) + (len(SLOTS) * TEST if test else 0))


CAPS = counts((EPOCHS,) * len(SLOTS))


def recipe() -> dict[str, object]:
    return {
        'method': METHOD, 'cipher': 'SIMON32/64', 'rounds': ROUNDS, 'plaintext_bits': 32,
        'target_column_msb0': TARGET_MSB0, 'label': 'absolute-cipher-bit', 'slots': list(SLOTS),
        'fits': len(SLOTS), 'repeats': REPEATS, 'params': dict(PARAMS),
        'train_rows': TRAIN, 'dev_rows': DEV, 'test_rows': TEST, 'batch_size': BATCH,
        'plaintext_sampling': 'unique-uint32-without-replacement/across-all-six-splits',
        'data_seed': DATA_SEED, 'init_seeds': list(INIT_SEEDS), 'order_seed': ORDER_SEED,
        'epoch_order': 'SeedSequence([order_seed,zero-based-repeat,zero-based-epoch])',
        'key_policy': 'one-distinct-synthetic-key-per-repeat/data-only',
        'shared_panel_within_repeat': True, 'structured_initialization_shared': True,
        'dense_layers': [32, 128, 131, 1], 'node_width': 37, 'node_counts': [6, 3, 1],
        'support_count': 16, 'shuffled_leaf_sha256': SHUFFLE_SHA256,
        'optimizer': 'Adam', 'lr': LR, 'betas': list(BETAS), 'eps': EPS,
        'clip': CLIP, 'weight_decay': WEIGHT_DECAY, 'dtype': 'FP32', 'tf32': False, 'amp': False,
        'max_epochs': EPOCHS, 'min_epochs': MIN_EPOCHS,
        'early_stop': {'warmup_epochs': WARMUP, 'patience': PATIENCE, 'min_delta': MIN_DELTA},
        'selection': 'single-bit-DEV-BCE/earliest-min',
        'test_admission': 'after-all-six-strict-disk-restores',
        'primary_contrast': ['correct', 'shuffled'], 'primary_metric': 'TEST BCE',
        'constant_baseline': 'TRAIN Jeffreys probability and majority/tie-zero',
        'kat_real_blocks': KAT_BLOCKS, 'count_caps': dict(CAPS),
        'wall_seconds': WALL_SECONDS, 'cpu_seconds': CPU_SECONDS, 'rss_bytes': RSS_BYTES,
        'output_bytes': OUTPUT_BYTES, 'checkpoint_bytes': CHECKPOINT_BYTES,
        'gpu_device': GPU_DEVICE, 'gpu_uuid': GPU_UUID, 'retry_count': 0,
    }

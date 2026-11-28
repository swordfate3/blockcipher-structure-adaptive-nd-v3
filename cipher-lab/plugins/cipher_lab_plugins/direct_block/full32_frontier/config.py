"""Frozen full-word round comparison and work accounting."""
from __future__ import annotations

import math

from ..bit_structure import config as bit
from ..cube_compare.config import BETAS, CLIP, EPS, GPU_DEVICE, GPU_UUID, LR, WEIGHT_DECAY
from ..split8.config import Counts

METHOD = 'direct-block-simon32-full32-frontier-v1'
ROUNDS = (3, 4, 5, 6)
ARMS = ('correct', 'shuffled')
REPEATS = 2
INIT_SEEDS = (71, 73)
ORDER_SEED = 20261129
TRAIN, DEV, TEST, BATCH, EPOCHS = 65536, 8192, 65536, 256, 30
PANEL_SLOTS = tuple(f'r{index + 1}-{arm}' for index in range(REPEATS) for arm in ARMS)
SLOTS = tuple(f'c{depth}-{slot}' for depth in ROUNDS for slot in PANEL_SLOTS)
PARAMETERS = dict(zip(ROUNDS, (22645, 30156, 37667, 45178), strict=True))
RADIUS = math.sqrt(math.log(2 * len(SLOTS) / .05) / (2 * TEST))
KAT_BLOCKS = 2
WALL_SECONDS, CPU_SECONDS = 7200, 28800
RSS_BYTES, OUTPUT_BYTES, CHECKPOINT_BYTES = bit.RSS_BYTES, 512 * 2**20, bit.CHECKPOINT_BYTES


def counts(completed: int) -> Counts:
    if type(completed) is not int or not 0 <= completed <= len(ROUNDS):
        raise ValueError('invalid completed rounds')
    fits = completed * len(PANEL_SLOTS)
    return Counts(real_blocks=KAT_BLOCKS + completed * REPEATS * (TRAIN + DEV + TEST),
                  null_words=0, optimizer_updates=fits * EPOCHS * (TRAIN // BATCH),
                  forward_rows=fits * (EPOCHS * (TRAIN + DEV) + TEST))


CAPS = counts(len(ROUNDS))


def recipe() -> dict[str, object]:
    return {
        'method': METHOD, 'cipher': 'SIMON32/64', 'rounds': list(ROUNDS), 'slots': list(SLOTS),
        'arms': list(ARMS), 'repeats': REPEATS, 'fits': len(SLOTS), 'plaintext_bits': 32,
        'output_bits': 32, 'label': 'full-absolute-ciphertext/MSB0', 'node_width': 37,
        'parameters': {str(k): v for k, v in PARAMETERS.items()},
        'shuffle': 'simon32-full32-frontier-shuffle-v1/within-word-digest-order-next',
        'train_rows': TRAIN, 'dev_rows': DEV, 'test_rows': TEST, 'batch_size': BATCH,
        'order_seed': ORDER_SEED, 'init_seeds': list(INIT_SEEDS),
        'epoch_order': 'SeedSequence([order_seed,zero-based-repeat,zero-based-epoch])',
        'plaintext_sampling': ('OS-entropy/uniform-without-replacement/unique-uint32/'
                               'shared-train-dev-across-rounds/fresh-test-per-round'),
        'key_policy': 'two-distinct-synthetic-keys/reused-across-rounds/data-only',
        'optimizer': 'Adam', 'lr': LR, 'betas': list(BETAS), 'eps': EPS,
        'clip': CLIP, 'weight_decay': WEIGHT_DECAY, 'dtype': 'FP32', 'tf32': False, 'amp': False,
        'epochs': EPOCHS, 'selection': 'full-32-bit-DEV-BCE/earliest-min',
        'test_admission': 'per-round-after-all-four-strict-disk-restores',
        'primary_metric': 'correct-bits/(32*TEST-rows)',
        'ci': {'method': 'Hoeffding-Bonferroni', 'alpha': .05, 'family': len(SLOTS),
               'radius': RADIUS, 'unit': 'plaintext'},
        'stop': {'positive_control_correct_lower_gt': .51, 'accuracy_lo': .49, 'accuracy_hi': .51,
                 'minimum_stop_round': 4, 'requires_both_arms_and_panels': True},
        'kat_real_blocks': KAT_BLOCKS, 'count_caps': dict(CAPS),
        'wall_seconds': WALL_SECONDS, 'cpu_seconds': CPU_SECONDS, 'rss_bytes': RSS_BYTES,
        'output_bytes': OUTPUT_BYTES, 'checkpoint_bytes': CHECKPOINT_BYTES,
        'gpu_device': GPU_DEVICE, 'gpu_uuid': GPU_UUID, 'retry_count': 0,
    }

"""Exact, stdlib-only fresh single-pair recipe and completed-work totals."""
from __future__ import annotations

import math

from ..mask_aux import config as gate
from ..prefix_bridge import config as bridge
from ..prefix_bridge.config import Counts
from .layout import NIBBLE_RULE, STRUCTURE_SHA256, verified_card
from .present_public import PRESENT_P_LAYER

METHOD = 'innovation1-present-card-fresh-1m-v1'
REFERENCE = 'ac7181115c131543592140a26b21c2e7b1964e2b'
SLOT = 'r7-present-card'
SLOTS = (SLOT,)
ROUNDS: tuple[int, ...] = (7,)
KAT_BLOCKS, KAT_NAME = gate.KAT_BLOCKS, gate.KAT_NAME
DIFFERENCE = 0xD00000
TRAIN_ROWS, DEV_ROWS, TEST_ROWS, BATCH_SIZE = 500_000, 32_768, 131_072, 5_000
DATA_SEED, ORDER_SEED, INIT_SEED = 20261127, 20261126, 26111741
EPOCHS, WARMUP, PATIENCE, MIN_DELTA = 30, 10, 8, 1e-5
MIN_EPOCHS = WARMUP + PATIENCE
PARAMETERS = 1_087_509
ROUTE = (('nibble', .5), ('word', .5))
LR_MAX, LR_MIN = .001, .0001
BETAS, EPS, CLIP, WEIGHT_DECAY = (.9, .999), 1e-8, 1., 0.
WALL_SECONDS, CPU_SECONDS = 7200, 28800
RSS_BYTES, OUTPUT_BYTES, CHECKPOINT_BYTES = 8 * 2**30, 512 * 2**20, 16 * 2**20
GPU_DEVICE, GPU_UUID = bridge.GPU_DEVICE, bridge.GPU_UUID


def _steps(epochs: int) -> int:
    if type(epochs) is not int or not 1 <= epochs <= EPOCHS:
        raise ValueError('invalid epochs')
    return epochs * (TRAIN_ROWS // BATCH_SIZE)


def learning_rate(step: int) -> float:
    planned = _steps(EPOCHS)
    if type(step) is not int or not 0 <= step < planned:
        raise ValueError('step outside schedule')
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * step / (planned - 1)))


def counts(epochs: int, *, test: bool = True) -> Counts:
    if (type(epochs) is not int or not MIN_EPOCHS <= epochs <= EPOCHS
            or type(test) is not bool):
        raise ValueError('invalid completed fit')
    rows = TRAIN_ROWS + DEV_ROWS
    final = TEST_ROWS if test else 0
    return Counts(real_blocks=KAT_BLOCKS + 2 * (rows + final), null_words=0,
                  optimizer_updates=_steps(epochs), forward_rows=rows * epochs + final)


CAPS = counts(EPOCHS)


def recipe() -> dict[str, object]:
    # Fail the stdlib check on any drift in the full public structure table.
    verified_card(PRESENT_P_LAYER, NIBBLE_RULE)
    return {
        'method': METHOD, 'model_reference': REFERENCE, 'cipher': 'PRESENT-80',
        'rounds': list(ROUNDS), 'slots': list(SLOTS), 'fits': 1, 'fresh': True,
        'difference': DIFFERENCE, 'difference_at': 'plaintext', 'unit': 'pair',
        'sampling': 'balanced-per-split/positive-xor/negative-independent/both-encrypted',
        'cross_split_pairs': 'exact-disjoint-unordered-plaintext-and-C7/no-retry',
        'train_rows': TRAIN_ROWS, 'train_ciphertexts': 2 * TRAIN_ROWS,
        'dev_rows': DEV_ROWS, 'test_rows': TEST_ROWS, 'label_columns': 1,
        'secret_key_draw_bits': 80, 'secret_key_draws': 1,
        'data_seed': DATA_SEED, 'order_seed': ORDER_SEED, 'init_seed': INIT_SEED,
        'data_schedule': 'SeedSequence([data_seed,split_id,stream])/train=0,dev=1,test=2/'
                         'balanced-permutation=0,endpoints=1',
        'order_schedule': 'SeedSequence([order_seed,zero-based-epoch])',
        'input': 'C7-only/N,1,2,64/MSB0/internal-left-right-xor',
        'readout': 'single-pair/four-slot-reshape-only', 'parameters': PARAMETERS,
        'nibble_sha256': STRUCTURE_SHA256, 'word_geometry': 'original',
        'route': dict(ROUTE), 'router_frozen': True, 'auxiliary': False,
        'batch_size': BATCH_SIZE, 'steps_per_epoch': _steps(1),
        'dev_tail_rows': DEV_ROWS % BATCH_SIZE, 'test_tail_rows': TEST_ROWS % BATCH_SIZE,
        'max_epochs': EPOCHS, 'min_epochs': MIN_EPOCHS, 'optimizer': 'Adam',
        'lr_max': LR_MAX, 'lr_min': LR_MIN, 'betas': list(BETAS), 'eps': EPS,
        'clip': CLIP, 'weight_decay': WEIGHT_DECAY, 'amsgrad': False,
        'schedule': 'zero-based-fixed-cosine', 'schedule_steps': _steps(EPOCHS),
        'dtype': 'FP32', 'amp': False, 'tf32': False,
        'selection': 'pair-DEV-class-BCE/earliest-min',
        'early_stop': {'warmup_epochs': WARMUP, 'patience': PATIENCE, 'min_delta': MIN_DELTA},
        'test_admission': 'once-after-disk-sha-and-strict-restore', 'gate': KAT_NAME,
        'kat_real_blocks': KAT_BLOCKS, 'trajectories': CAPS['real_blocks'] - KAT_BLOCKS,
        'count_caps': dict(CAPS), 'wall_seconds': WALL_SECONDS, 'cpu_seconds': CPU_SECONDS,
        'rss_bytes': RSS_BYTES, 'output_bytes': OUTPUT_BYTES, 'checkpoint_bytes': CHECKPOINT_BYTES,
        'gpu_device': GPU_DEVICE, 'gpu_uuid': GPU_UUID,
        'primary': 'correct-pairs/TEST-pairs', 'reference_accuracy': .5, 'retry_count': 0,
    }

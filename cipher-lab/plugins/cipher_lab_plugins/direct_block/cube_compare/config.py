"""Public constants and the fixed recipe; importing this executes no science."""
from __future__ import annotations

from typing import Final

METHOD = 'direct-block-cube-sharedm-v1'
ARMS: Final = ('mlp', 'sharedm')
SLOTS = ('mlp-best', 'sharedm-best')
SPLITS = ('train', 'dev', 'test')
FAMILIES = (131072, 8192, 8192)
PAIRS_PER_FAMILY = 8
BLOCK_BITS = 64
WIDTH = 2 * BLOCK_BITS
KEY_BITS = 128
ROUNDS = 7
ACTIVE_BITS = (0, 21, 42, 60)
MASK = sum(1 << bit for bit in ACTIVE_BITS)
ACTIVE_COLUMNS = tuple(end - bit for end in (63, 127) for bit in ACTIVE_BITS)
EPOCHS, BATCH = 5, 256
DATA_SEED, ORDER_SEED, INIT_SEED = 20261123, 20261124, 31
LR, BETAS, EPS, WEIGHT_DECAY, CLIP = .001, (.9, .999), 1e-8, 0., 1.
CHECKPOINT_BYTES = 16 * 2**20
OUTPUT_BYTES, RSS_BYTES = 512 * 2**20, 8 * 2**30
WALL_SECONDS, CPU_SECONDS = 7200, 28800
CPU_THREADS, INTEROP_THREADS = 4, 1
GPU_PHYSICAL_INDEX, GPU_DEVICE = 1, 'cuda:0'
GPU_UUID = 'GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5'
GPU_FREE_BYTES = 16 * 2**30


def planned_counts() -> dict[str, int]:
    train, dev, test = (n * PAIRS_PER_FAMILY for n in FAMILIES)
    return dict(real_blocks=2 * (train + dev + test), null_words=0,
                optimizer_updates=2 * EPOCHS * (train // BATCH),
                forward_rows=2 * (EPOCHS * (train + dev) + test))


def preflight_counts() -> dict[str, int]:
    return dict(real_blocks=18, null_words=0, optimizer_updates=2, forward_rows=32)


def total_caps() -> dict[str, int]:
    return {key: value + preflight_counts()[key] for key, value in planned_counts().items()}


def recipe() -> dict[str, object]:
    return {
        'method': METHOD, 'cipher': 'SIMON64/128', 'rounds': ROUNDS,
        'key_bits': KEY_BITS, 'key_protocol': 'single-key', 'input_width': WIDTH,
        'target_width': WIDTH, 'input': 'P || P\u2032', 'target': 'C || C\u2032',
        'bit_order': 'MSB0',
        'active_positions_lsb0': list(ACTIVE_BITS), 'xor_mask': f'0x{MASK:016x}',
        'active_columns_msb0': list(ACTIVE_COLUMNS), 'pairs_per_family': PAIRS_PER_FAMILY,
        'families': dict(zip(SPLITS, FAMILIES, strict=True)),
        'pairs': {split: n * PAIRS_PER_FAMILY for split, n in zip(SPLITS, FAMILIES, strict=True)},
        'test_denominator_bits': FAMILIES[2] * PAIRS_PER_FAMILY * len(ACTIVE_COLUMNS),
        'arms': list(ARMS), 'parameters': {'mlp': 49536, 'sharedm': 1876179},
        'epochs': EPOCHS, 'batch_size': BATCH, 'selection': 'active-BCE/earliest',
        'training_loss': 'full-128-BCE', 'optimizer': 'Adam', 'lr': LR, 'betas': list(BETAS),
        'eps': EPS, 'weight_decay': WEIGHT_DECAY, 'grad_clip': CLIP, 'precision': 'FP32',
        'amp': False, 'tf32': False, 'data_seed': DATA_SEED, 'order_seed': ORDER_SEED,
        'init_seed': INIT_SEED, 'scientific_counts': planned_counts(),
        'preflight_counts': preflight_counts(), 'total_caps': total_caps(), 'retries': 0,
        'wall_seconds': WALL_SECONDS, 'cpu_seconds': CPU_SECONDS, 'rss_stop_bytes': RSS_BYTES,
        'output_bytes': OUTPUT_BYTES, 'checkpoint_bytes': CHECKPOINT_BYTES,
        'cpu_threads': CPU_THREADS, 'interop_threads': INTEROP_THREADS,
        'physical_gpu_index': GPU_PHYSICAL_INDEX, 'gpu_uuid': GPU_UUID,
        'gpu_free_bytes_min': GPU_FREE_BYTES,
    }

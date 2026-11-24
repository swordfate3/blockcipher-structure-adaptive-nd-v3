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


REUSE_METHOD = 'direct-block-cube-sharedm-50epoch-reuse-v1'
REUSE_EPOCHS = 50
WARMUP_EPOCHS, PATIENCE, MIN_DELTA = 10, 8, 1e-5
REUSE_WALL_SECONDS, REUSE_CPU_SECONDS = 10800, 43200
SOURCE_RUN_ID = 'cube-sharedm-simon7-20261123-02'
SOURCE_RUN_SHA256 = 'bc2c87a5e2ab919021c7f4cdf556830f99b59380b858352ae72d30c93669f523'
SOURCE_SPLITS = (
    (41943816, 'e01ca03e68a3feaaa98553cbb2d7662bdd943a31e304cc7abcec4ccaaaa1cf8a'),
    (2622216, '116ebebf97f4a1df3685f53bc9aa02d98f8d8d58ab202d3e85477f0ba5b6f2a3'),
    (2622216, 'f00dfe78821c28024b826eb8ad8444ebf1acd37e9525001061fe4b9c4f990ddc'),
)


def source_recipe() -> dict[str, object]:
    return {'run_id': SOURCE_RUN_ID, 'run_sha256': SOURCE_RUN_SHA256,
            'splits': {f'dataset/{split}.npz': {'bytes': size, 'sha256': digest}
                       for split, (size, digest) in zip(SPLITS, SOURCE_SPLITS, strict=True)}}


def reuse_counts(mlp_epochs: int, sharedm_epochs: int) -> dict[str, int]:
    if any(type(n) is not int or not 1 <= n <= REUSE_EPOCHS for n in (mlp_epochs, sharedm_epochs)):
        raise ValueError('invalid completed epochs')
    train, dev, test = (n * PAIRS_PER_FAMILY for n in FAMILIES)
    epochs = mlp_epochs + sharedm_epochs
    return dict(real_blocks=0, null_words=0, optimizer_updates=epochs * (train // BATCH),
                forward_rows=epochs * (train + dev) + 2 * test)


def reuse_recipe() -> dict[str, object]:
    caps = reuse_counts(REUSE_EPOCHS, REUSE_EPOCHS)
    legacy = recipe()
    del legacy['scientific_counts']
    return {**legacy, 'method': REUSE_METHOD, 'mode': 'saved-panels', 'epochs': REUSE_EPOCHS,
            'early_stopping': dict(warmup_epochs=WARMUP_EPOCHS, patience=PATIENCE,
                                   min_delta=MIN_DELTA),
            'source': source_recipe(), 'preflight_counts': dict.fromkeys(caps, 0),
            'max_scientific_counts': caps, 'total_caps': caps,
            'wall_seconds': REUSE_WALL_SECONDS, 'cpu_seconds': REUSE_CPU_SECONDS}


LSTM_METHOD = 'direct-block-cube-lstm-reuse-v1'
LSTM_SLOT = 'lstm-best'


def lstm_counts(epochs: int) -> dict[str, int]:
    if type(epochs) is not int or not 1 <= epochs <= REUSE_EPOCHS:
        raise ValueError('invalid completed epochs')
    train, dev, test = (n * PAIRS_PER_FAMILY for n in FAMILIES)
    return dict(real_blocks=0, null_words=0, optimizer_updates=epochs * (train // BATCH),
                forward_rows=epochs * (train + dev) + test)


def lstm_recipe() -> dict[str, object]:
    caps = lstm_counts(REUSE_EPOCHS)
    return {**reuse_recipe(), 'method': LSTM_METHOD, 'topology': 'lstm-single',
            'arms': ['lstm'], 'fits': 1, 'parameters': {'lstm': 216192},
            'lstm': dict(sequence_steps=64, sequence_order='MSB0 (P_i, P′_i)',
                         input_size=2, hidden_size=128, num_layers=2, batch_first=True,
                         bias=True, dropout=0., bidirectional=False, proj_size=0,
                         initial_state='fresh zeros [2,N,128] per call',
                         readout='h_n[-1] -> Linear(128,128)'),
            'preflight_counts': dict.fromkeys(caps, 0),
            'max_scientific_counts': caps, 'total_caps': caps}

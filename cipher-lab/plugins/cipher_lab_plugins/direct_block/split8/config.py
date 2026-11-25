"""Frozen public recipe and count accounting; standard library only."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

from ..cube_compare.config import (
    BETAS,
    CLIP,
    EPS,
    GPU_DEVICE,
    GPU_FREE_BYTES,
    GPU_UUID,
    LR,
    WEIGHT_DECAY,
)

METHOD = 'direct-block-simon32-split8-v1'
ROUNDS = (1, 2, 3)
COLS = ((0, 1), (2, 3), (4, 5), (6, 7))
CHUNKS = range(len(COLS))
SLOTS = tuple(f'r{r}-b{j}' for r in ROUNDS for j in CHUNKS)
FITS = len(SLOTS)
Split = Literal['train', 'dev', 'test']
SPLITS: tuple[Split, ...] = ('train', 'dev', 'test')
PLAINTEXT_BITS = 32
DATA_SEED, ORDER_SEED, INIT_SEED = 20261123, 20261124, 31
EPOCHS, WARMUP, PATIENCE, MIN_DELTA = 50, 10, 8, 1e-5
MIN_EPOCHS = WARMUP + PATIENCE
KAT_BLOCKS = 2
WALL_SECONDS, CPU_SECONDS = 10800, 43200
RSS_BYTES, OUTPUT_BYTES, CHECKPOINT_BYTES = 8 * 2**30, 512 * 2**20, 16 * 2**20


@dataclass(frozen=True, slots=True)
class Sizes:
    train: int
    dev: int
    test: int
    batch: int

    def __post_init__(self) -> None:
        if (any(type(n) is not int or n <= 0 for n in (self.train, self.dev, self.test, self.batch))
                or self.train % self.batch or self.total > 2**32):
            raise ValueError('invalid split sizes')

    @property
    def total(self) -> int:
        return self.train + self.dev + self.test

    def slices(self) -> dict[Split, slice]:
        return {'train': slice(0, self.train), 'dev': slice(self.train, self.train + self.dev),
                'test': slice(self.train + self.dev, self.total)}


SIZES = Sizes(262144, 32768, 65536, 256)


class Counts(TypedDict):
    real_blocks: int
    null_words: int
    optimizer_updates: int
    forward_rows: int


def counts(epochs12: Sequence[int]) -> Counts:
    if (len(epochs12) != FITS
            or any(type(e) is not int or not MIN_EPOCHS <= e <= EPOCHS for e in epochs12)):
        raise ValueError('invalid completed epochs')
    epochs = sum(epochs12)
    return Counts(real_blocks=KAT_BLOCKS + len(ROUNDS) * SIZES.total, null_words=0,
                  optimizer_updates=SIZES.train // SIZES.batch * epochs,
                  forward_rows=(SIZES.train + SIZES.dev) * epochs + SIZES.test * FITS)


CAPS = counts((EPOCHS,) * FITS)


def recipe() -> dict[str, object]:
    return {
        'method': METHOD, 'cipher': 'SIMON32/64', 'rounds': list(ROUNDS), 'plaintext_bits': 32,
        'target_columns_msb0': [list(pair) for pair in COLS], 'label': 'absolute-cipher-high8',
        'train_rows': SIZES.train, 'dev_rows': SIZES.dev, 'test_rows': SIZES.test,
        'plaintext_sampling': 'unique-uint32-without-replacement/shared-splits',
        'data_seed': DATA_SEED, 'secret_key_draw_bits': 64, 'secret_key_draws': 1,
        'fits': FITS, 'mlp': [32, 128, 128, 2], 'activation': 'ReLU', 'params_per_network': 20994,
        'init_seeds_by_chunk': [INIT_SEED + j for j in CHUNKS], 'order_seed': ORDER_SEED,
        'max_epochs': EPOCHS, 'batch_size': SIZES.batch, 'optimizer': 'Adam',
        'lr': LR, 'betas': list(BETAS), 'eps': EPS, 'clip': CLIP, 'weight_decay': WEIGHT_DECAY,
        'dtype': 'FP32', 'amp': False, 'tf32': False, 'selection': 'two-bit-DEV-BCE/earliest-min',
        'early_stop': {'warmup_epochs': WARMUP, 'patience': PATIENCE, 'min_delta': MIN_DELTA},
        'min_epochs': MIN_EPOCHS, 'kat_real_blocks': KAT_BLOCKS,
        'science_real_blocks': len(ROUNDS) * SIZES.total, 'count_caps': dict(CAPS),
        'wall_seconds': WALL_SECONDS, 'cpu_seconds': CPU_SECONDS, 'rss_bytes': RSS_BYTES,
        'output_bytes': OUTPUT_BYTES, 'checkpoint_bytes': CHECKPOINT_BYTES,
        'gpu_device': GPU_DEVICE, 'gpu_uuid': GPU_UUID, 'gpu_entry_free_bytes': GPU_FREE_BYTES,
        'primary': 'correct-bits/TEST-times-8', 'retry_count': 0,
    }

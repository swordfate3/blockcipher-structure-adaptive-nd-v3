"""Frozen cycle configuration and accounting."""
from __future__ import annotations

from ..bit_structure import config as bit
from ..split8.config import Counts

METHOD = 'direct-block-simon32-bit-frontier-v1'
ROUNDS = (4, 5, 6, 7, 8)
SLOTS = tuple(f'c{depth}-{slot}' for depth in ROUNDS for slot in bit.SLOTS)
ACC_MIN, ACC_MAX, BCE_GAIN_MAX = .49, .51, .005
WALL_SECONDS, CPU_SECONDS = 7200, 28800
OUTPUT_BYTES = 512 * 2**20
STRUCTURED = (28528, 35965, 43476, 50987, 58498)
DENSE = (28535, 35945, 43485, 51025, 58435)


def parameters(depth: int) -> dict[str, int]:
    at = ROUNDS.index(depth)
    return {'correct': STRUCTURED[at], 'shuffled': STRUCTURED[at], 'mlp': DENSE[at]}


def counts(rounds: int, epochs: int) -> Counts:
    if (type(rounds) is not int or not 0 <= rounds <= len(ROUNDS)
            or type(epochs) is not int or not 0 <= epochs <= rounds * len(bit.SLOTS) * bit.EPOCHS):
        raise ValueError('invalid completed cycle counts')
    return Counts(real_blocks=bit.KAT_BLOCKS + rounds * bit.REPEATS * (
        bit.TRAIN + bit.DEV + bit.TEST), null_words=0,
        optimizer_updates=epochs * (bit.TRAIN // bit.BATCH),
        forward_rows=epochs * (bit.TRAIN + bit.DEV) + rounds * len(bit.SLOTS) * bit.TEST)


CAPS = counts(len(ROUNDS), bit.EPOCHS * len(SLOTS))


def recipe() -> dict[str, object]:
    return {**bit.recipe(), 'method': METHOD, 'rounds': list(ROUNDS), 'slots': list(SLOTS),
            'fits': len(SLOTS), 'params': {str(r): parameters(r) for r in ROUNDS},
            'node_counts': [20, 34, 50, 66, 82], 'support_count': [24, 30, 32, 32, 32],
            'dense_layers': [[32, 128, n, 1] for n in (187, 244, 302, 360, 417)],
            'shuffled_leaf_sha256': None,
            'shuffle': 'within-word-leaf-rotation/round-specific-sha256-order',
            'plaintext_sampling': ('unique-uint32/paired-train-dev-across-rounds/'
                                   'fresh-test-per-round'),
            'key_policy': 'two-distinct-synthetic-keys/reused-across-rounds/data-only',
            'test_admission': 'per-round-after-all-six-strict-disk-restores',
            'stop': {'accuracy_min': ACC_MIN, 'accuracy_max': ACC_MAX,
                     'max_constant_bce_improvement': BCE_GAIN_MAX, 'requires_both_panels': True},
            'wall_seconds': WALL_SECONDS, 'cpu_seconds': CPU_SECONDS,
            'output_bytes': OUTPUT_BYTES, 'count_caps': dict(CAPS)}

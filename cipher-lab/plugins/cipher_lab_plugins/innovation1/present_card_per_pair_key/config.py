"""Fixed matched-key recipe and completed-work accounting."""
from __future__ import annotations

from ..prefix_bridge.config import Counts
from ..present_card_fresh import config as pure
from ..present_card_swap import config as swap

FIXED = 'fixed-key'
PER_PAIR = 'per-pair-key'
SLOT = FIXED
SLOTS = (FIXED, PER_PAIR)
METHOD = 'innovation1-present-card-per-pair-key-v1'
CONTRAST = f'{PER_PAIR}_vs_{FIXED}'
TRAIN_DEV_HASHES = frozenset({'train-fixed-key', 'train-per-pair-key', 'dev'})
TRAIN_ROWS, DEV_ROWS, TEST_ROWS = pure.TRAIN_ROWS, pure.DEV_ROWS, pure.TEST_ROWS
DATA_SEED, DIFFERENCE, ROUNDS = pure.DATA_SEED, pure.DIFFERENCE, pure.ROUNDS
SWAP_SEED = swap.SWAP_SEED
KAT_NAME = 'public-present-kat-and-paired-key-reference'
KAT_BLOCKS = pure.KAT_BLOCKS + 8
PARAMETERS = dict.fromkeys(SLOTS, pure.PARAMETERS)
OBJECTIVES = dict.fromkeys(SLOTS, 'classification-bce')
MASK_SITES = MAX_AUXILIARY_TARGETS = 0


def counts(epochs: tuple[int, ...], *, test: bool = True, diagnostic: bool = False) -> Counts:
    if (diagnostic or len(epochs) > len(SLOTS) or (test and len(epochs) != len(SLOTS))
            or any(type(n) is not int or not pure.MIN_EPOCHS <= n <= pure.EPOCHS for n in epochs)):
        raise ValueError('INVALID_COMPLETED_EPOCHS')
    test_rows = TEST_ROWS if test else 0
    return Counts(real_blocks=KAT_BLOCKS + 2 * (2 * TRAIN_ROWS + DEV_ROWS + test_rows),
                  null_words=0, optimizer_updates=sum(pure._steps(n) for n in epochs),
                  forward_rows=(TRAIN_ROWS + DEV_ROWS) * sum(epochs) + len(SLOTS) * test_rows)


CAPS = counts((pure.EPOCHS,) * len(SLOTS))


def recipe() -> dict[str, object]:
    base = swap.recipe()
    base.pop('dev_diagnostics')
    return {**base, 'method': METHOD, 'slots': list(SLOTS), 'parameters': dict(PARAMETERS),
            'objectives': dict(OBJECTIVES), 'shared_panel': False,
            'shared_train_plaintext_and_labels': True, 'shared_dev_test': True,
            'train_key_policy': dict(zip(SLOTS, ('one-synthetic-key', 'one-key-per-pair'),
                                         strict=True)),
            'dev_test_key_policy': 'one-synthetic-key-per-pair',
            'secret_key_draws': 1 + TRAIN_ROWS + DEV_ROWS + TEST_ROWS,
            'train_ciphertexts': 4 * TRAIN_ROWS,
            'cross_split_keys': 'exact-disjoint/no-retry',
            'train_swap_probability': dict.fromkeys(SLOTS, .5),
            'test_admission': 'once-after-both-strict-disk-restores-and-activation',
            'primary_contrast': [PER_PAIR, FIXED], 'count_caps': dict(CAPS),
            'gate': KAT_NAME, 'kat_real_blocks': KAT_BLOCKS,
            'trajectories': CAPS['real_blocks'] - KAT_BLOCKS}

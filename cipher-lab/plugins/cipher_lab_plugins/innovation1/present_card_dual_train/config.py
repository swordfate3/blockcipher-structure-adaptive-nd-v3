"""Fixed dual-order training comparison and completed-work accounting."""
from __future__ import annotations

from ..prefix_bridge.config import Counts
from ..present_card_fresh import config as pure
from ..present_card_swap import config as swap

METHOD = 'innovation1-present-card-dual-train-v1'
SLOT = 'swap-50'
SLOTS = (SLOT, 'dual-train')
SWAP_SEED = swap.SWAP_SEED
TRAIN_ROWS, DEV_ROWS, TEST_ROWS = pure.TRAIN_ROWS, pure.DEV_ROWS, pure.TEST_ROWS
DATA_SEED, DIFFERENCE, ROUNDS = pure.DATA_SEED, pure.DIFFERENCE, pure.ROUNDS
PARAMETERS = dict.fromkeys(SLOTS, pure.PARAMETERS)
OBJECTIVES = {SLOT: 'classification-bce', SLOTS[1]: 'arithmetic-mean-probability-bce'}
CONTRAST = f'{SLOTS[1]}_vs_{SLOT}'
MASK_SITES = MAX_AUXILIARY_TARGETS = 0


def counts(epochs: tuple[int, ...], *, test: bool = True, diagnostic: bool = False) -> Counts:
    if (len(epochs) > len(SLOTS) or ((test or diagnostic) and len(epochs) != len(SLOTS))
            or any(type(n) is not int or not pure.MIN_EPOCHS <= n <= pure.EPOCHS for n in epochs)):
        raise ValueError('INVALID_COMPLETED_EPOCHS')
    test_rows = TEST_ROWS if test else 0
    diagnostic_rows = 4 * DEV_ROWS if diagnostic or test else 0
    extra_train = epochs[1] * TRAIN_ROWS if len(epochs) == len(SLOTS) else 0
    return Counts(real_blocks=pure.KAT_BLOCKS + 2 * (TRAIN_ROWS + DEV_ROWS + test_rows),
                  null_words=0, optimizer_updates=sum(pure._steps(n) for n in epochs),
                  forward_rows=(TRAIN_ROWS + 2 * DEV_ROWS) * sum(epochs) + extra_train
                  + diagnostic_rows + 4 * test_rows)


CAPS = counts((pure.EPOCHS,) * len(SLOTS))


def recipe() -> dict[str, object]:
    return {**swap.recipe(), 'method': METHOD, 'slots': list(SLOTS),
            'parameters': dict(PARAMETERS), 'objectives': dict(OBJECTIVES),
            'probability_aggregation': 'FP64-log-space/FP32-equivalent-logit',
            'train_swap_probability': {SLOT: .5, SLOTS[1]: None},
            'selection': 'dual-order-mean-probability-DEV-class-BCE/earliest-min',
            'test_order': 'dual-order-mean-probability-primary/original-secondary',
            'primary_contrast': [SLOTS[1], SLOT], 'count_caps': dict(CAPS)}

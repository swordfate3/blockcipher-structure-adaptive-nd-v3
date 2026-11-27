"""Fixed paired swap and mask recipe and completed-work accounting."""
from __future__ import annotations

from ..prefix_bridge.config import Counts
from ..present_card_fresh import config as pure
from ..present_card_mask import config as masked
from ..present_card_swap import config as swap

METHOD = 'innovation1-present-card-swap-mask-v1'
SLOT = 'swap-50'
SLOTS = (SLOT, 'swap-50-mask')
SWAP_SEED = 20261128
TRAIN_ROWS, DEV_ROWS, TEST_ROWS = pure.TRAIN_ROWS, pure.DEV_ROWS, pure.TEST_ROWS
DATA_SEED, DIFFERENCE, ROUNDS = pure.DATA_SEED, pure.DIFFERENCE, pure.ROUNDS
MASK_SEED, MASK_SITES, AUX_WEIGHT = masked.MASK_SEED, masked.MASK_SITES, masked.AUX_WEIGHT
PARAMETERS = {SLOT: pure.PARAMETERS, SLOTS[1]: masked.PARAMETERS}
OBJECTIVES = {SLOT: 'classification-bce', SLOTS[1]: 'classification-bce+0.1*masked-bce'}
CONTRAST = f'{SLOTS[1]}_vs_{SLOT}'


def counts(epochs: tuple[int, ...], *, test: bool = True, diagnostic: bool = False) -> Counts:
    if (len(epochs) > len(SLOTS) or ((test or diagnostic) and len(epochs) != len(SLOTS))
            or any(type(n) is not int or not pure.MIN_EPOCHS <= n <= pure.EPOCHS for n in epochs)):
        raise ValueError('INVALID_COMPLETED_EPOCHS')
    test_rows = TEST_ROWS if test else 0
    diagnostic_rows = 2 * DEV_ROWS * len(SLOTS) if diagnostic or test else 0
    masked_rows = epochs[1] * TRAIN_ROWS if len(epochs) == len(SLOTS) else 0
    return Counts(real_blocks=pure.KAT_BLOCKS + 2 * (TRAIN_ROWS + DEV_ROWS + test_rows),
                  null_words=0, optimizer_updates=sum(pure._steps(n) for n in epochs),
                  forward_rows=(TRAIN_ROWS + DEV_ROWS) * sum(epochs) + masked_rows
                  + diagnostic_rows + len(SLOTS) * test_rows)


CAPS = counts((pure.EPOCHS,) * len(SLOTS))
MAX_AUXILIARY_TARGETS = pure.EPOCHS * TRAIN_ROWS * MASK_SITES


def recipe() -> dict[str, object]:
    auxiliary = masked.recipe()
    return {**swap.recipe(), **{key: auxiliary[key] for key in (
                'mask_seed', 'mask_sites_per_pair', 'aux_weight', 'mask_schedule',
                'masked_view', 'aux_head', 'flag_projection',
            )}, 'method': METHOD, 'slots': list(SLOTS), 'parameters': dict(PARAMETERS),
            'auxiliary': True, 'auxiliary_by_slot': {SLOT: False, SLOTS[1]: True},
            'objectives': dict(OBJECTIVES), 'train_swap_probability': dict.fromkeys(SLOTS, .5),
            'swap_seed': SWAP_SEED, 'shared_initialization': 'classifier-and-geometry',
            'primary_contrast': [SLOTS[1], SLOT], 'count_caps': dict(CAPS),
            'max_auxiliary_targets': MAX_AUXILIARY_TARGETS}

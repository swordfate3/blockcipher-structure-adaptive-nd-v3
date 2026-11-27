"""Bound the shared-panel position experiment."""
from __future__ import annotations

from ..prefix_bridge.config import Counts
from ..present_card_fresh import config as pure
from ..present_card_mask import config as masked

METHOD = 'innovation1-present-card-mask-positions-v1'
RANDOM, FIRST, FOURTH = 'random-16', 'first-of-four', 'fourth-of-four'
SLOTS = (RANDOM, FIRST, FOURTH)
SLOT = RANDOM  # The panel provider receives one shared panel identifier.
TRAIN_ROWS, DEV_ROWS, TEST_ROWS = pure.TRAIN_ROWS, pure.DEV_ROWS, pure.TEST_ROWS
DATA_SEED, DIFFERENCE, ROUNDS = pure.DATA_SEED, pure.DIFFERENCE, pure.ROUNDS


def counts(epochs: tuple[int, ...], *, test: bool = True) -> Counts:
    if (len(epochs) > len(SLOTS) or (test and len(epochs) != len(SLOTS))
            or any(type(n) is not int or not pure.MIN_EPOCHS <= n <= pure.EPOCHS for n in epochs)):
        raise ValueError('INVALID_COMPLETED_EPOCHS')
    test_rows = TEST_ROWS if test else 0
    evaluation_rows = len(SLOTS) * test_rows
    return {'real_blocks': pure.KAT_BLOCKS + 2 * (TRAIN_ROWS + DEV_ROWS + test_rows),
            'null_words': 0, 'optimizer_updates': sum(pure._steps(n) for n in epochs),
            'forward_rows': (2 * TRAIN_ROWS + DEV_ROWS) * sum(epochs) + evaluation_rows}


CAPS = counts((pure.EPOCHS,) * len(SLOTS))


def recipe() -> dict[str, object]:
    base = masked.recipe()
    return {**base, 'method': METHOD, 'fits': len(SLOTS), 'slots': list(SLOTS),
            'count_caps': dict(CAPS), 'positions_msb0': {
                RANDOM: 'uniform-without-replacement', FIRST: list(range(0, 64, 4)),
                FOURTH: list(range(3, 64, 4))}, 'shared_panel': True,
            'test_admission': 'single-panel-after-all-three-disk-restores',
            'primary_contrast': [FOURTH, FIRST],
            'max_auxiliary_targets': pure.EPOCHS * TRAIN_ROWS * masked.MASK_SITES * len(SLOTS)}

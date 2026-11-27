"""Fixed stdlib-only recipe and completed-work accounting."""
from __future__ import annotations

from ..prefix_bridge.config import Counts
from ..present_card_fresh import config as pure

METHOD = 'innovation1-present-card-swap-v1'
SLOT = 'original'
SLOTS = (SLOT, 'swap-50')
SWAP_SEED = 20261128
TRAIN_ROWS, DEV_ROWS, TEST_ROWS = pure.TRAIN_ROWS, pure.DEV_ROWS, pure.TEST_ROWS
DATA_SEED, DIFFERENCE, ROUNDS = pure.DATA_SEED, pure.DIFFERENCE, pure.ROUNDS


def counts(epochs: tuple[int, ...], *, test: bool = True, diagnostic: bool = False) -> Counts:
    if (len(epochs) > len(SLOTS) or ((test or diagnostic) and len(epochs) != len(SLOTS))
            or any(type(n) is not int or not pure.MIN_EPOCHS <= n <= pure.EPOCHS for n in epochs)):
        raise ValueError('INVALID_COMPLETED_EPOCHS')
    test_rows = TEST_ROWS if test else 0
    diagnostic_rows = 2 * DEV_ROWS * len(SLOTS) if diagnostic or test else 0
    return Counts(real_blocks=pure.KAT_BLOCKS + 2 * (TRAIN_ROWS + DEV_ROWS + test_rows),
                  null_words=0, optimizer_updates=sum(pure._steps(n) for n in epochs),
                  forward_rows=(TRAIN_ROWS + DEV_ROWS) * sum(epochs) + diagnostic_rows
                  + len(SLOTS) * test_rows)


CAPS = counts((pure.EPOCHS,) * len(SLOTS))
CONTRAST = f'{SLOTS[1]}_vs_{SLOT}'


def recipe() -> dict[str, object]:
    return {**pure.recipe(), 'method': METHOD, 'slots': list(SLOTS), 'fits': len(SLOTS),
            'shared_panel': True, 'shared_initialization': True, 'shared_training_order': True,
            'schedule': 'per-arm-zero-based-fixed-cosine', 'swap_seed': SWAP_SEED,
            'train_swap_probability': dict(zip(SLOTS, (0., .5), strict=True)),
            'swap_schedule': 'PCG64(SeedSequence([swap_seed,zero-based-update]))/per-row',
            'selection': 'original-order-pair-DEV-class-BCE/earliest-min',
            'dev_diagnostics': 'both-restored-bests/original-and-swapped/unchanged-labels',
            'test_admission': 'single-panel-after-both-disk-restores-and-dev-diagnostics',
            'test_order': 'original-for-both-arms', 'primary_contrast': [SLOTS[1], SLOT],
            'count_caps': dict(CAPS)}

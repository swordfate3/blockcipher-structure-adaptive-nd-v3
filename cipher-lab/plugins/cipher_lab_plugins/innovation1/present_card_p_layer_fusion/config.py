"""Fixed three-arm fusion recipe and completed-work accounting."""
from __future__ import annotations

from ..prefix_bridge.config import Counts
from ..present_card_fresh import config as pure
from ..present_card_swap.config import SWAP_SEED
from . import wiring

METHOD = 'innovation1-present-card-p-layer-fusion-v1'
BASE, SHUFFLED, P_LAYER = wiring.BASE, wiring.SHUFFLED, wiring.P_LAYER
SLOTS = (BASE, SHUFFLED, P_LAYER)
SLOT = BASE
TRAIN_ROWS, DEV_ROWS, TEST_ROWS = pure.TRAIN_ROWS, pure.DEV_ROWS, pure.TEST_ROWS
DATA_SEED, DIFFERENCE, ROUNDS = pure.DATA_SEED, pure.DIFFERENCE, pure.ROUNDS
ADDED_PARAMETERS = {BASE: 0, SHUFFLED: 9216, P_LAYER: 9216}
PARAMETERS = {slot: pure.PARAMETERS + added for slot, added in ADDED_PARAMETERS.items()}
CONTRASTS = ((P_LAYER, SHUFFLED), (SHUFFLED, BASE), (P_LAYER, BASE))


def counts(epochs: tuple[int, ...], *, test: bool = True) -> Counts:
    if (len(epochs) > len(SLOTS) or (test and len(epochs) != len(SLOTS))
            or any(type(n) is not int or not pure.MIN_EPOCHS <= n <= pure.EPOCHS for n in epochs)):
        raise ValueError('INVALID_COMPLETED_EPOCHS')
    test_rows = TEST_ROWS if test else 0
    return Counts(real_blocks=pure.KAT_BLOCKS + 2 * (TRAIN_ROWS + DEV_ROWS + test_rows),
                  null_words=0, optimizer_updates=sum(pure._steps(n) for n in epochs),
                  forward_rows=(TRAIN_ROWS + DEV_ROWS) * sum(epochs) + len(SLOTS) * test_rows)


CAPS = counts((pure.EPOCHS,) * len(SLOTS))


def recipe() -> dict[str, object]:
    return {**pure.recipe(), 'method': METHOD, 'slots': list(SLOTS), 'fits': len(SLOTS),
            'shared_panel': True, 'shared_backbone_initialization': True,
            'shared_training_order': True, 'parameters': dict(PARAMETERS),
            'added_parameters': dict(ADDED_PARAMETERS),
            'wiring_sha256': {slot: wiring.wiring_sha256(slot) for slot in SLOTS},
            'fusion': 'weighted-merge/pre-backbone/residual/shared-96x96/zero-init/no-bias',
            'schedule': 'per-arm-zero-based-fixed-cosine', 'swap_seed': SWAP_SEED,
            'train_swap_probability': dict.fromkeys(SLOTS, .5),
            'swap_schedule': 'PCG64(SeedSequence([swap_seed,zero-based-update]))/per-row',
            'selection': 'original-order-pair-DEV-class-BCE/earliest-min',
            'test_admission': 'single-panel-after-all-three-disk-restores',
            'test_order': 'original-for-all-arms', 'primary_contrast': list(CONTRASTS[0]),
            'count_caps': dict(CAPS)}

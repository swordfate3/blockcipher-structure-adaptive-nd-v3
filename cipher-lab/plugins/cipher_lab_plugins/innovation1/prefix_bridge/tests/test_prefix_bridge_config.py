from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest


def test_recipe_and_cap_arithmetic() -> None:
    from cipher_lab_plugins.innovation1.prefix_bridge.config import CAPS, SLOTS, counts, recipe

    assert SLOTS == ('s2-only', 'r2-cipher', 'r2-joint', 'r3-cipher', 'r3-joint',
                     'r4-cipher', 'r4-joint', 'r5-cipher', 'r5-joint', 'r6-cipher', 'r6-joint',
                     'r7-cipher', 'r7-joint')
    assert len(set(SLOTS)) == 13  # runtime/checkpoint slot keys, not model instantiation
    assert CAPS == {'real_blocks': 3538958, 'null_words': 0,
                    'optimizer_updates': 99840, 'forward_rows': 16400384}
    epochs = (18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)
    assert counts(epochs) == {'real_blocks': 3538958, 'null_words': 0,
                             'optimizer_updates': 79872, 'forward_rows': 13205504}
    public = recipe()
    expected = {
        'cipher': 'PRESENT-80', 'rounds': [2, 3, 4, 5, 6, 7], 'slots': list(SLOTS), 'fits': 13,
        'pairs_per_group': 4, 'train_rows': 32768, 'dev_rows': 8192, 'test_rows': 32768,
        'difference': 0xD00000, 'secret_key_draw_bits': 80, 'secret_key_draws': 1,
        'batch_size': 128, 'max_epochs': 30, 'min_epochs': 18,
        'data_seed': 20261125, 'order_seed': 20261126, 'init_seed': 31,
        'stem_inputs': 54, 'channels': 96, 'norm_groups': 8, 'public_features': 13,
        'router_top_k': 2, 'suffix_kernels': [3, 5, 7, 9, 11],
        'label_columns': 1, 'selection': 'group-DEV-BCE/earliest-min',
        'early_stop': {'warmup_epochs': 10, 'patience': 8, 'min_delta': 1e-5},
        'optimizer': 'Adam', 'lr': .001, 'betas': [.9, .999], 'eps': 1e-8,
        'clip': 1., 'weight_decay': 0., 'dtype': 'FP32', 'amp': False, 'tf32': False,
        'kat_real_blocks': 14, 'roundstateoutputs': 3538944, 'trajectories': 589824,
        'count_caps': CAPS, 'wall_seconds': 14400, 'cpu_seconds': 57600, 'rss_bytes': 8 * 2**30,
        'output_bytes': 512 * 2**20, 'checkpoint_bytes': 16 * 2**20,
        'gpu_uuid': 'GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5', 'gpu_device': 'cuda:0',
        'primary': 'correct-groups/TEST-groups', 'retry_count': 0,
    }
    assert {k: public[k] for k in expected} == expected
    assert recipe() == public
    public['count_caps'] = {}
    assert recipe()['count_caps'] == CAPS


@pytest.mark.parametrize('epochs', [
    (), (18,) * 12, (18,) * 14, (17,) * 13, (31,) * 13,
    (True,) * 13, (18.,) * 13, ('18',) * 13,
])
def test_counts_require_exactly_13_completed_integer_epochs(epochs: Sequence[Any]) -> None:
    from cipher_lab_plugins.innovation1.prefix_bridge.config import counts

    with pytest.raises(ValueError, match='completed epochs'):
        counts(epochs)

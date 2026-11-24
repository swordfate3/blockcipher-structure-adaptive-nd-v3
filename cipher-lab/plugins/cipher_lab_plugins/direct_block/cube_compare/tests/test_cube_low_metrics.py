from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize('rounds', [1, 2, 3])
def test_cube_low_counts_copy_halves_and_whole_pair(rounds: int) -> None:
    from cipher_lab_plugins.direct_block.cube_compare.low_metrics import LowScore, train_majority

    train = np.zeros((2, 128), dtype=np.uint8)
    train[0, :32] = 1  # tie -> zero
    train[:, 32:64] = 1
    majority = train_majority((train[:1], train[1:]))
    assert majority == (0,) * 32 + (1,) * 32 + (0,) * 64
    labels = np.ones((8, 128), dtype=np.uint8)
    inputs = np.zeros_like(labels, dtype=np.float32)
    inputs[:, :32] = inputs[:, 64:96] = 1
    logits = np.full_like(inputs, -1)
    logits[:, 32:64] = logits[:, 96:] = 0  # zeros predict 1
    logits[0] = 0
    logits[1] = 0
    logits[1, 0] = -1   # one incorrect bit: entire pair fails
    score = LowScore(8, majority, rounds)
    for at in (slice(0, 3), slice(3, 8)):
        score.add(np.full(len(labels[at]), 7, dtype=np.uint64), inputs[at], labels[at], logits[at])
    result = score.result()
    assert result['denominators'] == dict(active8=64, full128=1024, x64=512, y64=512,
                                          block0=512, block1=512, wholepair=8)
    methods = result['scores']
    model = methods['lstm']
    assert model['correct'] == dict(active8=40, full128=639, x64=127, y64=512,
                                    block0=319, block1=320, wholepair=1)
    assert model['correct_by_column'] == [1] + [2] * 31 + [8] * 32 + [2] * 32 + [8] * 32
    assert result['label_ones_by_column'] == [8] * 128
    assert result['label_zeros_by_column'] == [0] * 128
    assert result['families'] == [{'family_id': 7, 'rows': 8,
                                   'correct': {name: value['correct']
                                               for name, value in methods.items()}}]
    assert methods['majority']['correct']['full128'] == 256
    if rounds == 1:
        assert methods['copy']['correct']['y64'] == 512
        assert methods['copy']['correct']['x64'] == methods['majority']['correct']['x64'] == 0
    else:
        assert 'copy' not in methods

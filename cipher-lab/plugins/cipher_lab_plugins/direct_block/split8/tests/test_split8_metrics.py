from __future__ import annotations

import numpy as np
import pytest

from cipher_lab.execution import Meter


def test_split8_two_bit_concat_success_not_whole_word(split8_meter: Meter) -> None:
    from cipher_lab_plugins.direct_block.split8 import metrics

    majority = metrics.train_majority(np.array([0x80, 0xc0, 0x00, 0x80], np.uint8), 2,
                                        split8_meter)
    assert majority == (1, 0, 0, 0, 0, 0, 0, 0)
    test = np.array([[1, 1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1, 0, 1]], np.uint8)
    pred = test.copy()
    pred[1, 7] = 0
    blocks = [2 * pred[:, j:j+2].astype(np.float32) - 1 for j in range(0, 8, 2)]
    score = metrics.Score(2, majority)
    with pytest.raises(ValueError):
        score.add(test, blocks[:-1])
    with pytest.raises(ValueError):
        score.add(test, [*blocks[:-1], blocks[-1][:, :1]])
    with pytest.raises(ValueError):
        score.result()
    score.add(test, blocks)
    result = score.result()
    assert result['bit_correct'] == 15 and result['bit_denominator'] == 16
    assert result['bit_accuracy'] == 15 / 16
    assert result['eight_all_correct'] == 1 and result['rows'] == 2
    assert result['correct_by_column'] == [2] * 7 + [1]
    assert result['label_ones_by_column'] == [1, 2, 0, 1, 1, 1, 1, 1]
    assert result['label_zeros_by_column'] == [1, 0, 2, 1, 1, 1, 1, 1]
    assert result['majority_bit_correct'] == 8 and result['majority_bit_accuracy'] == 8 / 16
    assert result['majority_bit_denominator'] == 16
    assert [n['bit_correct'] for n in result['networks']] == [4, 4, 4, 3]

from __future__ import annotations

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.cube_compare import config


def test_direct64_absolute_full_half_whole_column_and_r1_copy_metrics() -> None:
    from cipher_lab_plugins.direct_block.cube_compare.direct_metrics import (
        DirectScore,
        train_majority,
    )

    x = np.zeros((2, 64), np.float32)
    x[:, 0] = 1
    y = np.zeros((2, 64), np.uint8)
    y[:, 32] = 1
    y[1, 31] = 1
    assert not any(train_majority([np.zeros_like(y), np.ones_like(y)]))
    logits = 2 * y.astype(np.float32) - 1
    logits[1, 0] = 0  # >=0 is a predicted one; only the second word has an error
    for n in config.LOW_ROUNDS:
        metric = DirectScore(2, (0,) * 64, n)
        with pytest.raises(ValueError, match='incomplete'):
            metric.result()
        metric.add(x, y, logits)
        result = metric.result()
        assert result['denominators'] == {'full64': 128, 'x32': 64, 'y32': 64, 'wholeword64': 2}
        scores = result['scores']
        assert scores['lstm']['correct'] == {'full64': 127, 'x32': 63, 'y32': 64, 'wholeword64': 1}
        assert scores['lstm']['accuracy']['full64'] == 127 / 128
        assert scores['lstm']['correct_by_column'] == [1] + [2] * 63
        assert result['label_ones_by_column'] == [0] * 31 + [1, 2] + [0] * 31
        assert result['label_zeros_by_column'] == [2] * 31 + [1, 0] + [2] * 31
        assert scores['majority']['correct']['full64'] == 125
        if n == 1:
            assert scores['copy']['correct'] == scores['lstm']['correct']
        else:
            assert 'copy' not in scores
    assert config.direct_denominators(131072) == {
        'full64': 8388608, 'x32': 4194304, 'y32': 4194304, 'wholeword64': 131072,
    }

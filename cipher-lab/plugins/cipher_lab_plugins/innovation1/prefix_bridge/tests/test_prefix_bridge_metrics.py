from __future__ import annotations

import numpy as np
import pytest
from cipher_lab_plugins.innovation1.prefix_bridge.metrics import compare, score


def test_group_scores_and_paired_counts() -> None:
    y = np.array([[1], [0], [0], [1]], dtype=np.uint8)
    a = score(np.array([[1], [1], [-2], [-3]], np.float32), y)
    b = score(np.array([[1], [1], [0], [3]], np.float32), y)
    assert a.saved['correct_groups'] == 2 and a.saved['groups'] == 4
    assert a.saved['accuracy'] == .5
    assert a.saved['confusion'] == {'tn': 1, 'fp': 1, 'fn': 1, 'tp': 1}
    assert a.saved['bce'] == pytest.approx(np.logaddexp(0, [-1, 1, -2, 3]).mean())
    assert compare(a, b) == {
        'percentage_points': 0., 'both_correct': 1, 'joint_only_correct': 1,
        'control_only_correct': 1, 'both_wrong': 1,
    }
    with pytest.raises(ValueError):
        score(np.full((4, 1), np.inf, np.float32), y)

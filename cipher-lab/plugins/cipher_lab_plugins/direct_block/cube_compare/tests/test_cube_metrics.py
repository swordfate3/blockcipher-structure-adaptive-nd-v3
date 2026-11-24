from __future__ import annotations

import numpy as np
import pytest


def test_cube_train_only_majority_and_paired_family_covariance() -> None:
    from cipher_lab_plugins.direct_block.cube_compare.config import ACTIVE_COLUMNS
    from cipher_lab_plugins.direct_block.cube_compare.metrics import (
        PairedScore,
        paired_interval,
        train_majority,
    )

    train = np.zeros((4, 128), np.uint8)
    train[:3, ACTIVE_COLUMNS[0]] = 1
    train[:2, ACTIVE_COLUMNS[1]] = 1
    frozen = train_majority([train[:1], train[1:]])
    assert frozen == (1, 0, 0, 0, 0, 0, 0, 0)
    family_ids = np.repeat(np.array([11, 7, 23], np.uint64), 8)
    labels = np.zeros((24, 128), np.uint8)
    labels[:8, ACTIVE_COLUMNS] = 1
    mlp = np.full((24, 128), -1., np.float32)
    shared = mlp.copy()
    # family 11: left +64; family 7: left -32; family 23: 0.
    shared[:8, ACTIVE_COLUMNS] = 1.
    shared[8:16, ACTIVE_COLUMNS[:4]] = 1.
    score = PairedScore(24, frozen)
    for at in (slice(0, 10), slice(10, 19), slice(19, None)):
        score.add(family_ids[at], labels[at], mlp[at], shared[at])
    result = score.result()
    assert result['denominator_bits'] == 192
    assert result['label_ones_by_bit'] == [8] * 8
    assert result['label_zeros_by_bit'] == [16] * 8
    assert result['label_ones'] == 64 and result['label_zeros'] == 128
    assert result['correct_by_bit'] == {'majority': [8] + [16] * 7, 'mlp': [16] * 8,
                                       'sharedm': [16] * 4 + [24] * 4}
    assert result['families'] == [
        {'family_id': 7, 'majority': 56, 'mlp': 64, 'sharedm': 32},
        {'family_id': 11, 'majority': 8, 'mlp': 0, 'sharedm': 64},
        {'family_id': 23, 'majority': 56, 'mlp': 64, 'sharedm': 64},
    ]
    ci = paired_interval([32, 64, 64], [64, 0, 64])
    diffs = np.asarray([-.5, 1., 0.])
    delta, se = float(diffs.mean()), float(diffs.std(ddof=1) / np.sqrt(3))
    assert ci == pytest.approx({'delta': delta, 'se': se, 'lower': delta - 2.241402727604947 * se,
                               'upper': delta + 2.241402727604947 * se})
    assert paired_interval([16, 40], [8, 32])['se'] == 0  # paired covariance, not binomial
    result_ci = result['comparisons']
    assert isinstance(result_ci, dict)
    assert result_ci['sharedm_minus_mlp'] == ci
    assert result_ci['sharedm_minus_majority'] == paired_interval([32, 64, 64], [56, 8, 56])
    assert result_ci['mlp_minus_majority'] == pytest.approx({'delta': 8 / 192})


def test_cube_metrics_reject_incomplete_or_nonfinite_scores() -> None:
    from cipher_lab_plugins.direct_block.cube_compare.metrics import PairedScore, paired_interval

    score = PairedScore(16, (0,) * 8)
    z = np.zeros((8, 128), np.float32)
    labels = z.astype(np.uint8)
    families = np.zeros(8, dtype=np.uint64)
    with pytest.raises(ValueError, match='incomplete'):
        score.result()
    broken = z.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match='logits'):
        score.add(families, labels, broken, z)
    score.add(families, labels, z, z)
    with pytest.raises(ValueError, match='family'):
        score.add(families, labels, z, z)
    with pytest.raises(ValueError, match='famil'):
        paired_interval([0], [0])


def test_cube_lstm_complete_family_stats_and_no_old_pairing() -> None:
    from cipher_lab_plugins.direct_block.cube_compare.config import ACTIVE_COLUMNS
    from cipher_lab_plugins.direct_block.cube_compare.metrics import LSTMScore, paired_interval

    ids = np.repeat(np.array([11, 7, 23], np.uint64), 8)
    y = np.zeros((24, 128), np.uint8)
    y[:8, ACTIVE_COLUMNS] = 1
    z = np.full(y.shape, -1, np.float32)
    z[:8, ACTIVE_COLUMNS] = 0  # ties in logits predict one
    z[8:16, ACTIVE_COLUMNS[:4]] = 1
    score = LSTMScore(24, (1, 0, 0, 0, 0, 0, 0, 0))
    score.add(ids[:10], y[:10], z[:10])
    with pytest.raises(ValueError, match='incomplete'):
        score.result()
    # Invalid partial additions must not contaminate the complete result.
    bad = z[10:].copy()
    bad[-1, 0] = np.nan
    with pytest.raises(ValueError, match='logits'):
        score.add(ids[10:], y[10:], bad)
    with pytest.raises(ValueError, match='family'):
        score.add(np.zeros(14, np.uint64), y[10:], z[10:])
    score.add(ids[10:], y[10:], z[10:])
    result = score.result()
    assert result['confidence'] == .975 and 'familywise_alpha' not in result
    assert result['label_ones_by_bit'] == [8] * 8 and result['denominator_bits'] == 192
    assert result['correct_by_bit'] == {'majority': [8] + [16] * 7,
                                       'lstm': [16] * 4 + [24] * 4}
    assert result['families'] == [{'family_id': i, 'majority': m, 'lstm': s}
                                  for i, m, s in ((7, 56, 32), (11, 8, 64), (23, 56, 64))]
    assert result['comparisons'] == {
        'lstm_minus_majority': paired_interval([32, 64, 64], [56, 8, 56])}

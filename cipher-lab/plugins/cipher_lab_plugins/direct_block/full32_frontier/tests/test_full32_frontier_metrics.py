import math
from typing import cast

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.full32_frontier import data, metrics

KEYS = ('r1-correct', 'r1-shuffled', 'r2-correct', 'r2-shuffled')


def test_score_counts_bce_and_radius() -> None:
    labels = np.zeros((2, 32), dtype=np.uint8)
    logits = np.zeros((2, 32), dtype=np.float32)
    labels[:, 16:] = 1
    logits[0, 16] = 1
    result = metrics.score(logits, labels)
    assert result['rows'] == 2
    assert result['correct_bits'] == 33
    assert result['accuracy'] == 33 / 64
    assert result['high_correct'] == 32
    assert result['low_correct'] == 1
    assert result['high_accuracy'] == 1
    assert result['low_accuracy'] == 1 / 32
    assert result['per_bit_correct'] == [2] * 16 + [1] + [0] * 15
    assert result['bce'] == pytest.approx((63 * math.log(2) + math.log1p(math.exp(-1))) / 64)
    radius = math.sqrt(math.log(2 * 16 / .05) / (2 * 65536))
    assert result['ci_low'] == pytest.approx(33 / 64 - radius)
    assert result['ci_high'] == pytest.approx(33 / 64 + radius)
    assert metrics.score(logits, (logits > 0).astype(np.uint8))['ci_high'] == 1
    assert metrics.score(logits, (logits <= 0).astype(np.uint8))['ci_low'] == 0


def test_constant_uses_train_columns_and_ties_to_zero() -> None:
    train = np.zeros((2, 32), dtype=np.uint8)
    train[:, :16] = 1
    train[0, 16:] = 1
    test = np.ones((1, 32), dtype=np.uint8)
    result = metrics.constant_score(train, test)
    assert result['per_bit_correct'] == [1] * 16 + [0] * 16
    assert result['bce'] == pytest.approx(-(math.log(5 / 6) + math.log(.5)) / 2)
    assert result['accuracy'] == .5
    assert result.keys() == metrics.score(test.astype(np.float32), test).keys() - {
        'ci_low', 'ci_high'
    }


@pytest.mark.parametrize('logits,labels', [
    (np.zeros((1, 32)), np.zeros((1, 32), dtype=np.uint8)),
    (np.zeros((1, 32), dtype=np.float32), np.zeros((1, 32), dtype=np.int8)),
    (np.zeros((1, 32), dtype=np.float32), np.full((1, 32), 2, dtype=np.uint8)),
    (np.full((1, 32), np.inf, dtype=np.float32), np.zeros((1, 32), dtype=np.uint8)),
    (np.zeros((2, 32), dtype=np.float32), np.zeros((1, 32), dtype=np.uint8)),
    (np.zeros((0, 32), dtype=np.float32), np.zeros((0, 32), dtype=np.uint8)),
])
def test_invalid_score_arrays(logits: object, labels: object) -> None:
    with pytest.raises(ValueError):
        metrics.score(cast(data.Floats, logits), cast(data.Bytes, labels))


@pytest.mark.parametrize('key', KEYS)
def test_near_chance_requires_every_interval(key: str) -> None:
    scores = {name: {'ci_low': .49, 'ci_high': .51} for name in KEYS}
    assert metrics.classify(4, scores) == 'NEAR_CHANCE'
    assert metrics.classify(6, scores) == 'NEAR_CHANCE'
    for low, high in [(.489, .51), (.49, .511)]:
        scores[key] = {'ci_low': low, 'ci_high': high}
        assert metrics.classify(4, scores) == 'CONTINUE'
        assert metrics.classify(5, scores) == 'CONTINUE'
        assert metrics.classify(6, scores) == 'ROUND_CEILING'


@pytest.mark.parametrize('key', ('r1-correct', 'r2-correct'))
def test_positive_control_is_strict_and_uses_correct_wiring_only(key: str) -> None:
    scores: dict[str, dict[str, float]] = {
        name: {'ci_low': .512, 'ci_high': .55} if name.endswith('-correct')
        else {'ci_low': 0, 'ci_high': 1} for name in KEYS
    }
    assert metrics.classify(3, scores) == 'CONTINUE'
    scores[key]['ci_low'] = .51
    assert metrics.classify(3, scores) == 'POSITIVE_CONTROL_FAILED'


@pytest.mark.parametrize('interval', [None, {}, {'ci_low': .51, 'ci_high': .49},
                                     {'ci_low': float('nan'), 'ci_high': .51},
                                     {'ci_low': .49, 'ci_high': 1.01},
                                     {'ci_low': '0.49', 'ci_high': .51}])
def test_malformed_decisions_raise(interval: object) -> None:
    scores: dict[str, object] = {name: {'ci_low': .49, 'ci_high': .51} for name in KEYS}
    with pytest.raises(ValueError):
        metrics.classify(7, scores)
    scores.pop(KEYS[0])
    with pytest.raises(ValueError):
        metrics.classify(4, scores)
    scores[KEYS[0]] = interval
    with pytest.raises(ValueError):
        metrics.classify(4, scores)

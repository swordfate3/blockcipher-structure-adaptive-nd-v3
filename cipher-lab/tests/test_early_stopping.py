from __future__ import annotations

import math
from typing import cast

import pytest

from cipher_lab.training.early_stopping import EarlyStopping, Settings


def test_warmup_and_first_possible_stop() -> None:
    tracker = EarlyStopping(('a', 'b'), Settings(24, 12, 1e-5), epoch_ceiling=100)
    for epoch in range(1, 36):
        assert tracker.observe(epoch, {'b': .5, 'a': .25}) is None
    assert tracker.observe(36, {'a': .25, 'b': .5}) == 'EARLY_STOPPING'
    with pytest.raises(ValueError):
        tracker.observe(37, {'a': .1, 'b': .1})


def test_real_recipe_exact_target_at_first_possible_stop_epoch_counts() -> None:
    # Mirrors configs/innovation1-skinny-layout-early-stop.json: warmup 24, patience 12,
    # min_delta 1e-5, so 24 + 12 = 36 is the first epoch that may stop.
    settings = Settings(24, 12, 1e-5)
    tracker = EarlyStopping(('joint',), settings, epoch_ceiling=100)
    target = 0.7 - settings.min_delta
    # Why this is the regression case: the float target is hit exactly, while the old
    # `anchor - loss >= min_delta` form rounds the subtracted difference to
    # 9.99999999995449e-06 and reports stagnation.
    assert target == 0.69999
    assert 0.7 - target < settings.min_delta
    for epoch in range(1, 36):
        assert tracker.observe(epoch, {'joint': 0.7}) is None
    # A loss exactly on the float target is enough improvement, so no early stop.
    assert tracker.observe(36, {'joint': target}) is None


def test_real_recipe_one_ulp_above_target_is_not_enough_improvement() -> None:
    settings = Settings(24, 12, 1e-5)
    tracker = EarlyStopping(('joint',), settings, epoch_ceiling=100)
    target = 0.7 - settings.min_delta
    just_short = math.nextafter(target, math.inf)
    assert target < just_short < 0.7
    for epoch in range(1, 36):
        assert tracker.observe(epoch, {'joint': 0.7}) is None
    # No fuzzy tolerance: one ULP above the target is the twelfth stagnation.
    assert tracker.observe(36, {'joint': just_short}) == 'EARLY_STOPPING'


def test_warmup_best_cumulative_gains_and_one_arm_extends_pair() -> None:
    tracker = EarlyStopping((0, 1), Settings(2, 2, .125), epoch_ceiling=10)
    for epoch, left, right in [(1, .5, .5), (2, .75, .75), (3, .4375, .5),
                               (4, .375, .5), (5, .375, .5)]:
        assert tracker.observe(epoch, {0: left, 1: right}) is None
    # The exact-threshold cumulative improvement at 4 resets only arm 0.
    assert tracker.observe(6, {0: .375, 1: .5}) == 'EARLY_STOPPING'


def test_ceiling_has_precedence_over_joint_stop() -> None:
    tracker = EarlyStopping(('a',), Settings(1, 1, 0.), epoch_ceiling=2)
    assert tracker.observe(1, {'a': .2}) is None
    assert tracker.observe(2, {'a': .2}) == 'EPOCH_CEILING'


def test_invalid_observation_is_atomic() -> None:
    tracker = EarlyStopping(('a', 'b'), Settings(1, 1, .125), epoch_ceiling=4)
    assert tracker.observe(1, {'a': .5, 'b': .5}) is None
    for epoch in (0, 1, 3, True, 2.0):
        with pytest.raises(ValueError):
            tracker.observe(cast(int, epoch), {'a': .125, 'b': .125})
    for scores in ({'a': .125}, {'a': .125, 'c': .125}, {'a': .125, 'b': .125, 'c': .125}):
        with pytest.raises(ValueError):
            tracker.observe(2, scores)
    for value in (True, '1', float('nan'), float('inf'), -float('inf'), -.1, 10**1000):
        with pytest.raises(ValueError):
            tracker.observe(2, {'a': .125, 'b': cast(float, value)})
    # Even the valid first arm in an invalid observation must not reset its anchor.
    assert tracker.observe(2, {'a': .375, 'b': .5}) is None
    assert tracker.observe(3, {'a': .375, 'b': .5}) == 'EARLY_STOPPING'


@pytest.mark.parametrize('args', [(0, 2, .1), (True, 2, .1), (1, 0, .1), (1, 1., .1),
                                  (1, 2, True), (1, 2, float('nan')), (1, 2, -.1)])
def test_invalid_settings(args: tuple[int, int, float]) -> None:
    with pytest.raises(ValueError):
        Settings(*args)


def test_requires_unique_nonempty_versions_and_valid_ceiling() -> None:
    settings = Settings(2, 1, .1)
    for versions in ((), ('a', 'a')):
        with pytest.raises(ValueError):
            EarlyStopping(versions, settings, epoch_ceiling=3)
    with pytest.raises(ValueError):
        EarlyStopping(('a',), settings, epoch_ceiling=1)

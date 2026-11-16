import math

import numpy as np
import pytest
from cipher_lab_plugins.differential_relations.metrics import (
    accuracy_comparisons,
    fit_baselines,
    logloss,
    mcnemar_exact,
    paired_bce_comparison,
    relation_metrics,
)


def test_stable_logloss_and_missing_classes() -> None:
    labels = np.zeros((4, 4), dtype=np.uint8)
    logits = np.full((4, 4), -1000.)
    assert np.isfinite(logloss(logits, labels)).all()
    assert logloss(-logits, labels)[0, 0] == 1000.
    m = relation_metrics(logits, labels)[0]
    assert m['accuracy'] == 1 and m['recall_0'] == 1
    assert m['recall_1'] is None and m['balanced_accuracy'] is None and m['mcc'] is None
    assert m['undefined_reasons']
    with pytest.raises(ValueError):
        logloss(logits * np.nan, labels)


def test_train_only_laplace_stump_tie() -> None:
    orbits = np.array([0, 1, 2, 3], dtype=np.uint64)
    labels = np.tile(np.array([0, 1, 0, 1], dtype=np.uint8)[:, None], (1, 4))
    base = fit_baselines(orbits, labels)
    assert base.stump_bits.tolist() == [0] * 4
    np.testing.assert_array_equal(base.constant, [.5] * 4)
    np.testing.assert_allclose(base.stump_probabilities, [[.25, .75]] * 4)
    cp, sp = base.probabilities(orbits)
    assert cp.shape == sp.shape == labels.shape
    assert sp[0, 0] == .25 and sp[1, 0] == .75
    const = fit_baselines(orbits, np.zeros((4, 4), dtype=np.uint8))
    # Empty cells still get Laplace probability 1/2; identical predictors tie at bit2.
    assert const.stump_bits.tolist() == [2] * 4
    assert np.all(const.constant == 1 / 6)


def test_paired_unit_ci_and_warnings() -> None:
    n = 1000
    model = np.full((n, 4), .4)
    differences = np.linspace(.05, .15, n)
    baseline = model + differences[:, None]
    result = paired_bce_comparison(model, baseline, baseline)
    c = result['comparisons']['constant']
    assert c['mean_difference'] == pytest.approx(.1)
    sd = np.std(differences, ddof=1)
    assert c['sample_sd'] == pytest.approx(sd)
    assert c['ci_low'] == pytest.approx(.1 - 2.241402727604947 * sd / math.sqrt(n))
    assert result['main_signal'] is True
    zero = paired_bce_comparison(model, model, model)
    assert zero['comparisons']['constant']['uncertain'] and not zero['main_signal']
    extreme = baseline.copy()
    extreme[-1] += 1000
    out = paired_bce_comparison(model, extreme, baseline)
    assert out['comparisons']['constant']['max_centered_square_fraction'] > .01
    assert out['main_signal'] is False
    nonfinite = model.copy()
    nonfinite[0, 0] = np.nan
    assert not paired_bce_comparison(nonfinite, baseline, baseline)['main_signal']


def test_exact_mcnemar_holm_and_threshold() -> None:
    assert mcnemar_exact(0, 0) == 1
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert mcnemar_exact(3, 1) == pytest.approx(.625)
    assert mcnemar_exact(200000, 200000) == 1
    labels = np.zeros((100, 4), dtype=np.uint8)
    model = np.full((100, 4), -1.)
    base = model.copy()
    base[:20] = 1
    items = accuracy_comparisons(model, base, base, labels)
    assert len(items) == 8
    assert all(row['accuracy_signal'] for row in items)
    assert all(row['adjusted_p'] == pytest.approx(8 * 2 ** -19) for row in items)


def test_all_equal_input_stump_tie_selects_smallest_lsb0() -> None:
    base = fit_baselines(np.zeros(8, dtype=np.uint64), np.zeros((8, 4), dtype=np.uint8))
    assert base.stump_bits.tolist() == [0] * 4
    assert np.all(base.stump_probabilities[:, 0] == .1)
    assert np.all(base.stump_probabilities[:, 1] == .5)


def test_metrics_manual_confusion_and_logit_zero_tie() -> None:
    y = np.tile(np.array([0, 0, 1, 1], dtype=np.uint8)[:, None], (1, 4))
    z = np.tile(np.array([-1., 1., 0., 1.])[:, None], (1, 4))
    m = relation_metrics(z, y)[0]
    assert m['confusion'] == {'tn': 1, 'fp': 1, 'fn': 1, 'tp': 1}
    assert m['accuracy'] == m['balanced_accuracy'] == m['recall_0'] == m['recall_1'] == .5
    assert m['mcc'] == 0


def test_practical_threshold_requires_both_baselines() -> None:
    n = 1000
    model = np.full((n, 4), .5)
    strong = model + np.linspace(.02, .04, n)[:, None]
    weak = model + np.linspace(.001, .002, n)[:, None]
    result = paired_bce_comparison(model, strong, weak)
    assert result['comparisons']['stump']['ci_low'] > 0
    assert not result['main_signal']
    out = accuracy_comparisons(np.full((1000, 4), -1.), np.full((1000, 4), -1.),
                               np.full((1000, 4), -1.), np.zeros((1000, 4), dtype=np.uint8))
    assert all(not row['accuracy_signal'] for row in out)


def test_mcnemar_matches_integer_binomial_tails() -> None:
    for n in range(1, 40):
        for k in range(n // 2 + 1):
            expected = min(1., 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
            assert mcnemar_exact(k, n - k) == pytest.approx(expected, abs=1e-13)


def test_nonzero_constant_paired_difference_is_still_zero_variance() -> None:
    model = np.full((1000, 4), .4)
    base = np.full((1000, 4), .5)
    out = paired_bce_comparison(model, base, base)
    assert out['comparisons']['constant']['uncertain']
    assert out['comparisons']['constant']['sample_sd'] == 0.
    assert out['main_signal'] is False

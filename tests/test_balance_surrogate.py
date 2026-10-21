"""Balance scorer: features, engine labels and split-by-cube evaluation."""
from __future__ import annotations

import random

import pytest
import torch

from cipher_experiments.representations.balance_surrogate import (
    Sample,
    cube_key,
    fit_scorer,
    label_with_engine,
    make_features,
    sample_questions,
    score,
    to_tensors,
)


def test_features_encode_rounds_cube_and_target() -> None:
    sample = Sample(rounds=4, cube=(0, 5), target=7, balanced=False)
    features = make_features(sample)
    assert len(features) == 130
    assert features[0] == 4 / 16 and features[1] == 2 / 32
    assert features[2 + 0] == 1.0 and features[2 + 5] == 1.0
    assert sum(features[2:66]) == 2.0
    assert features[66 + 7] == 1.0 and sum(features[66:]) == 1.0


@pytest.mark.parametrize("sample,error", [
    (Sample(0, (1,), 0, False), ValueError),
    (Sample(4, (1,), 64, False), ValueError),
    (Sample(4, (64,), 0, False), ValueError),
])
def test_features_reject_bad_questions(sample: Sample, error: type[Exception]) -> None:
    with pytest.raises(error):
        make_features(sample)


def test_engine_labels_are_deterministic_and_consistent() -> None:
    questions = sample_questions(random.Random(3), count=12, rounds_range=(2, 3),
                                 size_range=(1, 3))
    first = label_with_engine(questions)
    second = label_with_engine(questions)
    assert [s.balanced for s in first] == [s.balanced for s in second]
    assert len({cube_key(s) for s in first}) == len(first)


def test_scorer_trains_and_scores_within_range() -> None:
    questions = label_with_engine(sample_questions(random.Random(4), count=40,
                                                   rounds_range=(1, 2), size_range=(1, 2)))
    model, losses = fit_scorer(questions, seed=1, epochs=3, hidden=16)
    assert losses[-1] <= losses[0]
    probabilities = score(model, questions)
    assert probabilities.shape == (40,)
    assert bool(((probabilities >= 0) & (probabilities <= 1)).all())
    with pytest.raises(ValueError, match="hidden"):
        fit_scorer(questions, hidden=4)


def test_tensors_have_expected_shapes() -> None:
    questions = [Sample(3, (1, 2), 0, True), Sample(3, (1,), 1, False)]
    features, labels = to_tensors(questions)
    assert features.shape == (2, 130) and labels.shape == (2, 1)
    assert labels.flatten().tolist() == [1.0, 0.0]
    assert features.dtype == torch.float32

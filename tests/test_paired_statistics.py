"""Tiny artificial panels, always the frozen 10000 resamples, no fitting."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cipher_experiments.paired_statistics import (
    ConstantBaseline,
    paired_comparison,
    paired_interval,
)
from cipher_experiments.prediction_evidence import (
    PredictionEvidence,
    load_prediction_evidence,
    write_prediction_evidence,
)


def evidence(**changes: Any) -> PredictionEvidence:
    labels = np.array([1, 0, 0, 1], dtype=np.uint8)
    values: dict[str, Any] = dict(
        logits=np.array([2., -3., 0., -1.], dtype=np.float32), labels=labels,
                  rows=4, split="test", x_sha256="a" * 64,
                  y_sha256=hashlib.sha256(labels.astype(np.float32).tobytes()).hexdigest(),
                  config_sha256="c" * 64, checkpoint_sha256="d" * 64)
    values.update(changes)
    return PredictionEvidence(**values)


def baseline(e: PredictionEvidence, **changes: Any) -> ConstantBaseline:
    values: dict[str, Any] = dict(bit=0, probability=.5, x_sha256=e.x_sha256, y_sha256=e.y_sha256,
                  rows=e.rows, split=e.split)
    values.update(changes)
    return ConstantBaseline(**values)


@pytest.mark.parametrize("difference", [0., .125, -.125])
def test_constant_difference_and_zero(difference: float) -> None:
    first = np.full(3, difference, dtype=np.float64)
    result = paired_interval(first, np.zeros(3), block=2, comparator_id=2, metric_id=1)
    assert result == {"point": difference, "lower": difference, "upper": difference,
                      "resamples": 10000, "seed": 197022, "method": "percentile",
                      "cell_alpha": .05 / 18}


def test_independent_scalar_pcg64_linear_quantile_oracle_and_rng_isolation() -> None:
    first = np.array([0., 1., .4, -.2, 1.7, -.8, .9])
    second = np.array([.3, 0., -.1, -.2, .2, .4, .5])
    rng = np.random.Generator(np.random.PCG64(196801 + 100 + 10 + 1))
    samples = np.array([sum(float(first[j] - second[j]) for j in rng.integers(0, 7, 7)) / 7
                        for _ in range(10000)])
    expected = np.quantile(samples, [.05 / 18 / 2, 1 - .05 / 18 / 2], method="linear")
    state = np.random.get_state()
    actual = paired_interval(first, second, block=1, comparator_id=1, metric_id=1)
    assert actual["seed"] == 196912 and actual["resamples"] == 10000
    assert actual["point"] == float(np.mean(first - second))
    assert [actual["lower"], actual["upper"]] == pytest.approx(expected, abs=1e-15)
    assert actual == paired_interval(first, second, block=1, comparator_id=1, metric_id=1)
    assert np.random.get_state()[0] == state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], state[1])
    assert np.random.get_state()[2:] == state[2:]
    reverse = paired_interval(second, first, block=1, comparator_id=1, metric_id=1)
    assert reverse["point"] == -float(actual["point"])
    assert reverse["lower"] == pytest.approx(-float(actual["upper"]), abs=1e-15)
    assert reverse["upper"] == pytest.approx(-float(actual["lower"]), abs=1e-15)


def test_all_eighteen_cells_keep_alpha_and_distinct_seeds() -> None:
    cells = [paired_interval(np.zeros(1), np.zeros(1), block=b, comparator_id=c, metric_id=m)
             for b in range(3) for c in range(3) for m in range(2)]
    assert len({r["seed"] for r in cells}) == 18
    assert all(r["cell_alpha"] == .05 / 18 for r in cells)


def test_resampling_chunks_never_exceed_64(monkeypatch: pytest.MonkeyPatch) -> None:
    real = np.random.Generator
    sizes: list[tuple[int, int]] = []

    class Spy:
        def __init__(self, bit_generator: Any) -> None:
            self.rng = real(bit_generator)

        def integers(self, low: int, high: int, size: tuple[int, int]) -> Any:
            sizes.append(size)
            return self.rng.integers(low, high, size=size)

    monkeypatch.setattr(np.random, "Generator", Spy)
    paired_interval(np.arange(7, dtype=np.float64), np.zeros(7),
                    block=0, comparator_id=0, metric_id=0)
    assert sum(size[0] for size in sizes) == 10000
    assert all(0 < size[0] <= 64 and size[1] == 7 for size in sizes)


@pytest.mark.parametrize("kwargs", [
    {"block": True}, {"block": -1}, {"block": 3}, {"block": np.int64(0)},
    {"comparator_id": 3}, {"comparator_id": False}, {"metric_id": 2}, {"metric_id": .0},
])
def test_reject_invalid_cell(kwargs: dict[str, Any]) -> None:
    params = dict(block=0, comparator_id=0, metric_id=0)
    params.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        paired_interval(np.zeros(2), np.zeros(2), **params)


@pytest.mark.parametrize("first,second", [
    (np.zeros(2, dtype=np.float32), np.zeros(2)), (np.zeros(0), np.zeros(0)),
    (np.zeros((2, 1)), np.zeros((2, 1))), (np.zeros(2), np.zeros(3)),
    (np.array([np.nan]), np.zeros(1)), (np.zeros(1), np.array([np.inf])),
    ([0., 1.], np.zeros(2)),
])
def test_reject_invalid_contributions(first: Any, second: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        paired_interval(first, second, block=0, comparator_id=0, metric_id=0)


def test_comparison_model_and_constant_signs_fp64_no_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = evidence(logits=np.array([100., -100., -100., 100.], dtype=np.float32))
    comparator = replace(candidate, logits=-candidate.logits, config_sha256="e" * 64,
                         checkpoint_sha256="f" * 64)
    result = paired_comparison(candidate, comparator, block=0, comparator_id=2)
    assert result["accuracy"]["point"] == 1.
    assert result["bce"]["point"] == 100.  # no sigmoid/log clipping at 1e-12
    seen: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int]] = []
    real = paired_interval

    def spy(a: Any, b: Any, **kwargs: Any) -> Any:
        seen.append((a.copy(), b.copy(), kwargs["metric_id"]))
        return real(a, b, **kwargs)

    monkeypatch.setattr("cipher_experiments.paired_statistics.paired_interval", spy)
    result = paired_comparison(candidate, baseline(candidate, bit=0, probability=.9),
                               block=0, comparator_id=0)
    np.testing.assert_array_equal(seen[0][0], [1., 1., 1., 1.])
    np.testing.assert_array_equal(seen[0][1], [0., 1., 1., 0.])
    assert [entry[2] for entry in seen] == [0, 1]
    assert result["bce"]["point"] > 0
    assert np.mean(seen[1][0] - seen[1][1]) == result["bce"]["point"]


def test_train_tie_and_saved_q_tie_use_bits_not_probability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TRAIN residual exactly half ones, so residual majority=0, smoothed probability=.5.
    import torch

    from cipher_experiments.evaluation import evaluate_model

    labels = np.array([1, 0, 0, 1], dtype=np.uint8)
    ref = np.array([1, 1, 0, 0], dtype=np.uint8)
    assert np.sum(labels ^ ref) == 2
    e = evidence(reference_bits=ref, reference_majority=0, reference_probability=.5)
    index = write_prediction_evidence(
        tmp_path, split=e.split, logits=e.logits, labels=e.labels, x_sha256=e.x_sha256,
        y_sha256=e.y_sha256, config_sha256=e.config_sha256, checkpoint_sha256=e.checkpoint_sha256,
        reference_bits=ref, reference_majority=0, reference_probability=.5,
        train_majority=0, train_probability=.5)
    saved = load_prediction_evidence(tmp_path / index["manifest"])
    correct: list[np.ndarray[Any, Any]] = []
    real = paired_interval

    def spy(a: Any, b: Any, **kwargs: Any) -> Any:
        if kwargs["metric_id"] == 0:
            correct.append(b.copy())
        return real(a, b, **kwargs)

    monkeypatch.setattr("cipher_experiments.paired_statistics.paired_interval", spy)
    q = baseline(saved, reference_bits=saved.reference_bits)
    paired_comparison(saved, q, block=0, comparator_id=1)
    paired_comparison(saved, baseline(saved), block=0, comparator_id=0)
    np.testing.assert_array_equal(correct[0], ref == labels)
    np.testing.assert_array_equal(correct[1], labels == 0)
    assert not np.array_equal(correct[0], correct[1])

    class Fixed(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    online = evaluate_model(Fixed(), torch.from_numpy(e.logits.copy()).reshape(-1, 1),
                            torch.from_numpy(labels.astype(np.float32)).reshape(-1, 1),
                            device=torch.device("cpu"), batch_size=2, majority=0, probability=.5,
                            zero_constant_reference=torch.from_numpy(ref.astype(np.float32)).reshape(
                                -1, 1), zero_constant_majority=0, zero_constant_probability=.5)
    assert online["zero_round_constant_reference_constant"]["accuracy"] == np.mean(correct[0])


@pytest.mark.parametrize("changes", [
    {"x_sha256": "b" * 64}, {"y_sha256": "b" * 64}, {"rows": 3}, {"split": "dev"},
])
def test_reject_baseline_wrong_identity(changes: dict[str, Any]) -> None:
    e = evidence()
    with pytest.raises(ValueError):
        paired_comparison(e, baseline(e, **changes), block=0, comparator_id=0)


def test_reject_model_wrong_row_order_not_just_label_distribution() -> None:
    e = evidence()
    reordered = replace(e, x_sha256="b" * 64)
    with pytest.raises(ValueError):
        paired_comparison(e, reordered, block=0, comparator_id=2)
    y = e.labels[[1, 0, 2, 3]].copy()
    other = replace(e, labels=y, y_sha256=hashlib.sha256(
        y.astype(np.float32).tobytes()).hexdigest())
    with pytest.raises(ValueError):
        paired_comparison(e, other, block=0, comparator_id=2)
    with pytest.raises(ValueError, match="self"):
        paired_comparison(e, e, block=0, comparator_id=2)


@pytest.mark.parametrize("changes", [
    {"bit": True}, {"bit": 2}, {"probability": .0}, {"probability": 1.},
    {"probability": float("nan")}, {"probability": np.float64(.5)},
    {"reference_bits": np.array([1, 0, 2, 0], dtype=np.uint8)},
    {"reference_bits": np.zeros(4, dtype=np.float32)}, {"reference_bits": np.zeros(3, np.uint8)},
    {"rows": True}, {"x_sha256": "invalid"},
])
def test_reject_invalid_constant(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        baseline(evidence(), **changes)


def test_extreme_probability_bce_is_not_silently_clipped() -> None:
    e = evidence()
    result = paired_comparison(e, baseline(e, probability=1e-100), block=0, comparator_id=0)
    assert result["bce"]["point"] > 100.


@pytest.mark.parametrize("bit,probability", [(0, .5), (1, .3), (0, .9)])
def test_q_loss_and_accuracy_row_contributions_are_independently_defined(
    bit: int, probability: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    e = evidence()
    ref = np.array([1, 0, 1, 0], dtype=np.uint8)
    seen: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = []
    real = paired_interval

    def spy(a: Any, b: Any, **kwargs: Any) -> Any:
        seen.append((a.copy(), b.copy()))
        return real(a, b, **kwargs)

    monkeypatch.setattr("cipher_experiments.paired_statistics.paired_interval", spy)
    paired_comparison(e, baseline(e, bit=bit, probability=probability, reference_bits=ref),
                      block=2, comparator_id=1)
    np.testing.assert_array_equal(seen[0][1], ((ref ^ bit) == e.labels).astype(np.float64))
    probabilities = np.where(ref == 1, 1 - probability, probability)
    expected_loss = -(e.labels * np.log(probabilities)
                      + (1 - e.labels) * np.log1p(-probabilities))
    np.testing.assert_allclose(seen[1][0], expected_loss, rtol=0, atol=1e-15)
    z = e.logits.astype(np.float64)
    np.testing.assert_array_equal(seen[1][1], np.logaddexp(0., z) - e.labels * z)


def test_different_method_identity_does_not_require_config_checkpoint_equality() -> None:
    e = evidence()
    other = replace(e, config_sha256="f" * 64)
    result = paired_comparison(e, other, block=1, comparator_id=2)
    assert result["accuracy"]["point"] == result["bce"]["point"] == 0.


def test_postprocessing_records_numpy_and_bit_generator_environment() -> None:
    e = evidence()
    result = paired_comparison(e, baseline(e), block=0, comparator_id=0)
    assert result["environment"] == {"numpy": np.__version__, "bit_generator": "PCG64"}

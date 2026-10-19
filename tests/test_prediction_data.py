from __future__ import annotations

import random

import pytest
import torch
from torch import Tensor

from cipher_experiments.data import make_simon, prediction
from cipher_experiments.data.prediction import make_inputs, make_labels


def _joined(inputs: dict[str, Tensor]) -> Tensor:
    assert tuple(inputs) == ("train", "dev", "test")
    return torch.cat(tuple(inputs.values()), dim=0)


def test_make_inputs_is_unique_deterministic_and_has_no_labels() -> None:
    first = make_inputs(7, 4, 3, seed=17)
    replay = make_inputs(7, 4, 3, seed=17)
    assert tuple(first) == ("train", "dev", "test")
    assert [first[name].shape for name in first] == [(7, 64), (4, 64), (3, 64)]
    all_inputs = _joined(first)
    assert torch.equal(all_inputs, _joined(replay))
    assert all(
        value.dtype == torch.float32 and value.device.type == "cpu" for value in first.values()
    )
    assert all(not value.requires_grad for value in first.values())
    assert bool(((all_inputs == 0) | (all_inputs == 1)).all())
    assert torch.unique(all_inputs, dim=0).shape[0] == all_inputs.shape[0]


def test_simon_labels_match_existing_generator_with_same_plaintexts() -> None:
    counts = (8, 5, 4)
    options = {"seed": 77, "key_seed": 19, "rounds": 7, "target_bit": 0}
    inputs = make_inputs(*counts, seed=options["seed"])
    legacy = make_simon(*counts, **options)
    for name in inputs:
        actual = make_labels(
            inputs[name],
            dataset="simon",
            rounds=options["rounds"],
            target_bit=options["target_bit"],
            key_seed=options["key_seed"],
            split=name,
        )
        assert torch.equal(inputs[name], legacy[name][0])
        assert torch.equal(actual, legacy[name][1])
        assert actual.shape == (counts[("train", "dev", "test").index(name)], 1)
        assert actual.dtype == torch.float32 and actual.device.type == "cpu"
        assert not actual.requires_grad


def test_synthetic_labels_use_first_input_bit_without_cipher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> int:
        raise AssertionError("synthetic labels must not encrypt")

    monkeypatch.setattr(prediction, "_encrypt", unexpected)
    x = make_inputs(3, 2, 2, seed=31)["dev"]
    y = make_labels(
        x,
        dataset="synthetic",
        rounds=3,
        target_bit=0,
        key_seed=4,
        split="dev",
    )
    assert torch.equal(y, x[:, :1])


def test_null_labels_are_replayable_split_scoped_and_use_local_rng() -> None:
    inputs = make_inputs(32, 32, 32, seed=53)
    random.seed(9901)
    before = random.getstate()
    labels = {
        split: make_labels(
            inputs[split],
            dataset="simon",
            rounds=4,
            target_bit=11,
            key_seed=71,
            label_mode="null",
            null_seed=191302,
            split=split,
        )
        for split in inputs
    }
    after = random.getstate()
    assert before == after
    for split in inputs:
        replay = make_labels(
            inputs[split],
            dataset="simon",
            rounds=4,
            target_bit=11,
            key_seed=71,
            label_mode="null",
            null_seed=191302,
            split=split,
        )
        assert torch.equal(labels[split], replay)
        assert bool(((replay == 0) | (replay == 1)).all())
    assert not torch.equal(labels["train"], labels["dev"])
    assert not torch.equal(labels["train"], labels["test"])
    assert not torch.equal(labels["dev"], labels["test"])


def test_null_stream_is_independent_of_other_split_calls() -> None:
    inputs = make_inputs(12, 12, 12, seed=53)

    def label(split: str) -> Tensor:
        return make_labels(
            inputs[split],
            dataset="simon",
            rounds=3,
            target_bit=0,
            key_seed=4,
            label_mode="null",
            split=split,
        )

    expected_test = label("test")
    label("train")
    label("dev")
    assert torch.equal(expected_test, label("test"))


@pytest.mark.parametrize(
    ("counts", "seed", "error"),
    [((0, 1, 1), 1, ValueError), ((1, 1, 1), True, TypeError), ((1, 1, 1), -1, ValueError)],
)
def test_make_inputs_rejects_invalid_counts_and_seed(
    counts: tuple[int, int, int], seed: int, error: type[Exception]
) -> None:
    with pytest.raises(error):
        make_inputs(*counts, seed=seed)


@pytest.mark.parametrize(
    "bad_inputs",
    [
        torch.zeros((1, 63)),
        torch.zeros((1, 64), dtype=torch.float64),
        torch.zeros((1, 64), requires_grad=True),
        torch.full((1, 64), float("nan")),
        torch.full((1, 64), 0.5),
        torch.empty((0, 64)),
    ],
)
def test_labels_reject_invalid_input_tensors(bad_inputs: Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_labels(
            bad_inputs,
            dataset="synthetic",
            rounds=1,
            target_bit=0,
            key_seed=1,
            split="train",
        )


@pytest.mark.parametrize(
    "options",
    [
        {"dataset": "unknown"},
        {"label_mode": "unknown"},
        {"label_mode": "null", "dataset": "synthetic"},
        {"split": "validation"},
        {"rounds": 45},
        {"rounds": True},
        {"target_bit": 64},
        {"target_bit": False},
        {"key_seed": True},
        {"null_seed": -1},
    ],
)
def test_labels_reject_invalid_parameters(options: dict[str, object]) -> None:
    kwargs: dict[str, object] = {
        "dataset": "simon",
        "rounds": 1,
        "target_bit": 0,
        "key_seed": 1,
        "split": "train",
    }
    kwargs.update(options)
    with pytest.raises((TypeError, ValueError)):
        make_labels(torch.zeros((1, 64)), **kwargs)  # type: ignore[arg-type]

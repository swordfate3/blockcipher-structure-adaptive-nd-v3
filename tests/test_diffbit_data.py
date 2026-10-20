"""Fixed-difference output-bit task: key-free r1 identity and disjoint key splits."""
from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

import pytest
import torch

from cipher_experiments.ciphers.simon import encrypt
from cipher_experiments.config import ExperimentConfig
from cipher_experiments.data.diffbit import (
    bias_summary,
    key_for,
    key_indices,
    make_diff_inputs,
    make_diff_labels,
    round_one_reference,
)
from cipher_experiments.training import input_panels, labels_for, reference_panels

DELTA = 0x0000000000000040


def config(**patch: Any) -> ExperimentConfig:
    base = ExperimentConfig(name="diff-fixture", dataset="simon-diff", rounds=1,
                            hidden_width=64, model_kind="mlp", train_rows=8, dev_rows=8,
                            test_rows=8, batch_size=4, epochs=1, data_seed=7,
                            input_difference=DELTA, key_count=4, train_key_count=2)
    return replace(base, **patch)


@pytest.mark.parametrize("target_bit", [0, 31, 33, 63])
def test_one_round_difference_is_key_free_and_exact(target_bit: int) -> None:
    """The r1 differential must equal real encryption for any key, so r1 is a control."""
    x = make_diff_inputs(8, 8, 8, seed=11)["train"]
    reference = round_one_reference(x, delta=DELTA, target_bit=target_bit)
    plaintexts = [_pack(row) for row in x.to(torch.int64).tolist()]
    for seed in (1, 2, 3):
        key = random.Random(seed).getrandbits(128)
        for index, plaintext in enumerate(plaintexts):
            difference = (encrypt(plaintext, key, 1) ^ encrypt(plaintext ^ DELTA, key, 1))
            expected = float(bool(difference & (1 << (63 - target_bit))))
            assert reference[index, 0].item() == expected


def test_most_sparse_differences_give_degenerate_bits() -> None:
    """Sparse input differences leave many output bits constant.

    This pins the trap behind the task: a constant predictor can look perfect, so
    every candidate target must be screened with ``bias_summary`` first.
    """
    x = make_diff_inputs(64, 8, 8, seed=13)["train"]
    constant = make_diff_labels(x, delta=0x1, rounds=2, target_bit=63, key_seed=5,
                                key_count=4, train_key_count=2, split="train")
    summary = bias_summary(constant, keys_per_row=32)
    assert summary["positive_rate"] == 1.0 and summary["spread"] == 0.0

    zero = make_diff_labels(x, delta=0x40, rounds=2, target_bit=0, key_seed=5,
                            key_count=4, train_key_count=2, split="train")
    degenerate = bias_summary(zero, keys_per_row=32)
    assert degenerate["positive_rate"] == 0.0 and degenerate["spread"] == 0.0


def test_screened_target_has_intermediate_bias() -> None:
    """A screened (delta, round, bit) choice must not be near 0 or 1."""
    x = make_diff_inputs(256, 8, 8, seed=21)["train"]
    labels = make_diff_labels(x, delta=0x8000, rounds=3, target_bit=0, key_seed=5,
                             key_count=4, train_key_count=2, split="train")
    summary = bias_summary(labels, keys_per_row=128)
    assert 0.05 < summary["positive_rate"] < 0.95
    assert labels.shape == (256, 1)


def test_train_and_evaluation_splits_use_disjoint_keys() -> None:
    assert key_indices("train", 4, 2) == (0, 1)
    assert key_indices("dev", 4, 2) == (2, 3)
    assert key_indices("test", 4, 2) == (2, 3)
    assert set(key_indices("train", 4, 2)).isdisjoint(key_indices("test", 4, 2))
    assert key_for(9, 0) != key_for(9, 1) and key_for(9, 0) == key_for(9, 0)


def test_training_wiring_uses_diffbit_panels_and_r1_reference() -> None:
    cfg = config()
    panels = input_panels(cfg)
    assert panels["train"].shape == (8, 64)
    labels = labels_for(cfg, panels["train"], "train")
    assert labels.shape == (8, 1)
    reference = reference_panels(cfg, panels)
    assert reference and torch.equal(reference["train"],
                                     round_one_reference(panels["train"], delta=DELTA,
                                                         target_bit=0))
    # No key-free reference exists once the differential propagates past one round.
    assert reference_panels(config(rounds=2), panels) == {}


def test_bias_summary_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        bias_summary(torch.tensor([[0.5]]), keys_per_row=1)


def test_null_labels_are_split_separated() -> None:
    cfg = config(label_mode="null")
    x = input_panels(cfg)["train"]
    first = labels_for(cfg, x, "train")
    second = labels_for(cfg, x, "train")
    assert torch.equal(first, second) and first.shape == (8, 1)
    assert not torch.equal(first, labels_for(cfg, x, "dev"))


@pytest.mark.parametrize("patch", [
    {"input_difference": None}, {"input_difference": 0}, {"input_difference": 1 << 64},
    {"key_count": 1}, {"train_key_count": 4}, {"train_key_count": None}, {"rounds": 7},
])
def test_diff_config_requires_held_out_keys(patch: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        config(**patch)


def test_diff_fields_are_rejected_outside_simon_diff() -> None:
    with pytest.raises(ValueError, match="simon-diff"):
        ExperimentConfig(name="x", dataset="simon", rounds=3, hidden_width=64,
                         input_difference=DELTA)


def test_shared_key_ranges_do_not_leak_into_panels() -> None:
    """Sanity check that the label path really consults the configured key range."""
    cfg = config(train_rows=4, dev_rows=4, test_rows=4)
    same = labels_for(cfg, input_panels(cfg)["train"], "train")
    wider = labels_for(config(train_rows=4, dev_rows=4, test_rows=4, key_count=6,
                              train_key_count=2), input_panels(cfg)["train"], "train")
    assert torch.equal(same, wider)  # TRAIN keys unchanged when only the pool grows


def _pack(row: list[int]) -> int:
    value = 0
    for bit in row:
        value = (value << 1) | int(bit)
    return value

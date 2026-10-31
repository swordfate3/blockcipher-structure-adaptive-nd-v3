"""Low-round configuration contracts only: no encryption, fitting, or model forward."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import NoReturn

import pytest
import torch
from torch import Tensor

from cipher_experiments import training
from cipher_experiments.ciphers import simon
from cipher_experiments.config import ExperimentConfig, load_config
from cipher_experiments.data import prediction
from cipher_experiments.models import prediction as model_prediction

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("bit", "xor2", "xor3", "xor-and", "null")
ROUNDS = (3, 4)
BITS = {"bit": (), "xor2": (0, 8), "xor3": (0, 8, 16), "xor-and": (0, 8, 16)}


def folder(*, smoke: bool, low_round: bool = True) -> Path:
    return ROOT / ("configs/innovation2-cipher-expression"
                   + ("-low-round" if low_round else "") + ("-smoke" if smoke else ""))


def configs(*, smoke: bool) -> list[ExperimentConfig]:
    return [load_config(folder(smoke=smoke) / f"r{rounds}-{arm}.json")
            for rounds in ROUNDS for arm in ARMS]


@pytest.fixture(autouse=True)
def forbid_scientific_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("encryption, real public references, and model forwards forbidden")

    monkeypatch.setattr(simon, "encrypt", forbidden)
    monkeypatch.setattr(prediction, "_encrypt", forbidden)
    monkeypatch.setattr(training, "public_reference", forbidden)
    monkeypatch.setattr(model_prediction, "public_reference", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)


@pytest.mark.parametrize("smoke", [False, True])
def test_exact_ten_file_matrix(smoke: bool) -> None:
    assert {p.name for p in folder(smoke=smoke).iterdir()} == {
        f"r{rounds}-{arm}.json" for rounds in ROUNDS for arm in ARMS
    }


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("rounds", ROUNDS)
@pytest.mark.parametrize("arm", ARMS)
def test_only_rounds_and_name_differ_from_corresponding_r6(
    smoke: bool, rounds: int, arm: str,
) -> None:
    source = folder(smoke=smoke, low_round=False) / f"{arm}.json"
    target = folder(smoke=smoke) / f"r{rounds}-{arm}.json"
    original_raw: dict[str, object] = json.loads(source.read_text(encoding="utf-8"))
    actual_raw: dict[str, object] = json.loads(target.read_text(encoding="utf-8"))
    original = load_config(source)
    actual = load_config(target)
    assert original.rounds == 6 and original.name.startswith("r6-")
    expected_name = f"r{rounds}-" + original.name[3:]
    assert actual_raw == original_raw | {"rounds": rounds, "name": expected_name}
    assert {key for key in actual_raw if actual_raw[key] != original_raw[key]} == {
        "rounds", "name",
    }
    # Check all loaded fields too, including defaults omitted from serialized JSON.
    for field in fields(ExperimentConfig):
        expected = {"rounds": rounds, "name": expected_name}.get(
            field.name, getattr(original, field.name))
        assert getattr(actual, field.name) == expected, field.name
    expression = "bit" if arm == "null" else arm
    assert actual.label_expression == expression and actual.label_bits == BITS[expression]
    assert (actual.dataset, actual.experiment, actual.model_kind, actual.hidden_width,
            actual.model_features, actual.prediction_mode, actual.target_bit) == (
                "simon", "prediction", "mlp", 128, "full", "direct", 0)
    assert actual.label_mode == ("null" if smoke or arm == "null" else "cipher")


@pytest.mark.parametrize("rounds", ROUNDS)
@pytest.mark.parametrize("arm", ARMS)
def test_formal_cpu_parameter_count_without_forward(rounds: int, arm: str) -> None:
    config = load_config(folder(smoke=False) / f"r{rounds}-{arm}.json")
    # Construct on CPU, irrespective of the formal config's declared CUDA device.
    model = model_prediction.build_predictor(
        config.hidden_width, model_kind=config.model_kind, rounds=config.rounds,
        target_bit=config.target_bit, feature_mode=config.model_features,
        prediction_mode=config.prediction_mode,
    )
    assert config.hidden_width == 128
    assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}
    assert sum(parameter.numel() for parameter in model.parameters()) == 24961


def test_formal_budget_arithmetic_not_executed_work() -> None:
    formal = configs(smoke=False)
    for config in formal:
        assert (config.train_rows, config.dev_rows, config.test_rows,
                config.train_pool_rows, config.batch_size, config.epochs,
                config.dev_every_epochs, config.device, config.max_seconds) == (
                    65536, 8192, 65536, 65536, 256, 32, 1, "cuda:0", 1800)
        assert config.epochs * ((config.train_rows + config.batch_size - 1)
                                // config.batch_size) == 8192
        assert config.epochs * config.train_rows == 2097152
    assert sum(c.epochs * ((c.train_rows + c.batch_size - 1) // c.batch_size)
               for c in formal) == 81920
    assert sum(c.epochs * c.train_rows for c in formal) == 20971520
    # DEV selection opportunities (not extra labels): final epoch is always included.
    assert sum(sum(epoch % c.dev_every_epochs == 0 or epoch == c.epochs
                   for epoch in range(1, c.epochs + 1)) for c in formal) == 320
    cipher = [c for c in formal if c.label_mode == "cipher"]
    null = [c for c in formal if c.label_mode == "null"]
    assert len(cipher) == 8 and len(null) == 2
    # Labels are constructed once per split; expressions use one ciphertext per row,
    # not one encryption per operand, epoch, or public-reference panel.
    assert all(c.train_rows + c.dev_rows + c.test_rows == 139264 for c in formal)
    assert sum(c.train_rows + c.dev_rows + c.test_rows for c in cipher) == 8 * 139264 == 1114112
    assert sum(c.train_rows + c.dev_rows + c.test_rows for c in null) == 2 * 139264 == 278528


def test_smoke_budget_is_twenty_updates_without_running_training() -> None:
    smokes = configs(smoke=True)
    for config in smokes:
        assert (config.train_rows, config.dev_rows, config.test_rows,
                config.train_pool_rows, config.batch_size, config.epochs,
                config.dev_every_epochs, config.device, config.label_mode) == (
                    8, 4, 4, 8, 4, 1, 1, "cpu", "null")
        assert config.epochs * ((config.train_rows + config.batch_size - 1)
                                // config.batch_size) == 2
        assert config.epochs * config.train_rows == 8
    assert sum(c.epochs * ((c.train_rows + c.batch_size - 1) // c.batch_size)
               for c in smokes) == 20
    assert sum(c.epochs * c.train_rows for c in smokes) == 80


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("rounds", ROUNDS)
@pytest.mark.parametrize("arm", ARMS)
def test_reference_panels_preserve_bit_only_interface(
    monkeypatch: pytest.MonkeyPatch, smoke: bool, rounds: int, arm: str,
) -> None:
    config = load_config(folder(smoke=smoke) / f"r{rounds}-{arm}.json")
    panels = {split: torch.zeros(rows, 64)
              for split, rows in (("train", 2), ("dev", 3), ("test", 4))}
    calls: list[tuple[Tensor, int, int]] = []

    def fake_reference(inputs: Tensor, rounds: int, target_bit: int) -> Tensor:
        calls.append((inputs, rounds, target_bit))
        return inputs[:, :1]

    monkeypatch.setattr(training, "public_reference", fake_reference)
    references = training.reference_panels(config, panels)
    if config.label_expression != "bit":
        assert references == {} and calls == []
    else:
        # Includes the bit-expression null control: preserve the existing interface.
        assert list(references) == list(panels)
        assert len(calls) == len(panels)
        for (split, inputs), (seen, seen_rounds, seen_bit) in zip(
            panels.items(), calls, strict=True,
        ):
            assert seen is inputs
            assert (seen_rounds, seen_bit) == (rounds, config.target_bit)
            assert torch.equal(references[split], inputs[:, :1])

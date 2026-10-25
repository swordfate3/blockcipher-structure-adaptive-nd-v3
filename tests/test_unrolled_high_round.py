"""Synthetic contracts for high-round direct unrolled SIMON models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from cipher_experiments import training
from cipher_experiments.artifacts import file_sha256
from cipher_experiments.config import ExperimentConfig, load_config
from cipher_experiments.evaluation import load_checkpoint
from cipher_experiments.models.prediction import build_predictor
from cipher_experiments.models.unrolled_simon import UnrolledSimonPredictor


def _inputs(count: int = 8) -> Tensor:
    generator = torch.Generator().manual_seed(8018)
    return torch.randint(0, 2, (count, 64), generator=generator).float()


def _manual_unroll(model: UnrolledSimonPredictor, inputs: Tensor) -> Tensor:
    high, low = inputs[:, :32], inputs[:, 32:]
    previous_logits: Tensor | None = None
    logits: Tensor | None = None
    for round_index in range(model.rounds):
        rotated_one = torch.cat((high[:, 1:], high[:, :1]), dim=1)
        rotated_eight = torch.cat((high[:, 8:], high[:, :8]), dim=1)
        rotated_two = torch.cat((high[:, 2:], high[:, :2]), dim=1)
        local_state = torch.stack((low, rotated_one, rotated_eight, rotated_two), dim=-1)
        local_state = local_state.mul(2.0).sub(1.0)
        condition = model.conditions[round_index].unsqueeze(0).expand(inputs.shape[0], -1, -1)
        logits = model.cell(torch.cat((local_state, condition), dim=-1)).squeeze(-1)
        if round_index == model.rounds - 2:
            previous_logits = logits
        low, high = high, torch.sigmoid(logits)
    assert logits is not None
    if model.target_bit < 32:
        return logits[:, model.target_bit : model.target_bit + 1]
    assert previous_logits is not None
    bit = model.target_bit - 32
    return previous_logits[:, bit : bit + 1]


@pytest.mark.parametrize(("rounds", "expected_parameters"), [(6, 1089), (8, 1345)])
def test_factory_parameter_count_manual_unroll_and_finite_gradients(
    rounds: int, expected_parameters: int,
) -> None:
    torch.manual_seed(1800 + rounds)
    model = build_predictor(32, rounds=rounds, target_bit=0, model_kind="simon-unrolled")
    assert isinstance(model, UnrolledSimonPredictor)
    assert model.rounds == rounds and model.hidden_width == 32
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameters
    assert expected_parameters == 321 + 128 * rounds

    inputs = _inputs()
    actual = model(inputs)
    torch.testing.assert_close(actual, _manual_unroll(model, inputs), rtol=0, atol=0)
    labels = torch.randint(0, 2, (len(inputs), 1), generator=torch.Generator().manual_seed(1810))
    F.binary_cross_entropy_with_logits(actual, labels.float()).backward()  # type: ignore[no-untyped-call]
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters() if parameter.grad is not None)


def test_round_eight_config_loads_and_round_nine_is_rejected(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="unrolled-r8", dataset="simon", model_kind="simon-unrolled", rounds=8,
        hidden_width=32, target_bit=0,
    )
    path = tmp_path / "r8.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert load_config(path) == config

    invalid = config.to_dict()
    invalid["rounds"] = 9
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="rounds1..8"):
        load_config(path)


def test_round_eight_null_training_strict_loads_before_test_without_encryption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig(
        name="unrolled-r8-null-fixture", dataset="simon", model_kind="simon-unrolled",
        rounds=8, target_bit=0, hidden_width=32, label_mode="null",
        train_rows=4, dev_rows=2, test_rows=2, batch_size=2, epochs=1,
    )
    output = tmp_path / "run"
    calls: list[str] = []
    original_labels = training.labels_for
    original_load = load_checkpoint

    def labels(cfg: ExperimentConfig, inputs: Tensor, split: str) -> Tensor:
        if split == "test":
            assert calls[-1] == "loaded"
            selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
            assert selection["test_labels_generated"] is False
        calls.append(split)
        return original_labels(cfg, inputs, split)

    def strict_load(*args: Any, **kwargs: Any) -> Any:
        loaded = original_load(*args, **kwargs)
        model, saved, _, _ = loaded
        assert isinstance(model, UnrolledSimonPredictor)
        assert saved == config and model.rounds == 8
        calls.append("loaded")
        return loaded

    from cipher_experiments.data import prediction as prediction_data

    def forbidden_encrypt(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("null-label test must never perform real SIMON encryption")

    monkeypatch.setattr(prediction_data, "_encrypt", forbidden_encrypt)
    monkeypatch.setattr(training, "labels_for", labels)
    monkeypatch.setattr(training, "load_checkpoint", strict_load)
    result = training.train(config, output)

    assert calls == ["train", "dev", "loaded", "test"]
    assert result["updates"] == 2
    assert result["presentations"] == 4
    checkpoint = output / "best.pt"
    loaded, saved, _, _ = original_load(
        checkpoint, file_sha256(checkpoint), device=torch.device("cpu"),
    )
    assert isinstance(loaded, UnrolledSimonPredictor)
    assert saved == config

"""Synthetic contracts for the reversible public-coordinate MLP mode."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from cipher_experiments import training
from cipher_experiments.artifacts import file_sha256
from cipher_experiments.cli import main
from cipher_experiments.config import ExperimentConfig
from cipher_experiments.evaluation import CHECKPOINT_SCHEMA, load_checkpoint
from cipher_experiments.models.mlp import BitMLP
from cipher_experiments.models.prediction import (
    PlaintextPredictor,
    build_predictor,
    public_coordinate_inverse,
    public_coordinate_transform,
)


def _rotate_left_32(value: int, amount: int) -> int:
    mask = (1 << 32) - 1
    return ((value << amount) | (value >> (32 - amount))) & mask


def _scalar_transform(value: int) -> int:
    x, y = value >> 32, value & 0xFFFFFFFF
    f = ((_rotate_left_32(x, 1) & _rotate_left_32(x, 8))
         ^ _rotate_left_32(x, 2))
    return ((y ^ f) << 32) | x


def _bits(value: int) -> list[float]:
    return [float((value >> (63 - index)) & 1) for index in range(64)]


def _config(**changes: Any) -> ExperimentConfig:
    values: dict[str, Any] = {
        "name": "public-coordinate-toy",
        "dataset": "simon",
        "model_kind": "mlp",
        "model_features": "full",
        "prediction_mode": "public-coordinate",
        "rounds": 2,
        "target_bit": 0,
    }
    values.update(changes)
    return ExperimentConfig(**values)


def test_coordinate_transform_matches_independent_scalar_msb_round_function() -> None:
    samples = [
        0,
        (1 << 64) - 1,
        1 << 63,
        1,
        0x0123456789ABCDEF,
        0xFEDCBA9876543210,
        0x8000000180000001,
    ]
    inputs = torch.tensor([_bits(value) for value in samples], dtype=torch.float32)
    result = public_coordinate_transform(inputs)
    expected = torch.tensor([_bits(_scalar_transform(value)) for value in samples])
    assert result.shape == (len(samples), 64)
    assert result.dtype is torch.float32
    assert torch.equal(result, expected)


def test_coordinate_transform_inverse_round_trips_1024_synthetic_rows() -> None:
    inputs = torch.randint(0, 2, (1024, 64), dtype=torch.bool).float()
    coordinates = public_coordinate_transform(inputs)
    restored = public_coordinate_inverse(coordinates)
    assert coordinates.shape == (1024, 64)
    assert restored.shape == inputs.shape
    assert coordinates.dtype is torch.float32
    assert restored.dtype is torch.float32
    assert torch.equal(restored, inputs)
    assert torch.equal(public_coordinate_transform(inputs.bool()).bool(), coordinates.bool())


@pytest.mark.parametrize("bad", [
    torch.zeros((0, 64), dtype=torch.float32),
    torch.zeros((2, 63), dtype=torch.float32),
    torch.zeros((2, 64), dtype=torch.float64),
    torch.full((2, 64), 0.5, dtype=torch.float32),
])
def test_coordinate_helpers_reject_invalid_inputs(bad: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        public_coordinate_transform(bad)
    with pytest.raises((TypeError, ValueError)):
        public_coordinate_inverse(bad)


@pytest.mark.parametrize("changes", [
    {"dataset": "synthetic"},
    {"model_kind": "logic16"},
    {"model_kind": "simon-unrolled"},
    {"model_features": "support"},
    {"rounds": 1},
    {"rounds": 45},
    {"experiment": "innovation1", "dataset": "differential"},
])
def test_config_rejects_public_coordinate_outside_its_contract(
        changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _config(**changes)


@pytest.mark.parametrize("rounds", [2, 3, 44])
def test_config_accepts_supported_rounds(rounds: int) -> None:
    config = _config(rounds=rounds)
    assert config.prediction_mode == "public-coordinate"


def test_coordinate_mode_preserves_mlp_parameter_count_state_and_rng() -> None:
    torch.manual_seed(719)
    direct = PlaintextPredictor(128, rounds=2, target_bit=0, prediction_mode="direct")
    direct_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(719)
    coordinate = PlaintextPredictor(
        128, rounds=2, target_bit=0, prediction_mode="public-coordinate")
    coordinate_rng = torch.random.get_rng_state().clone()

    assert isinstance(direct, BitMLP)
    assert sum(parameter.numel() for parameter in coordinate.parameters()) == 24961
    assert direct.state_dict().keys() == coordinate.state_dict().keys()
    assert all(torch.equal(direct.state_dict()[name], coordinate.state_dict()[name])
               for name in direct.state_dict())
    assert torch.equal(direct_rng, coordinate_rng)
    assert torch.equal(coordinate.indices, torch.arange(64))
    assert "indices" not in coordinate.state_dict()
    assert coordinate.layers[0].in_features == direct.layers[0].in_features == 64


def test_coordinate_predictor_forward_is_exact_plain_network_on_transformed_input() -> None:
    torch.manual_seed(87)
    model = build_predictor(13, model_kind="mlp", rounds=9, target_bit=37,
                            feature_mode="full", prediction_mode="public-coordinate").eval()
    assert isinstance(model, PlaintextPredictor)
    inputs = torch.randint(0, 2, (19, 64), dtype=torch.bool).float()
    transformed = public_coordinate_transform(inputs)
    with torch.no_grad():
        expected = model.layers(transformed.index_select(1, torch.arange(64)))
        actual = model(inputs)
    assert actual.shape == (19, 1)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("corruption", ["shape", "dtype"])
def test_coordinate_checkpoint_strictly_rejects_corrupt_tensor(
        tmp_path: Path, corruption: str) -> None:
    config = _config(hidden_width=8)
    model = build_predictor(config.hidden_width, model_kind=config.model_kind,
                            rounds=config.rounds, target_bit=config.target_bit,
                            feature_mode=config.model_features,
                            prediction_mode=config.prediction_mode)
    state = dict(model.state_dict())
    if corruption == "shape":
        state["layers.0.weight"] = state["layers.0.weight"][:-1]
    else:
        state["layers.0.weight"] = state["layers.0.weight"].double()
    path = tmp_path / f"corrupt-{corruption}.pt"
    torch.save({
        "schema": CHECKPOINT_SCHEMA,
        "config": config.to_dict(),
        "config_sha256": config.sha256(),
        "state_dict": state,
        "epoch": 1,
        "updates": 1,
        "majority": 0,
        "probability": 0.5,
    }, path)
    with pytest.raises(ValueError, match="state mismatch"):
        load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))


def test_null_training_strictly_loads_before_test_and_explicit_evaluate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(name="public-coordinate-null-toy", hidden_width=8, label_mode="null",
                     train_rows=8, dev_rows=4, test_rows=4, batch_size=4,
                     epochs=1, max_seconds=60)
    output = tmp_path / "toy-run"
    inputs = {
        split: torch.randint(0, 2, (rows, 64),
                             generator=torch.Generator().manual_seed(900 + rows)).float()
        for split, rows in (("train", 8), ("dev", 4), ("test", 4))
    }
    labels_by_split = {
        split: torch.randint(0, 2, (len(x), 1),
                             generator=torch.Generator().manual_seed(2000 + len(x))).float()
        for split, x in inputs.items()
    }
    load_completed = False
    label_requests: list[str] = []
    original_load = load_checkpoint

    def labels(_config: ExperimentConfig, x: torch.Tensor, split: str) -> torch.Tensor:
        nonlocal load_completed
        if split == "test":
            assert load_completed
            selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
            assert selection["test_labels_generated"] is False
        label_requests.append(split)
        return labels_by_split[split].clone()

    def strict_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal load_completed
        loaded = original_load(*args, **kwargs)
        load_completed = True
        return loaded

    monkeypatch.setattr(training, "input_panels", lambda _: inputs)
    monkeypatch.setattr(training, "labels_for", labels)
    monkeypatch.setattr(training, "load_checkpoint", strict_load)
    from cipher_experiments.data import prediction as prediction_data

    monkeypatch.setattr(
        prediction_data, "_encrypt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("null toy workflow must not invoke cipher encryption")),
    )
    result = training.train(config, output)
    assert result["updates"] == 2
    assert label_requests == ["train", "dev", "test"]
    initialization = json.loads((output / "initialization.json").read_text(encoding="utf-8"))
    torch.manual_seed(config.init_seed)
    equivalent_direct = build_predictor(
        config.hidden_width, model_kind="mlp", rounds=config.rounds,
        target_bit=config.target_bit, feature_mode="full", prediction_mode="direct")
    expected_state_sha = hashlib.sha256(b"".join(
        name.encode() + value.detach().cpu().contiguous().numpy().tobytes()
        for name, value in equivalent_direct.state_dict().items())).hexdigest()
    expected_trainable_sha = hashlib.sha256(b"".join(
        name.encode() + value.detach().cpu().contiguous().numpy().tobytes()
        for name, value in equivalent_direct.named_parameters())).hexdigest()
    assert initialization["state_sha256"] == expected_state_sha
    assert initialization["trainable_sha256"] == expected_trainable_sha
    assert initialization["parameters"] == sum(
        parameter.numel() for parameter in equivalent_direct.parameters())
    assert initialization["features"] == list(range(64))
    metadata = json.loads((output / "model_metadata.json").read_text(encoding="utf-8"))
    assert metadata["interpretation"] == (
        "public bijective coordinates only; not actual cipher state")
    assert metadata["learned_constants"] == "none"
    assert metadata["extra_labels"] == "none; absolute final target-bit labels only"
    assert "no actual key, trace, ciphertext, or other labels" in metadata["input"]

    checkpoint = output / "best.pt"
    loaded, saved, _, _ = load_checkpoint(checkpoint, file_sha256(checkpoint),
                                           device=torch.device("cpu"))
    assert isinstance(loaded, PlaintextPredictor)
    assert saved == config

    # The explicit evaluate command is separately tested using the same synthetic panels.
    monkeypatch.setattr(training, "datasets", lambda _: {
        split: (inputs[split], labels_by_split[split]) for split in inputs
    })
    evaluation_output = tmp_path / "explicit-evaluation"
    assert main(["evaluate", "--checkpoint", str(checkpoint), "--sha256",
                 file_sha256(checkpoint), "--output", str(evaluation_output)]) == 0
    evaluated = json.loads((evaluation_output / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluated["split"] == "test"
    assert evaluated["config_sha256"] == config.sha256()
    assert evaluated["metrics"] == result["metrics"]["test"]


def test_legacy_direct_mlp_checkpoint_still_strictly_loads(tmp_path: Path) -> None:
    config = ExperimentConfig(name="legacy-direct", hidden_width=8)
    raw_config = config.to_dict()
    for field in ("prediction_mode", "model_features", "model_kind"):
        raw_config.pop(field)
    old_sha = hashlib.sha256(json.dumps(raw_config, sort_keys=True).encode()).hexdigest()
    old_model = BitMLP(config.hidden_width)
    path = tmp_path / "legacy-direct.pt"
    torch.save({
        "schema": CHECKPOINT_SCHEMA,
        "config": raw_config,
        "config_sha256": old_sha,
        "state_dict": dict(old_model.state_dict()),
        "epoch": 1,
        "updates": 1,
        "majority": 0,
        "probability": 0.5,
    }, path)
    loaded, saved, _, _ = load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))
    assert saved.prediction_mode == "direct"
    assert torch.equal(loaded(torch.zeros((2, 64))), old_model(torch.zeros((2, 64))))

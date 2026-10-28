"""Checkpoint serialization boundary tests use only synthetic model/configuration data."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from cipher_experiments import checkpoints, evaluation
from cipher_experiments.artifacts import file_sha256
from cipher_experiments.config import ExperimentConfig
from cipher_experiments.models.prediction import build_predictor


def _synthetic_config(*, name: str = "checkpoint-boundary") -> ExperimentConfig:
    return ExperimentConfig(name=name, hidden_width=8)


def _write_payload(path: Path, payload: object) -> str:
    torch.save(payload, path)
    return file_sha256(path)


def _valid_payload(config: ExperimentConfig) -> dict[str, Any]:
    model = build_predictor(config.hidden_width, model_kind=config.model_kind,
                            rounds=config.rounds, target_bit=config.target_bit,
                            feature_mode=config.model_features,
                            prediction_mode=config.prediction_mode)
    return {
        "schema": checkpoints.CHECKPOINT_SCHEMA,
        "config": config.to_dict(),
        "config_sha256": config.sha256(),
        "state_dict": {name: value.detach().cpu().clone()
                       for name, value in model.state_dict().items()},
        "epoch": 1,
        "updates": 2,
        "majority": 0,
        "probability": 0.5,
    }


def test_save_checkpoint_round_trip_and_legacy_reexports(tmp_path: Path) -> None:
    config = _synthetic_config()
    model = build_predictor(config.hidden_width, model_kind=config.model_kind,
                            rounds=config.rounds, target_bit=config.target_bit,
                            feature_mode=config.model_features,
                            prediction_mode=config.prediction_mode)
    path = tmp_path / "synthetic.pt"

    checkpoints.save_checkpoint(path, model, config, 3, 7, 1, 0.75)

    payload = torch.load(path, weights_only=True, map_location="cpu")
    assert type(payload) is dict
    assert set(payload) == {
        "schema", "config", "config_sha256", "state_dict", "epoch", "updates",
        "majority", "probability",
    }
    assert payload["schema"] == "cipher-experiments-checkpoint-v1"
    assert payload["config"] == config.to_dict()
    assert payload["config_sha256"] == config.sha256()
    assert payload["epoch"] == 3 and payload["updates"] == 7
    assert type(payload["majority"]) is int and payload["majority"] == 1
    assert type(payload["probability"]) is float and payload["probability"] == 0.75
    assert set(payload["state_dict"]) == set(model.state_dict())
    assert all(type(value) is Tensor and value.device.type == "cpu"
               for value in payload["state_dict"].values())
    for name, value in model.state_dict().items():
        assert torch.equal(payload["state_dict"][name], value.detach().cpu())

    loaded, saved_config, majority, probability = checkpoints.load_checkpoint(
        path, file_sha256(path), device=torch.device("cpu"))
    assert saved_config == config
    assert majority == 1 and probability == 0.75
    example = torch.zeros((2, 64), dtype=torch.float32)
    assert torch.equal(loaded(example), model(example))
    assert not loaded.training

    # Existing imports continue to resolve to the exact implementation/constant.
    assert evaluation.CHECKPOINT_SCHEMA is checkpoints.CHECKPOINT_SCHEMA
    assert evaluation.load_checkpoint is checkpoints.load_checkpoint
    assert evaluation.evaluate_model.__module__ == "cipher_experiments.evaluation"


def test_loader_uses_weights_only_and_cpu_mapping(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _synthetic_config()
    path = tmp_path / "weights-only.pt"
    _write_payload(path, _valid_payload(config))
    original_load = torch.load
    observed: list[tuple[bool, object]] = []

    def checked_load(*args: Any, **kwargs: Any) -> Any:
        observed.append((kwargs.get("weights_only") is True, kwargs.get("map_location")))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    checkpoints.load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))

    assert observed == [(True, "cpu")]


def test_old_raw_config_mapping_keeps_its_recorded_sha(tmp_path: Path) -> None:
    config = _synthetic_config(name="legacy-raw-mapping")
    raw_config = config.to_dict()
    for field in ("model_kind", "model_features", "prediction_mode", "label_mode",
                  "null_seed", "train_pool_rows", "dev_every_epochs"):
        raw_config.pop(field)
    raw_sha = hashlib.sha256(json.dumps(raw_config, sort_keys=True,
                                        allow_nan=False).encode()).hexdigest()
    payload = _valid_payload(config)
    payload["config"] = raw_config
    payload["config_sha256"] = raw_sha
    path = tmp_path / "legacy.pt"
    sha = _write_payload(path, payload)

    model, loaded_config, _, _ = evaluation.load_checkpoint(
        path, sha, device=torch.device("cpu"))

    assert loaded_config.model_kind == "mlp"
    assert loaded_config.model_features == "full"
    assert loaded_config.prediction_mode == "direct"
    assert model.checkpoint_config_sha256 == raw_sha


@pytest.mark.parametrize(
    ("container", "message"),
    [("payload", "checkpoint schema fields mismatch"),
     ("config", "checkpoint schema mismatch"),
     ("state_dict", "checkpoint state names mismatch")],
)
def test_checkpoint_requires_builtin_dict_containers(
        tmp_path: Path, container: str, message: str) -> None:
    payload = _valid_payload(_synthetic_config())
    if container == "payload":
        payload_to_save: object = OrderedDict(payload)
    else:
        if container == "config":
            payload["config"] = OrderedDict(payload["config"])
        else:
            payload["state_dict"] = OrderedDict(payload["state_dict"])
        payload_to_save = payload
    path = tmp_path / f"invalid-container-{container}.pt"
    sha = _write_payload(path, payload_to_save)

    with pytest.raises(ValueError, match=message):
        checkpoints.load_checkpoint(path, sha, device=torch.device("cpu"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("majority", True, "invalid saved TRAIN majority"),
        ("majority", 2, "invalid saved TRAIN majority"),
        ("probability", 1, "invalid saved TRAIN probability"),
        ("probability", float("nan"), "invalid saved TRAIN probability"),
        ("config_sha256", "0" * 64, "configuration SHA mismatch"),
        ("state_dict", [], "state names mismatch"),
    ],
)
def test_checkpoint_rejects_invalid_metadata_types_and_state_container(
        tmp_path: Path, field: str, value: object, message: str) -> None:
    payload = _valid_payload(_synthetic_config())
    payload[field] = value
    path = tmp_path / f"invalid-{field}.pt"
    sha = _write_payload(path, payload)

    with pytest.raises(ValueError, match=message):
        checkpoints.load_checkpoint(path, sha, device=torch.device("cpu"))


def test_checkpoint_tensor_validation_keeps_exact_type_shape_layout_and_finiteness(
        tmp_path: Path) -> None:
    payload = _valid_payload(_synthetic_config())
    state = payload["state_dict"]
    assert type(state) is dict
    key = next(iter(state))
    valid = state[key]
    assert type(valid) is Tensor
    corruptions = {
        "wrong-type": 12,
        "wrong-dtype": valid.double(),
        "wrong-shape": torch.empty((1,), dtype=valid.dtype),
        "wrong-layout": torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=valid.dtype),
            size=valid.shape),
        "nonfinite": torch.full_like(valid, float("nan")),
    }
    for suffix, corrupted in corruptions.items():
        changed = dict(payload)
        changed_state = dict(state)
        changed_state[key] = corrupted
        changed["state_dict"] = changed_state
        path = tmp_path / f"{suffix}.pt"
        sha = _write_payload(path, changed)
        with pytest.raises(ValueError, match=f"checkpoint state mismatch: {key}"):
            checkpoints.load_checkpoint(path, sha, device=torch.device("cpu"))


def test_checkpoint_wiring_validation_and_module_dependency_direction(
        tmp_path: Path) -> None:
    config = ExperimentConfig(name="synthetic-circuit-checkpoint",
                              dataset="circuit-synthetic", model_kind="circuit-aligned",
                              hidden_width=64)
    payload = _valid_payload(config)
    state = payload["state_dict"]
    assert type(state) is dict
    wiring = state["first_gate.right_indices"].clone()
    wiring[0] = state["first_gate.left_indices"][0]
    state["first_gate.right_indices"] = wiring
    path = tmp_path / "invalid-wiring.pt"
    sha = _write_payload(path, payload)
    with pytest.raises(ValueError, match="wire pairs must not connect an input to itself"):
        checkpoints.load_checkpoint(path, sha, device=torch.device("cpu"))

    tree = ast.parse(inspect.getsource(checkpoints))
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "training" not in imports
    assert "evaluation" not in imports
    assert evaluation.load_checkpoint is checkpoints.load_checkpoint

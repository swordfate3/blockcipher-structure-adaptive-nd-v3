"""CNN integration uses null labels only, never real ciphertext targets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from cipher_experiments import training
from cipher_experiments.artifacts import file_sha256
from cipher_experiments.checkpoints import load_checkpoint
from cipher_experiments.cli import main
from cipher_experiments.config import ExperimentConfig
from cipher_experiments.models.prediction import build_predictor


@pytest.mark.parametrize("kind", ["simon-cnn", "simon-rotation-cnn"])
def test_conv_null_cli_and_strict_checkpoint_timing(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_experiments.data import prediction as data

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("engineering null workflow must not encrypt")

    monkeypatch.setattr(data, "_encrypt", forbidden)
    cfg = ExperimentConfig(name="conv-null", dataset="simon", model_kind=kind,
                           hidden_width=8, rounds=6, label_mode="null",
                           train_rows=8, dev_rows=4, test_rows=4,
                           batch_size=4, epochs=2, dev_every_epochs=2)
    out = tmp_path / "run"
    calls: list[str] = []
    old_labels, old_load = training.labels_for, load_checkpoint

    def labels(c: ExperimentConfig, x: torch.Tensor, split: str) -> torch.Tensor:
        if split == "test":
            assert calls[-1] == "loaded"
            assert json.loads((out / "selection.json").read_text(encoding="utf-8"))[
                "test_labels_generated"] is False
        calls.append(split)
        return old_labels(c, x, split)

    def load(*args: Any, **kwargs: Any) -> Any:
        value = old_load(*args, **kwargs)
        calls.append("loaded")
        return value

    monkeypatch.setattr(training, "labels_for", labels)
    monkeypatch.setattr(training, "load_checkpoint", load)
    result = training.train(cfg, out)
    assert calls == ["train", "dev", "loaded", "test"]
    assert result["updates"] == 4 and result["presentations"] == 16
    assert result["selected_epoch"] == 2
    meta = json.loads((out / "model_metadata.json").read_text(encoding="utf-8"))
    assert meta["kind"] == kind and meta["blocks"] == 6
    assert "not six cipher states" in meta["interpretation"]
    checkpoint = out / "best.pt"
    loaded, saved, _, _ = load_checkpoint(checkpoint, file_sha256(checkpoint),
                                          device=torch.device("cpu"))
    assert saved == cfg
    x = torch.zeros((2, 64))
    assert loaded(x).shape == (2, 1)
    monkeypatch.setattr(training, "labels_for", old_labels)
    assert main(["evaluate", "--checkpoint", str(checkpoint), "--sha256",
                 file_sha256(checkpoint), "--output", str(tmp_path / "evaluation")]) == 0
    evaluated = json.loads((tmp_path / "evaluation/evaluation.json").read_text(encoding="utf-8"))
    assert evaluated["metrics"] == result["metrics"]["test"]
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    key = next(k for k in payload["state_dict"] if k.endswith("gather_indices"))
    # A valid range/permutation is still not canonical adjacency and must be rejected.
    payload["state_dict"][key] = payload["state_dict"][key].roll(1)
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError):
        load_checkpoint(bad, file_sha256(bad), device=torch.device("cpu"))


@pytest.mark.parametrize("kind", ["simon-cnn", "simon-rotation-cnn"])
@pytest.mark.parametrize("patch", [
    {"dataset": "synthetic"}, {"rounds": 3}, {"rounds": True},
    {"model_features": "support"}, {"prediction_mode": "public-coordinate"},
    {"hidden_width": 7}, {"hidden_width": 257}, {"hidden_width": True},
    {"experiment": "innovation1", "dataset": "differential"},
])
def test_conv_config_rejects_incompatible_combinations(kind: str, patch: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(**{"name": "bad", "dataset": "simon", "model_kind": kind,
                            "rounds": 6, "hidden_width": 64, **patch})


@pytest.mark.parametrize("kind", ["simon-cnn", "simon-rotation-cnn"])
@pytest.mark.parametrize(("features", "mode"), [
    ("support", "direct"), ("full", "public-residual"), ("full", "public-coordinate"),
])
def test_conv_factory_does_not_silently_ignore_options(kind: str, features: str, mode: str) -> None:
    with pytest.raises(ValueError, match="full/direct"):
        build_predictor(model_kind=kind, rounds=6, feature_mode=features, prediction_mode=mode)

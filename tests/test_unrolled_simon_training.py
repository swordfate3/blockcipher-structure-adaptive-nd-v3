"""No-trajectory integration and strict state/TEST boundary regressions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from cipher_experiments import training
from cipher_experiments.artifacts import file_sha256
from cipher_experiments.cli import main
from cipher_experiments.config import ExperimentConfig, load_config
from cipher_experiments.evaluation import load_checkpoint
from cipher_experiments.models.prediction import build_predictor
from cipher_experiments.models.unrolled_simon import UnrolledSimonPredictor


@pytest.mark.parametrize(("prediction_mode", "rounds"), [
    ("direct", 3), ("plaintext-xor", 2), ("plaintext-xor-no-final-low", 2),
])
def test_unrolled_final_only_checkpoint_and_test_boundary(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        prediction_mode: str, rounds: int) -> None:
    config = ExperimentConfig(name="unrolled-null", dataset="simon", model_kind="simon-unrolled",
                              label_mode="null", hidden_width=8, rounds=rounds,
                              prediction_mode=prediction_mode,
                              train_rows=8, dev_rows=4, test_rows=4, batch_size=4, epochs=1)
    output = tmp_path / "run"
    calls: list[str] = []
    original_labels = training.labels_for
    original_load = load_checkpoint

    def labels(cfg: ExperimentConfig, x: torch.Tensor, split: str) -> torch.Tensor:
        if split == "test":
            assert calls[-1] == "loaded"
            selection = json.loads((output / "selection.json").read_text())
            assert selection["test_labels_generated"] is False
        calls.append(split)
        return original_labels(cfg, x, split)

    def load(*args: Any, **kwargs: Any) -> Any:
        result = original_load(*args, **kwargs)
        calls.append("loaded")
        return result

    from cipher_experiments.data import prediction as prediction_data

    def forbidden_encrypt(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("null workflow must never invoke actual encryption")

    monkeypatch.setattr(prediction_data, "_encrypt", forbidden_encrypt)
    monkeypatch.setattr(training, "labels_for", labels)
    monkeypatch.setattr(training, "load_checkpoint", load)
    result = training.train(config, output)
    assert calls == ["train", "dev", "loaded", "test"]
    assert result["updates"] == 2 and result["presentations"] == 8
    assert "hard_constants" not in result["metrics"]["test"]
    metadata = json.loads((output / "model_metadata.json").read_text())
    assert "final-only" in metadata["supervision"]
    assert "no real intermediate states" in metadata["supervision"]
    assert "no true key" in metadata["input"]
    assert metadata["final_low_input"] == (
        "zero after centering for every position in the final round only"
        if prediction_mode == "plaintext-xor-no-final-low"
        else "centered local low-state value in every round")
    if prediction_mode in ("plaintext-xor", "plaintext-xor-no-final-low"):
        assert metadata["prediction_mode"] == prediction_mode
        assert "high half" in metadata["public_reference"]
        assert "target_bit" in metadata["public_reference"]
    else:
        assert metadata["public_reference"] == "none (direct mode)"
    checkpoint = output / "best.pt"
    model, saved, _, _ = load_checkpoint(checkpoint, file_sha256(checkpoint),
                                         device=torch.device("cpu"))
    assert isinstance(model, UnrolledSimonPredictor)
    assert saved == config
    monkeypatch.setattr(training, "labels_for", original_labels)
    assert main(["evaluate", "--checkpoint", str(checkpoint), "--sha256", file_sha256(checkpoint),
                 "--output", str(tmp_path / "evaluation")]) == 0
    evaluated = json.loads((tmp_path / "evaluation/evaluation.json").read_text())
    assert evaluated["metrics"] == result["metrics"]["test"]
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    key = next(iter(payload["state_dict"]))
    payload["state_dict"][key] = payload["state_dict"][key].double()
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="state mismatch"):
        load_checkpoint(bad, file_sha256(bad), device=torch.device("cpu"))


@pytest.mark.parametrize("patch", [
    {"dataset": "synthetic"}, {"rounds": 9}, {"rounds": True},
    {"rounds": 1, "target_bit": 32}, {"model_features": "support"},
    {"prediction_mode": "public-residual"},
    {"experiment": "innovation1", "dataset": "differential"},
    {"model_kind": "mlp"}, {"rounds": 3},
    {"rounds": 2, "target_bit": 32}, {"rounds": 2, "model_features": "support"},
])
def test_unrolled_config_rejects_changed_contract(patch: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(**{"name": "bad", "dataset": "simon", "model_kind": "simon-unrolled",
                            "rounds": 2, "target_bit": 0,
                            "prediction_mode": "plaintext-xor", **patch})


@pytest.mark.parametrize("feature,mode", [("support", "direct"),
                                          ("full", "public-residual")])
def test_factory_rejects_ignored_options(feature: str, mode: str) -> None:
    with pytest.raises(ValueError, match="full/direct"):
        build_predictor(model_kind="simon-unrolled", feature_mode=feature, prediction_mode=mode)


def test_factory_rejects_plaintext_xor_for_other_models() -> None:
    with pytest.raises(ValueError, match="model_kind=simon-unrolled"):
        build_predictor(model_kind="mlp", prediction_mode="plaintext-xor")


def test_plaintext_xor_config_strict_json_round_trip(tmp_path: Path) -> None:
    config = ExperimentConfig(name="plaintext-xor", dataset="simon", model_kind="simon-unrolled",
                              prediction_mode="plaintext-xor", rounds=2, target_bit=31)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert load_config(path) == config
    invalid = config.to_dict()
    invalid["rounds"] = 3
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="plaintext-xor"):
        load_config(path)


def test_legacy_unrolled_direct_checkpoint_loads_without_mode_field(tmp_path: Path) -> None:
    import hashlib

    config = ExperimentConfig(name="old-unrolled", dataset="simon", model_kind="simon-unrolled",
                              hidden_width=8, rounds=3, target_bit=0)
    raw = config.to_dict()
    raw.pop("prediction_mode")
    raw.pop("dev_every_epochs")
    sha = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    model = build_predictor(config.hidden_width, model_kind=config.model_kind,
                            rounds=config.rounds, target_bit=config.target_bit)
    path = tmp_path / "legacy-unrolled.pt"
    torch.save({"schema": "cipher-experiments-checkpoint-v1", "config": raw,
                "config_sha256": sha, "state_dict": dict(model.state_dict()),
                "epoch": 1, "updates": 2, "majority": 0, "probability": .5}, path)

    loaded, saved, _, _ = load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))
    assert saved.prediction_mode == "direct"
    assert torch.equal(loaded(torch.zeros((2, 64))), model(torch.zeros((2, 64))))


def test_unrolled_configs_freeze_bounded_cpu_trial() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs/innovation2-unrolled").glob("*.json"))
    assert len(paths) == 6
    configs = [load_config(path) for path in paths]
    assert {c.rounds for c in configs} == {3, 4}
    assert all(c.device == "cpu" and c.epochs == 50 and c.train_rows == 4096
               and c.dev_rows == 1024 and c.test_rows == 8192 and c.batch_size == 256
               and c.hidden_width == 32 and c.data_seed == 192601
               and c.init_seed == 17 and c.key_seed == 190502
               and c.learning_rate == .001 and c.max_seconds == 120 for c in configs)
    for rounds in (3, 4):
        subset = [c for c in configs if c.rounds == rounds]
        assert {(c.model_kind, c.label_mode) for c in subset} == {
            ("simon-unrolled", "cipher"), ("simon-unrolled", "null"), ("mlp", "cipher")}
    smoke = load_config(root / "configs/innovation2-unrolled-smoke.json")
    assert smoke.label_mode == "null" and smoke.model_kind == "simon-unrolled"

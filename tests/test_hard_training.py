"""Hard-fitting CLI and freeze-before-TEST boundary; small fixtures only."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from cipher_experiments import hard_training
from cipher_experiments.artifacts import file_sha256
from cipher_experiments.cli import main
from cipher_experiments.config import ExperimentConfig, load_config
from cipher_experiments.evaluation import load_checkpoint
from cipher_experiments.models.structured_simon import HardSimonPredictor
from cipher_experiments.training import input_panels, labels_for


def fixture_config() -> ExperimentConfig:
    return ExperimentConfig(name="hard-null-functional", dataset="simon", rounds=1,
                            model_kind="simon-hard-search", hidden_width=64, label_mode="null",
                            epochs=1, train_rows=1, dev_rows=4, test_rows=4, batch_size=4)


def test_hard_cli_checkpoint_and_explicit_evaluate(tmp_path: Path) -> None:
    cfg = fixture_config()
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg.to_dict()))
    assert main(["check", "--config", str(cfg_path)]) == 0
    out = tmp_path / "run"
    assert main(["train", "--config", str(cfg_path), "--output", str(out)]) == 0
    result = json.loads((out / "results.json").read_text())
    assert result["gradient_updates"] == 0
    assert result["metrics"]["train"]["accuracy"] == 1.
    sha = file_sha256(out / "best.pt")
    model, _, _, _ = load_checkpoint(out / "best.pt", sha, device=torch.device("cpu"))
    assert isinstance(model, HardSimonPredictor)
    assert not any(p.requires_grad for p in model.parameters())
    assert main(["evaluate", "--checkpoint", str(out / "best.pt"), "--sha256", sha,
                 "--output", str(tmp_path / "eval")]) == 0
    evaluation = json.loads((tmp_path / "eval/evaluation.json").read_text())
    assert evaluation["metrics"] == result["metrics"]["test"]
    payload = torch.load(out / "best.pt", weights_only=True, map_location="cpu")
    payload["state_dict"]["key_logits"].fill_(.5)
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="hard assignment"):
        load_checkpoint(bad, file_sha256(bad), device=torch.device("cpu"))
    assert main(["train", "--config", str(cfg_path), "--output", str(out)]) == 1


def test_test_labels_after_strict_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_labels = labels_for
    original_load = load_checkpoint
    loaded = False

    def load(*args: Any, **kwargs: Any) -> Any:
        nonlocal loaded
        result = original_load(*args, **kwargs)
        loaded = True
        return result

    monkeypatch.setattr(hard_training, "load_checkpoint", load)
    out = tmp_path / "run"

    def check_labels(config: ExperimentConfig, x: torch.Tensor, split: str) -> torch.Tensor:
        if split == "test":
            assert loaded
            assert json.loads((out / "selection.json").read_text())[
                "test_labels_generated"] is False
        return original_labels(config, x, split)

    monkeypatch.setattr(hard_training, "labels_for", check_labels)
    hard_training.train(fixture_config(), out)
    assert loaded


def test_exhaustion_does_not_generate_test_labels(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_panels = input_panels
    original_labels = labels_for

    def inputs(config: ExperimentConfig) -> dict[str, torch.Tensor]:
        panels = original_panels(config)
        panels["train"][1] = panels["train"][0]  # explicit inconsistent toy fixture
        return panels

    def labels(config: ExperimentConfig, x: torch.Tensor, split: str) -> torch.Tensor:
        assert split != "test"
        if split == "train":
            return torch.tensor([[0.], [1.]])
        return original_labels(config, x, split)

    monkeypatch.setattr(hard_training, "input_panels", inputs)
    monkeypatch.setattr(hard_training, "labels_for", labels)
    out = tmp_path / "exhausted"
    result = hard_training.train(replace(fixture_config(), train_rows=2), out)
    assert result["status"] == "exhausted" and result["prediction_metrics"] is None
    assert not (out / "best.pt").exists() and not (out / "selection.json").exists()
    assert "y_sha256" not in json.loads((out / "data.json").read_text())["test"]


def test_hard_timeout_saves_failed_not_exhausted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: Any, **kwargs: Any) -> Any:
        raise TimeoutError("fixture cooperative timeout")

    monkeypatch.setattr(hard_training, "fit_hard_simon", timeout)
    out = tmp_path / "timeout"
    with pytest.raises(TimeoutError):
        hard_training.train(fixture_config(), out)
    exit_info = json.loads((out / "exit.json").read_text())
    assert exit_info["status"] == "failed" and exit_info["exit_code"] == 1
    assert not (out / "results.json").exists() and not (out / "best.pt").exists()
    assert "y_sha256" not in json.loads((out / "data.json").read_text())["test"]


@pytest.mark.parametrize("patch", [{"device": "cuda:0"}, {"epochs": 2}, {"rounds": 5},
                                    {"dataset": "synthetic"}, {"hidden_width": 32}])
def test_hard_config_contract(patch: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(fixture_config(), **patch)


def test_hard_assignment_validation_and_no_adam_entry(tmp_path: Path) -> None:
    model = HardSimonPredictor(rounds=1)
    for bad in [-1, 2, True]:
        with pytest.raises(ValueError):
            model.set_assignment(bad)
    from cipher_experiments.training import train
    with pytest.raises(ValueError, match="not Adam"):
        train(fixture_config(), tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


def test_hard_configs_parse() -> None:
    paths = Path(__file__).resolve().parents[1] / "configs/innovation2-hard"
    configs = [load_config(p) for p in paths.glob("*.json")]
    assert len(configs) == 4
    assert all(c.model_kind == "simon-hard-search" and c.device == "cpu" for c in configs)

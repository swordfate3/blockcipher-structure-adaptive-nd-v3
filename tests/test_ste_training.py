"""STE uses ordinary Adam training and strict prediction checkpoints, not fitting APIs."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from cipher_experiments import training
from cipher_experiments.artifacts import file_sha256
from cipher_experiments.cli import main
from cipher_experiments.config import ExperimentConfig, load_config
from cipher_experiments.evaluation import load_checkpoint
from cipher_experiments.models.ste_simon import STESimonPredictor
from cipher_experiments.training import labels_for


def config() -> ExperimentConfig:
    return ExperimentConfig(name="ste-null-fixture", dataset="simon", rounds=1,
                            model_kind="simon-ste", hidden_width=64, label_mode="null",
                            train_rows=8, dev_rows=4, test_rows=4, batch_size=4, epochs=1)


def test_ste_cli_checkpoint_and_freeze_before_test(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "run"
    loaded = False

    def load(*args: Any, **kwargs: Any) -> Any:
        nonlocal loaded
        result = load_checkpoint(*args, **kwargs)
        loaded = True
        return result

    def labels(cfg: ExperimentConfig, x: torch.Tensor, split: str) -> torch.Tensor:
        if split == "test":
            assert loaded
            assert json.loads((out / "selection.json").read_text())[
                "test_labels_generated"] is False
        return labels_for(cfg, x, split)

    monkeypatch.setattr(training, "load_checkpoint", load)
    monkeypatch.setattr(training, "labels_for", labels)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(config().to_dict()))
    assert main(["check", "--config", str(path)]) == 0
    assert main(["train", "--config", str(path), "--output", str(out)]) == 0
    result = json.loads((out / "results.json").read_text())
    assert result["updates"] == 2 and result["presentations"] == 8
    assert result["model_kind"] == "simon-ste"
    sha = file_sha256(out / "best.pt")
    model, _, _, _ = load_checkpoint(out / "best.pt", sha, device=torch.device("cpu"))
    assert isinstance(model, STESimonPredictor)
    assert all(p.requires_grad for p in model.parameters())
    metadata = json.loads((out / "model_metadata.json").read_text())
    assert metadata["parameter_search"] is False
    assert metadata["selection"] == "hard-forward absolute DEV BCE"
    assert main(["evaluate", "--checkpoint", str(out / "best.pt"), "--sha256", sha,
                 "--output", str(tmp_path / "eval")]) == 0
    evaluation = json.loads((tmp_path / "eval/evaluation.json").read_text())
    assert evaluation["metrics"] == result["metrics"]["test"]
    payload = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    payload["state_dict"]["key_logits"].fill_(float("nan"))
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="state mismatch"):
        load_checkpoint(bad, file_sha256(bad), device=torch.device("cpu"))


@pytest.mark.parametrize("patch", [{"dataset": "synthetic"}, {"hidden_width": 32},
                                    {"prediction_mode": "public-residual"}, {"rounds": 17}])
def test_ste_contract(patch: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(config(), **patch)


def test_ste_configs_parse() -> None:
    root = Path(__file__).resolve().parents[1] / "configs/innovation2-ste"
    configs = [load_config(p) for p in root.glob("*.json")]
    assert len(configs) == 5
    assert all(c.model_kind == "simon-ste" and c.device == "cpu" for c in configs)

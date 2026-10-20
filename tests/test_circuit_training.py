"""Small integration fixtures; not the 300-step diagnostic comparison."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cipher_experiments.artifacts import file_sha256
from cipher_experiments.cli import main
from cipher_experiments.config import ExperimentConfig, load_config
from cipher_experiments.evaluation import load_checkpoint
from cipher_experiments.models.circuit import CircuitPredictor
from cipher_experiments.training import train


@pytest.mark.parametrize("arm", ["aligned", "random"])
def test_toy_cli_checkpoint_and_wiring_rejection(tmp_path: Path, arm: str) -> None:
    config = ExperimentConfig(name="toy-fixture", dataset="circuit-synthetic",
                              model_kind="circuit-" + arm, hidden_width=64,
                              train_rows=8, dev_rows=4, test_rows=4, epochs=1, batch_size=4)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    assert main(["check", "--config", str(path)]) == 0
    out = tmp_path / "run"
    result = train(config, out)
    assert result["updates"] == 2 and result["presentations"] == 8
    model, _, _, _ = load_checkpoint(out / "best.pt", file_sha256(out / "best.pt"),
                                     device=torch.device("cpu"))
    assert isinstance(model, CircuitPredictor)
    assert main(["evaluate", "--checkpoint", str(out / "best.pt"),
                 "--sha256", file_sha256(out / "best.pt"),
                 "--output", str(tmp_path / "evaluation")]) == 0
    payload = torch.load(out / "best.pt", weights_only=True, map_location="cpu")
    # Values in range but duplicate coverage are invalid even under an exact file SHA.
    payload["state_dict"]["first_gate.left_indices"][0] = (
        payload["state_dict"]["first_gate.left_indices"][1])
    malformed = tmp_path / "invalid.pt"
    torch.save(payload, malformed)
    with pytest.raises(ValueError, match="cover|wire|input"):
        load_checkpoint(malformed, file_sha256(malformed), device=torch.device("cpu"))
    if arm == "aligned":
        payload = torch.load(out / "best.pt", weights_only=True, map_location="cpu")
        wire = payload["state_dict"]["first_gate.right_indices"]
        wire[0], wire[1] = wire[1].clone(), wire[0].clone()
        valid_but_wrong = tmp_path / "wrong-aligned.pt"
        torch.save(payload, valid_but_wrong)
        with pytest.raises(ValueError, match="noncanonical"):
            load_checkpoint(valid_but_wrong, file_sha256(valid_but_wrong),
                            device=torch.device("cpu"))


def test_diagnostic_configs_freeze_same_training_conditions() -> None:
    root = Path(__file__).resolve().parents[1] / "configs/structure-diagnostic"
    left, right = (load_config(root / (name + ".json")).to_dict()
                   for name in ("aligned", "random"))
    for field in ("name", "model_kind"):
        left.pop(field)
        right.pop(field)
    assert left == right
    assert left["dataset"] == "circuit-synthetic" and left["epochs"] == 300
    assert left["train_rows"] + left["dev_rows"] + left["test_rows"] == 256


@pytest.mark.parametrize("dataset,mode", [("simon", "direct"),
                                         ("circuit-synthetic", "public-residual")])
def test_toy_model_cannot_be_silently_used_for_cipher(dataset: str, mode: str) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(name="bad", dataset=dataset, model_kind="circuit-aligned",
                         prediction_mode=mode, hidden_width=64)

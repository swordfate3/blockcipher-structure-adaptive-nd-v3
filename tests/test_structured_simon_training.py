"""Structured predictor integration; independent null labels, not cipher fitting."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from cipher_experiments.artifacts import file_sha256
from cipher_experiments.cli import main
from cipher_experiments.config import ExperimentConfig, load_config
from cipher_experiments.evaluation import load_checkpoint
from cipher_experiments.models.structured_simon import StructuredSimonPredictor
from cipher_experiments.training import train


def test_structure_null_cli_and_checkpoint_no_secret_constructor(tmp_path: Path) -> None:
    config = ExperimentConfig(name="structure-null-fixture", dataset="simon", rounds=3,
                              model_kind="simon-structure", label_mode="null", hidden_width=64,
                              train_rows=8, dev_rows=4, test_rows=4, batch_size=4, epochs=1)
    out = tmp_path / "run"
    result = train(config, out)
    assert result["updates"] == 2 and result["presentations"] == 8
    assert "hard_constants" in result["metrics"]["dev"]
    assert "hard_constants" in result["metrics"]["test"]
    checkpoint = out / "best.pt"
    loaded, _, _, _ = load_checkpoint(checkpoint, file_sha256(checkpoint),
                                      device=torch.device("cpu"))
    assert isinstance(loaded, StructuredSimonPredictor)
    assert sum(p.numel() for p in loaded.parameters()) == 11
    metadata = json.loads((out / "model_metadata.json").read_text())
    assert metadata["key_schedule_constraint"] is False
    assert len(metadata["symbolic_coordinates"]) == 11
    assert main(["evaluate", "--checkpoint", str(checkpoint), "--sha256", file_sha256(checkpoint),
                 "--output", str(tmp_path / "evaluation")]) == 0
    evaluation = json.loads((tmp_path / "evaluation/evaluation.json").read_text())
    assert evaluation["metrics"] == result["metrics"]["test"]
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    key = next(iter(payload["state_dict"]))
    payload["state_dict"][key] = payload["state_dict"][key].double()
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="state mismatch"):
        load_checkpoint(bad, file_sha256(bad), device=torch.device("cpu"))


@pytest.mark.parametrize("patch", [
    {"dataset": "synthetic"}, {"rounds": 7}, {"rounds": True},
    {"rounds": 1, "target_bit": 32}, {"prediction_mode": "public-residual"},
    {"model_features": "support"}, {"hidden_width": 32},
])
def test_structured_config_rejects_ambiguous_contract(patch: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(**{"name": "bad", "dataset": "simon", "rounds": 3,
                            "hidden_width": 64, "model_kind": "simon-structure", **patch})


def test_learn_one_unknown_xor_constant_from_synthetic_circuit_samples() -> None:
    # One-node unknown-effect circuit fixture, no cipher/key API or key schedule.
    # Deliberately start the unknown parameter against the fixed toy constant.
    model = StructuredSimonPredictor(rounds=1, target_bit=0)
    with torch.no_grad():
        model.key_logits.fill_(-2.0)
    x = torch.zeros((16, 64))
    bits = ((torch.arange(16)[:, None] >> torch.arange(3, -1, -1)) & 1).float()
    x[:, [32, 1, 8, 2]] = bits
    public = bits[:, 0].bool() ^ (bits[:, 1].bool() & bits[:, 2].bool()) ^ bits[:, 3].bool()
    y = (~public).float().unsqueeze(1)  # fixed synthetic unknown constant=1
    optimizer = torch.optim.Adam(model.parameters(), lr=.2)
    initial = float(torch.nn.functional.binary_cross_entropy_with_logits(model(x), y).detach())
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    final = float(torch.nn.functional.binary_cross_entropy_with_logits(model(x), y).detach())
    assert final < initial * .2
    assert torch.equal(model.hard_logits(x) > 0, y.bool())


def test_structured_smoke_config_parses() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/innovation2-structure-smoke.json"
    config = load_config(path)
    assert config.label_mode == "null" and config.device == "cpu"
    assert config.model_kind == "simon-structure"
    proposals = sorted((path.parent / "innovation2-structured-proposal").glob("*.json"))
    assert len(proposals) == 3
    configs = [load_config(p) for p in proposals]
    assert {c.rounds for c in configs} == {1, 3, 4}
    assert all(c.data_seed == 191701 and c.label_mode == "cipher" and c.device == "cuda:0"
               for c in configs)

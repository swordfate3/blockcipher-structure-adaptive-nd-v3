"""Bounded real shared-M plumbing, not formal experimental quality."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from cipher_experiments.artifacts import file_sha256
from cipher_experiments.cli import main
from cipher_experiments.config import ExperimentConfig
from cipher_experiments.innovation1 import evaluate_checkpoint, train


def test_shared_m_four_task_training_checkpoint_and_evaluation(tmp_path: Path) -> None:
    config = ExperimentConfig(name="shared-m-tiny", experiment="innovation1",
                              dataset="differential",
                              train_rows=2, dev_rows=2, test_rows=2, batch_size=2, epochs=1,
                              max_seconds=120)
    result = train(config, tmp_path / "run")
    assert result["updates"] == 4
    assert result["presentations"] == 8
    assert set(result["updates_by_task"].values()) == {1}
    assert len(result["metrics"]["test"]) == 4
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["parameters"] == 1087509
    checkpoint = tmp_path / "run/best.pt"
    evaluate_checkpoint(checkpoint, file_sha256(checkpoint), "cpu", tmp_path / "eval")
    # Additive prediction defaults must not invalidate old shared-M identities.
    old = torch.load(checkpoint, weights_only=True, map_location="cpu")
    for field in ("model_features", "prediction_mode", "label_mode", "null_seed"):
        old["config"].pop(field)
    old["config_sha256"] = hashlib.sha256(json.dumps(
        old["config"], sort_keys=True).encode()).hexdigest()
    old_path = tmp_path / "old.pt"
    torch.save(old, old_path)
    evaluate_checkpoint(old_path, file_sha256(old_path), "cpu", tmp_path / "old-eval")
    # Invalid but SHA-bound states must fail before data generation/forward.
    for mutation in ("dtype", "nan"):
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        key = next(n for n, v in saved["state_dict"].items() if v.is_floating_point())
        value = saved["state_dict"][key]
        if mutation == "dtype":
            saved["state_dict"][key] = value.to(torch.float64)
        else:
            value.flatten()[0] = float("nan")
        malformed = tmp_path / (mutation + ".pt")
        torch.save(saved, malformed)
        with pytest.raises(ValueError, match="tensor mismatch"):
            evaluate_checkpoint(malformed, file_sha256(malformed), "cpu", tmp_path / mutation)
    exit_info = json.loads((tmp_path / "run/exit.json").read_text())
    assert exit_info["exit_code"] == 0 and exit_info["updates"] == 4


def test_shared_m_cli_check(tmp_path: Path) -> None:
    config = ExperimentConfig(name="config", experiment="innovation1", dataset="differential",
                              train_rows=2, dev_rows=2, test_rows=2, batch_size=2, epochs=1)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    assert main(["check", "--config", str(path)]) == 0

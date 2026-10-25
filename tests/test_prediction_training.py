"""Innovation2 loop and heldout order with arbitrary caller-provided labels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from cipher_experiments import training
from cipher_experiments.artifacts import file_sha256
from cipher_experiments.cli import main
from cipher_experiments.config import ExperimentConfig
from cipher_experiments.evaluation import evaluate_model, load_checkpoint


@pytest.mark.parametrize("label_mode", ["cipher", "null"])
def test_matched_direct_residual_loop_freezes_before_test(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label_mode: str) -> None:
    # Exercise exact training/evaluation/checkpoint flow, but never encrypt or
    # fit an actual cipher. Seeds/key fixture are not used to create labels.
    values: dict[str, torch.Tensor] = {}
    for split, n in (("train", 8), ("dev", 4), ("test", 4)):
        generator = torch.Generator().manual_seed(n + len(split))
        values[split] = torch.randint(0, 2, (n, 64), generator=generator).float()
    monkeypatch.setattr(training, "input_panels", lambda _: values)
    def labels(config: ExperimentConfig, x: torch.Tensor, split: str) -> torch.Tensor:
        if split == "test":
            root = tmp_path / config.prediction_mode
            assert (root / "selection.json").is_file()
            assert (root / "best.pt").is_file()
        return x[:, :1].clone()  # arbitrary caller labels, not a cipher label
    monkeypatch.setattr(training, "labels_for", labels)
    results = []
    for mode in ("direct", "public-residual"):
        config = ExperimentConfig(name="caller-labels", dataset="simon", rounds=4,
                                  train_rows=8, dev_rows=4, test_rows=4, epochs=1,
                                  batch_size=4, hidden_width=8, model_features="support",
                                  prediction_mode=mode, label_mode=label_mode)
        out = tmp_path / mode
        result = training.train(config, out)
        assert result["updates"] == 2
        assert "public_reference_constant" in result["metrics"]["test"]
        checkpoint = out / "best.pt"
        model, saved, _, _ = load_checkpoint(checkpoint, file_sha256(checkpoint),
                                             device=torch.device("cpu"))
        assert saved.prediction_mode == mode
        assert model(values["test"]).shape == (4, 1)
        results.append(result)
    assert results[0]["initial_state_sha256"] == results[1]["initial_state_sha256"]


def test_old_default_checkpoint_still_loads_after_new_config_defaults(tmp_path: Path) -> None:
    from cipher_experiments.models.mlp import BitMLP
    config = ExperimentConfig(name="old")
    raw = config.to_dict()
    for key in ("experiment", "model_kind", "model_features", "prediction_mode", "label_mode",
                "null_seed", "train_pool_rows", "dev_every_epochs"):
        raw.pop(key)
    import hashlib
    sha = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    path = tmp_path / "old.pt"
    torch.save({"schema": "cipher-experiments-checkpoint-v1", "config": raw,
                "config_sha256": sha, "state_dict": dict(BitMLP(config.hidden_width).state_dict()),
                "epoch": 1, "updates": 2, "majority": 0, "probability": .5}, path)
    _, saved, _, _ = load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))
    assert saved.model_features == "full" and saved.prediction_mode == "direct"
    output = tmp_path / "old-evaluation"
    assert main(["evaluate", "--checkpoint", str(path), "--sha256", file_sha256(path),
                 "--output", str(output)]) == 0
    result = json.loads((output / "evaluation.json").read_text())
    assert result["config_sha256"] == sha
    assert result["resolved_config_sha256"] == saved.sha256() != sha


def test_dev_cadence_skips_epochs_but_evaluates_intervals_and_selects_only_them(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
        ) -> None:
    inputs = {
        split: torch.randint(0, 2, (4, 64), generator=torch.Generator().manual_seed(seed)).float()
        for split, seed in (("train", 11), ("dev", 12), ("test", 13))
    }
    monkeypatch.setattr(training, "input_panels", lambda _: inputs)
    original_labels_for = training.labels_for
    label_calls: list[str] = []

    def null_labels(config: ExperimentConfig, x: torch.Tensor, split: str) -> torch.Tensor:
        if split == "test":
            assert (tmp_path / "sparse-dev/selection.json").is_file()
        label_calls.append(split)
        return original_labels_for(config, x, split)

    monkeypatch.setattr(training, "labels_for", null_labels)
    original_evaluate = evaluate_model
    evaluated_splits: list[str] = []
    training_dev_scores = iter((0.1, 0.2))

    def track_evaluation(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                         **kwargs: Any) -> dict[str, Any]:
        split = next(name for name, panel in inputs.items() if x is panel)
        evaluated_splits.append(split)
        metrics = original_evaluate(model, x, y, **kwargs)
        if split == "dev" and len(evaluated_splits) <= 2:
            metrics["bce"] = next(training_dev_scores)
        return metrics

    monkeypatch.setattr(training, "evaluate_model", track_evaluation)
    config = ExperimentConfig(
        name="sparse-dev-cadence", dataset="simon", label_mode="null", train_rows=4,
        dev_rows=4, test_rows=4, hidden_width=4, batch_size=2, epochs=4,
        dev_every_epochs=2, max_seconds=120)
    output = tmp_path / "sparse-dev"
    result = training.train(config, output)

    progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    assert [(record["epoch"], record["dev_evaluated"], record["dev"] is not None)
            for record in progress] == [
                (1, False, False), (2, True, True), (3, False, False), (4, True, True)]
    assert progress[1]["dev"]["bce"] == 0.1
    assert progress[3]["dev"]["bce"] == 0.2
    assert evaluated_splits == ["dev", "dev", "train", "dev", "test"]
    assert evaluated_splits.count("dev") == 3  # epochs 2/4 plus final selected-model metrics
    assert label_calls == ["train", "dev", "test"]
    assert result["selected_epoch"] == 2
    best = torch.load(output / "best.pt", weights_only=True, map_location="cpu")
    last = torch.load(output / "last.pt", weights_only=True, map_location="cpu")
    assert best["epoch"] == 2 and last["epoch"] == 4
    assert result["updates"] == 8
    output_text = capsys.readouterr().out
    assert "dev=not evaluated (no BCE)" in output_text
    assert "epoch=1/4" in output_text and "epoch=3/4" in output_text


@pytest.mark.parametrize("patch", [
    {"prediction_mode": "public-residual", "dataset": "synthetic"},
    {"prediction_mode": "public-residual", "rounds": 5},
    {"model_features": "support", "dataset": "synthetic"},
    {"label_mode": "null", "dataset": "synthetic"},
    {"label_mode": "null", "experiment": "innovation1", "dataset": "differential"},
    {"dev_every_epochs": True},
    {"dev_every_epochs": 0},
    {"dev_every_epochs": 3},
])
def test_invalid_mode_combinations(patch: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(**{"name": "bad", "dataset": "simon", **patch})


def test_dev_interval_defaults_and_innovation1_constraint() -> None:
    assert ExperimentConfig(name="one-epoch", epochs=1).dev_every_epochs == 1
    with pytest.raises(ValueError, match="dev_every_epochs=1"):
        ExperimentConfig(name="innovation1-interval", experiment="innovation1",
                         dataset="differential", train_rows=2, dev_rows=2, test_rows=2,
                         batch_size=2, epochs=4, dev_every_epochs=2)


def test_example_innovation2_configs_check_without_training(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_experiments.config import load_config
    root = Path(__file__).resolve().parents[1]
    # GPU example recipes are parsed on CPU; CLI's real device gate is not bypassed in use.
    monkeypatch.setattr(training, "device_for", lambda _: torch.device("cpu"))
    paths = sorted((root / "configs/innovation2").glob("*.json"))
    assert len(paths) == 5
    for path in paths:
        assert load_config(path).dataset == "simon"
        assert main(["check", "--config", str(path)]) == 0

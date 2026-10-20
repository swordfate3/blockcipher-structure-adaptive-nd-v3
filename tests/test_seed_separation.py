"""Batch-order seed separation, config-SHA compatibility and run determinism."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from cipher_experiments.artifacts import file_sha256
from cipher_experiments.config import ExperimentConfig, load_config
from cipher_experiments.evaluation import evaluate_model
from cipher_experiments.models.prediction import build_predictor
from cipher_experiments.training import input_panels, labels_for, train


def _without_timing(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        record.pop("elapsed_seconds", None)
    return records


def config(**patch: Any) -> ExperimentConfig:
    base = ExperimentConfig(name="seed-fixture", dataset="simon", rounds=1, hidden_width=64,
                            model_kind="simon-ste", label_mode="null", train_rows=16,
                            dev_rows=8, test_rows=8, batch_size=4, epochs=1)
    return replace(base, **patch)


def test_absent_shuffle_seed_keeps_the_historical_config_sha() -> None:
    legacy_mapping = config().to_dict()
    assert "shuffle_seed" not in legacy_mapping
    explicit = config(shuffle_seed=7)
    assert explicit.to_dict()["shuffle_seed"] == 7
    assert explicit.sha256() != config().sha256()
    # Filling the additive default must reproduce the same saved identity.
    assert ExperimentConfig(**legacy_mapping).sha256() == config().sha256()


def test_shuffle_seed_changes_order_but_not_the_plaintext_panels(tmp_path: Path) -> None:
    smooth = config(shuffle_seed=100, model_kind="mlp", learning_rate=0.05)
    panels_a = input_panels(smooth)
    panels_b = input_panels(config(shuffle_seed=200, model_kind="mlp", learning_rate=0.05))
    for split in ("train", "dev", "test"):
        assert panels_a[split].numpy().tobytes() == panels_b[split].numpy().tobytes()

    runs = {}
    for name, seed in (("a", 100), ("b", 200), ("a-again", 100)):
        order = config(shuffle_seed=seed, model_kind="mlp", learning_rate=0.05)
        runs[name] = train(order, tmp_path / name)
    # Same order seed reproduces the run exactly; a different one changes the path.
    assert runs["a"]["metrics"] == runs["a-again"]["metrics"]
    assert file_sha256(tmp_path / "a/best.pt") == file_sha256(tmp_path / "a-again/best.pt")
    assert runs["a"]["metrics"]["dev"]["bce"] != runs["b"]["metrics"]["dev"]["bce"]


def test_same_config_is_deterministic(tmp_path: Path) -> None:
    first = train(config(shuffle_seed=5), tmp_path / "one")
    second = train(config(shuffle_seed=5), tmp_path / "two")
    assert first["metrics"] == second["metrics"]
    assert first["initial_state_sha256"] == second["initial_state_sha256"]
    assert file_sha256(tmp_path / "one/best.pt") == file_sha256(tmp_path / "two/best.pt")
    assert _without_timing(tmp_path / "one/progress.jsonl") == _without_timing(
        tmp_path / "two/progress.jsonl")


def test_invalid_shuffle_seed_is_rejected() -> None:
    for bad in (-1, 2**32, True, 1.5, "7"):
        with pytest.raises(ValueError):
            config(shuffle_seed=bad)


def test_bce_gain_is_reported_against_each_baseline() -> None:
    cfg = config(shuffle_seed=11)
    x = input_panels(cfg)["train"]
    y = labels_for(cfg, x, "train")
    model = build_predictor(64, model_kind="simon-ste", rounds=1)
    metrics = evaluate_model(model, x, y, device=__import__("torch").device("cpu"),
                             batch_size=8, majority=0, probability=0.5)
    assert metrics["bce_gain_vs_train_constant"] == pytest.approx(
        metrics["train_constant"]["bce"] - metrics["bce"])
    assert "bce_gain_vs_public_reference" not in metrics


def test_order_seed_is_recorded_for_ste(tmp_path: Path) -> None:
    out = tmp_path / "meta"
    train(config(shuffle_seed=42), out)
    metadata = json.loads((out / "model_metadata.json").read_text())
    assert metadata["order_seed"] == 42
    assert metadata["score_floor_bce"] == pytest.approx(0.1269280110429725, abs=1e-12)


def test_repo_configs_still_load_with_the_new_field() -> None:
    paths = sorted((Path(__file__).resolve().parents[1] / "configs").rglob("*.json"))
    assert paths
    for path in paths:
        load_config(path)

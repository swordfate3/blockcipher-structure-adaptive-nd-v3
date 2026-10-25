"""Fixed-train-pool data and config compatibility checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from cipher_experiments.artifacts import file_sha256
from cipher_experiments.cli import main
from cipher_experiments.config import ExperimentConfig
from cipher_experiments.data.prediction import make_inputs, make_labels
from cipher_experiments.evaluation import load_checkpoint
from cipher_experiments.training import train


def _sha256(tensor: Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _null_labels(x: Tensor, split: str) -> Tensor:
    return make_labels(x, dataset="simon", rounds=6, target_bit=0, key_seed=19,
                       label_mode="null", null_seed=901, split=split)


def test_legacy_inputs_keep_exact_bytes_and_pool_panels_are_prefix_stable(
        monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = make_inputs(7, 4, 3, seed=17)
    assert {name: _sha256(x) for name, x in legacy.items()} == {
        "train": "d0ad791ff0db478550d2188faa73cf4443160df4304ed1f6a71de81761a6f536",
        "dev": "2ad26af922e82291e16f2a0f07645f0ed71e310b2e8ed4f18ffe1d0d143f1a8f",
        "test": "0aa64f36f3011d35d22a80a8b250b5cd12eced62fa1e2ff5a710993d12321051",
    }

    def unexpected_encrypt(*args: object, **kwargs: object) -> int:
        raise AssertionError("pool generation must not create cipher labels")

    from cipher_experiments.data import prediction
    with monkeypatch.context() as patch:
        patch.setattr(prediction, "_encrypt", unexpected_encrypt)
        small = make_inputs(8, 16, 12, seed=772, train_pool_rows=64)
        large = make_inputs(32, 16, 12, seed=772, train_pool_rows=64)
    assert torch.equal(small["train"], large["train"][:8])
    for split in ("dev", "test"):
        assert torch.equal(small[split], large[split])
        assert _sha256(small[split]) == _sha256(large[split])
    for split in ("train", "dev", "test"):
        small_y = make_labels(small[split], dataset="simon", rounds=6, target_bit=0,
                              key_seed=19, split=split)
        large_y = make_labels(large[split], dataset="simon", rounds=6, target_bit=0,
                              key_seed=19, split=split)
        assert torch.equal(small_y, large_y[:len(small_y)])
    all_rows = torch.cat((large["train"], large["dev"], large["test"]))
    assert torch.unique(all_rows, dim=0).shape[0] == len(all_rows)

    for split in ("train", "dev", "test"):
        assert torch.equal(_null_labels(small[split], split),
                           _null_labels(large[split], split)[:len(small[split])])
    assert torch.equal(_null_labels(small["train"], "train"),
                       _null_labels(large["train"], "train")[:8])


def test_train_pool_config_boundaries_and_scope() -> None:
    base: dict[str, Any] = {"name": "pool", "dataset": "simon", "train_rows": 8,
                            "dev_rows": 4, "test_rows": 4}
    assert ExperimentConfig(**base).train_pool_rows is None
    assert ExperimentConfig(**base, train_pool_rows=8).train_pool_rows == 8
    assert ExperimentConfig(**base, train_pool_rows=1_000_000).train_pool_rows == 1_000_000
    for invalid in (True, 7, 1_000_001):
        with pytest.raises(ValueError):
            ExperimentConfig(**base, train_pool_rows=invalid)
    with pytest.raises(ValueError, match="2,000,000"):
        ExperimentConfig(**{**base, "dev_rows": 1_000_000, "test_rows": 1},
                         train_pool_rows=1_000_000)
    with pytest.raises(ValueError, match="requires experiment=prediction"):
        ExperimentConfig(**base, experiment="innovation1", train_pool_rows=8)
    with pytest.raises(ValueError, match="requires experiment=prediction"):
        ExperimentConfig(name="not-simon", train_pool_rows=8)


def test_train_pool_type_and_make_inputs_bounds() -> None:
    for invalid in (True, 7, 1_000_001):
        with pytest.raises((TypeError, ValueError)):
            make_inputs(8, 4, 4, seed=1, train_pool_rows=invalid)
    with pytest.raises(ValueError, match="combined split counts"):
        make_inputs(8, 999_996, 999_996, seed=1, train_pool_rows=1_000_000)


def test_null_train_pool_cli_flow_records_config_and_evaluates(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ExperimentConfig(name="pool-null-fixture", dataset="simon",
                              model_kind="simon-unrolled", train_rows=8, train_pool_rows=16,
                              dev_rows=4, test_rows=4,
                              rounds=6, target_bit=0, hidden_width=8, label_mode="null",
                              batch_size=4, epochs=1, max_seconds=120)
    output = tmp_path / "run"
    from cipher_experiments import training
    original_labels_for = training.labels_for
    label_calls: list[tuple[str, int]] = []

    def count_label_rows(cfg: ExperimentConfig, x: Tensor, split: str) -> Tensor:
        label_calls.append((split, len(x)))
        return original_labels_for(cfg, x, split)

    monkeypatch.setattr(training, "labels_for", count_label_rows)
    result = train(config, output)
    assert result["updates"] == 2
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    saved_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert manifest["train_pool_rows"] == saved_config["train_pool_rows"] == 16
    assert manifest["label_mode"] == "null"
    assert label_calls == [("train", 8), ("dev", 4), ("test", 4)]
    model, loaded_config, _, _ = load_checkpoint(
        output / "best.pt", file_sha256(output / "best.pt"), device=torch.device("cpu"))
    assert loaded_config.train_pool_rows == 16
    panels = make_inputs(loaded_config.train_rows, loaded_config.dev_rows,
                         loaded_config.test_rows, seed=loaded_config.data_seed,
                         train_pool_rows=loaded_config.train_pool_rows)
    assert model(panels["test"]).shape == (4, 1)
    assert json.loads((output / "selection.json").read_text())["test_labels_generated"] is False
    evaluation = tmp_path / "evaluation"
    assert main(["evaluate", "--checkpoint", str(output / "best.pt"), "--sha256",
                 file_sha256(output / "best.pt"), "--output", str(evaluation)]) == 0
    result_path = evaluation / "evaluation.json"
    evaluation_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert evaluation_result["split"] == "test" and evaluation_result["metrics"]["rows"] == 4


def test_legacy_raw_checkpoint_without_pool_field_still_loads(
        tmp_path: Path) -> None:
    from cipher_experiments.models.mlp import BitMLP

    config = ExperimentConfig(name="old")
    raw = config.to_dict()
    for field in ("experiment", "model_kind", "model_features", "prediction_mode", "label_mode",
                  "null_seed", "train_pool_rows", "dev_every_epochs"):
        raw.pop(field)
    raw_sha = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    path = tmp_path / "legacy.pt"
    torch.save({"schema": "cipher-experiments-checkpoint-v1", "config": raw,
                "config_sha256": raw_sha,
                "state_dict": dict(BitMLP(config.hidden_width).state_dict()),
                "epoch": 1, "updates": 2, "majority": 0, "probability": .5}, path)
    _, loaded, _, _ = load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))
    assert loaded.train_pool_rows is None

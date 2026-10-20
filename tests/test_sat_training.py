"""SAT fitting integration and no-model semantics."""
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
from cipher_experiments.training import labels_for


def config() -> ExperimentConfig:
    return ExperimentConfig(name="sat-toy", dataset="simon", rounds=1, hidden_width=64,
                            model_kind="simon-hard-sat", epochs=1, label_mode="null",
                            train_rows=1, dev_rows=4, test_rows=4)


def test_sat_cli_roundtrip_and_test_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pysat")
    out = tmp_path / "run"
    loaded = False

    def load(*args: Any, **kwargs: Any) -> Any:
        nonlocal loaded
        result = load_checkpoint(*args, **kwargs)
        loaded = True
        return result

    def labels(cfg: ExperimentConfig, x: torch.Tensor, split: str) -> torch.Tensor:
        if split == "test":
            assert loaded and (out / "selection.json").exists()
        return labels_for(cfg, x, split)

    monkeypatch.setattr(hard_training, "load_checkpoint", load)
    monkeypatch.setattr(hard_training, "labels_for", labels)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(config().to_dict()))
    assert main(["check", "--config", str(path)]) == 0
    assert main(["train", "--config", str(path), "--output", str(out)]) == 0
    result = json.loads((out / "results.json").read_text())
    assert result["metrics"]["train"]["accuracy"] == 1.
    assert result["gradient_updates"] == 0
    sha = file_sha256(out / "best.pt")
    assert main(["evaluate", "--checkpoint", str(out / "best.pt"), "--sha256", sha,
                 "--output", str(tmp_path / "eval")]) == 0
    evaluation = json.loads((tmp_path / "eval/evaluation.json").read_text())
    assert evaluation["metrics"] == result["metrics"]["test"]


@pytest.mark.parametrize("status", ["unknown", "unsat"])
def test_sat_without_model_never_generates_test(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    from dataclasses import make_dataclass

    from cipher_experiments.fitting import sat_simon
    # Keep actual fitting dataclass out of this contract fixture.
    result_type = make_dataclass("Fit", [("assignment", int | None), ("status", str)])

    def fit(*args: Any, **kwargs: Any) -> Any:
        return result_type(None, status)

    def labels(cfg: ExperimentConfig, x: torch.Tensor, split: str) -> torch.Tensor:
        assert split != "test"
        return labels_for(cfg, x, split)

    monkeypatch.setattr(sat_simon, "fit_sat_simon", fit)
    monkeypatch.setattr(hard_training, "labels_for", labels)
    out = tmp_path / status
    result = hard_training.train(config(), out)
    assert result["status"] == status and result["prediction_metrics"] is None
    assert not (out / "best.pt").exists()


@pytest.mark.parametrize("patch", [{"train_rows": 513}, {"rounds": 7}, {"epochs": 2},
                                    {"device": "cuda:0"}])
def test_sat_config_bounds(patch: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(config(), **patch)


def test_cli_sat_missing_dependency_is_explicit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_experiments.fitting import sat_simon

    def missing() -> None:
        raise RuntimeError("SAT backend unavailable; install the optional sat extra")

    monkeypatch.setattr(sat_simon, "check_sat_backend", missing)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(config().to_dict()))
    assert main(["check", "--config", str(path)]) == 1
    assert "optional sat" in capsys.readouterr().err


def test_sat_configs_parse() -> None:
    root = Path(__file__).resolve().parents[1] / "configs/innovation2-sat"
    configs = [load_config(p) for p in root.glob("*.json")]
    assert len(configs) == 4
    assert all(c.model_kind == "simon-hard-sat" for c in configs)

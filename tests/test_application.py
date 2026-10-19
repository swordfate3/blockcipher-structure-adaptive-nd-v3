"""Self-contained startup acceptance; no archived outputs or remote access."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cipher_experiments.artifacts import file_sha256
from cipher_experiments.cli import main
from cipher_experiments.config import ExperimentConfig, load_config
from cipher_experiments.evaluation import load_checkpoint
from cipher_experiments.training import train


def tiny_config() -> ExperimentConfig:
    return ExperimentConfig(name="unit", train_rows=8, dev_rows=4, test_rows=4,
                            epochs=1, batch_size=4, hidden_width=8)


def test_real_train_checkpoint_evaluation_and_cli(tmp_path: Path) -> None:
    config = tiny_config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    assert main(["check", "--config", str(path)]) == 0
    output = tmp_path / "run"
    assert main(["train", "--config", str(path), "--output", str(output)]) == 0
    result = json.loads((output / "results.json").read_text())
    assert result["updates"] == 2 and result["presentations"] == 8
    assert result["metrics"]["test"]["rows"] == 4
    assert json.loads((output / "exit.json").read_text())["exit_code"] == 0
    required = {"manifest.json", "config.json", "data.json", "progress.jsonl", "training.log",
                "best.pt", "last.pt", "results.json", "exit.json"}
    assert required <= {p.name for p in output.iterdir()}
    checkpoint = output / "best.pt"
    assert main(["evaluate", "--checkpoint", str(checkpoint), "--sha256", file_sha256(checkpoint),
                 "--output", str(tmp_path / "eval")]) == 0
    assert main(["train", "--config", str(path), "--output", str(output)]) == 1
    with pytest.raises(ValueError, match="SHA256"):
        load_checkpoint(checkpoint, "0" * 64, device=torch.device("cpu"))


@pytest.mark.parametrize("patch", [{"epochs": True}, {"learning_rate": float("nan")},
                                   {"rounds": 45}, {"target_bit": 64}, {"device": "auto"}])
def test_strict_config_rejects_invalid_values(patch: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(**{**tiny_config().to_dict(), **patch})


def test_config_unknown_and_duplicate_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"name":"x","epochs":1,"epochs":2}')
    with pytest.raises(ValueError, match="duplicate"):
        load_config(path)
    path.write_text('{"name":"x","unknown":1}')
    assert main(["check", "--config", str(path)]) == 1


def test_failure_writes_exit_and_traceback(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cipher_experiments.training as training
    def fail(_: ExperimentConfig) -> None:
        raise RuntimeError("synthetic data fixture failure")
    monkeypatch.setattr(training, "input_panels", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="fixture"):
        train(tiny_config(), output)
    exit_info = json.loads((output / "exit.json").read_text())
    assert exit_info["status"] == "failed" and exit_info["updates"] == 0
    assert (output / "error.log").is_file()


def test_cli_help_never_trains() -> None:
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0

"""Logic candidate integration with synthetic or independent null labels only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cipher_experiments.artifacts import file_sha256
from cipher_experiments.cli import main
from cipher_experiments.config import ExperimentConfig
from cipher_experiments.evaluation import load_checkpoint
from cipher_experiments.training import train


@pytest.mark.parametrize("mode", ["direct", "public-residual"])
def test_logic_null_loop_checkpoint_and_explicit_evaluate(tmp_path: Path, mode: str) -> None:
    config = ExperimentConfig(name="logic-null-fixture", model_kind="logic16", dataset="simon",
                              label_mode="null", model_features="support", prediction_mode=mode,
                              rounds=4, hidden_width=64, train_rows=8, dev_rows=4, test_rows=4,
                              batch_size=4, epochs=1, max_seconds=60)
    out = tmp_path / "run"
    result = train(config, out)
    assert result["updates"] == 2 and result["presentations"] == 8
    assert result["model_kind"] == "logic16"
    path = out / "best.pt"
    loaded, saved, _, _ = load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))
    assert saved.model_kind == "logic16"
    assert loaded(torch.zeros(2, 64)).shape == (2, 1)
    assert main(["evaluate", "--checkpoint", str(path), "--sha256", file_sha256(path),
                 "--output", str(tmp_path / "evaluation")]) == 0
    # A valid file SHA alone cannot authorize invalid integer topology.
    payload = torch.load(path, weights_only=True, map_location="cpu")
    for value in payload["state_dict"].values():
        if value.dtype == torch.int64:
            value.fill_(10000)
            break
    else:
        raise AssertionError("logic checkpoint must contain persistent integer wiring")
    bad = tmp_path / "bad-wiring.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="wir|rang|index|topolog"):
        load_checkpoint(bad, file_sha256(bad), device=torch.device("cpu"))
    assert json.loads((out / "exit.json").read_text())["exit_code"] == 0


def test_logic_config_mode_rejections() -> None:
    with pytest.raises(ValueError, match="model_kind"):
        ExperimentConfig(name="invalid", model_kind="not-a-model")
    with pytest.raises(ValueError, match="model_kind"):
        ExperimentConfig(name="invalid", experiment="innovation1", dataset="differential",
                         model_kind="logic16")
    with pytest.raises(ValueError, match="hidden_width"):
        ExperimentConfig(name="invalid", model_kind="logic16", hidden_width=32)


def test_logic_proposal_configs_parse_and_match_without_gpu_work() -> None:
    from cipher_experiments.config import load_config
    root = Path(__file__).resolve().parents[1] / "configs/innovation2-logic-proposal"
    configs = {p.stem: load_config(p) for p in root.glob("*.json")}
    assert len(configs) == 4
    left, right = configs["r4-mlp-direct"].to_dict(), configs["r4-logic-direct"].to_dict()
    for field in ("name", "model_kind"):
        left.pop(field)
        right.pop(field)
    assert left == right
    assert left["data_seed"] == 191501  # distinct from already-observed first batch
    assert all(c.device == "cuda:0" for c in configs.values())

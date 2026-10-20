"""Fixed train-panel capacity so sample-efficiency sweeps share DEV/TEST rows."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from cipher_experiments.config import ExperimentConfig, load_config
from cipher_experiments.training import input_panels

ROOT = Path(__file__).resolve().parents[1]


def config(**patch: Any) -> ExperimentConfig:
    base = ExperimentConfig(name="pool-fixture", dataset="simon", rounds=4, hidden_width=64,
                            model_kind="simon-hard-search", epochs=1, train_rows=64,
                            dev_rows=32, test_rows=32, data_seed=192301)
    return replace(base, **patch)


def test_train_pool_keeps_dev_and_test_fixed_while_train_shrinks() -> None:
    small = input_panels(config(train_rows=64, train_pool=512))
    large = input_panels(config(train_rows=512))
    assert len(small["train"]) == 64 and len(large["train"]) == 512
    for split in ("dev", "test"):
        assert small[split].numpy().tobytes() == large[split].numpy().tobytes()
    # The TRAIN budget is a prefix of the shared pool, not a resampled panel.
    assert small["train"].numpy().tobytes() == large["train"][:64].numpy().tobytes()


def test_train_pool_is_not_a_new_identity_when_absent() -> None:
    assert "train_pool" not in config().to_dict()
    assert config(train_pool=512).to_dict()["train_pool"] == 512


@pytest.mark.parametrize("patch", [
    {"train_pool": 32}, {"train_pool": 1_000_001}, {"train_pool": True},
    {"train_pool": 512.0},
    {"train_pool": 1_000_000, "dev_rows": 1_000_000, "test_rows": 1},
])
def test_train_pool_bounds(patch: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        config(**patch)


def test_sweep_configs_share_one_panel() -> None:
    hard = sorted((ROOT / "configs/innovation2-sample-efficiency-hard").glob("*.json"))
    ste = sorted((ROOT / "configs/innovation2-sample-efficiency-ste").glob("*.json"))
    assert len(hard) == 19 and len(ste) == 7
    configs = [load_config(p) for p in hard + ste]
    assert {c.train_pool for c in configs} == {262144}
    assert {c.data_seed for c in configs} == {192301}
    # Three extra keys re-run the identification threshold at k=32/512.
    assert {c.key_seed for c in configs} == {190502, 190503, 190504, 190505}
    assert {c.rounds for c in configs} == {4} and {c.target_bit for c in configs} == {0}
    assert {(c.dev_rows, c.test_rows) for c in configs} == {(16384, 65536)}
    # The primary k-sweep has one distinct budget per step; the extra key-robustness
    # configs intentionally reuse k=32/512.
    sweep = [p for p in hard if not p.name.startswith("key")] + ste
    for arm in ([p for p in hard if not p.name.startswith("key")], ste):
        sizes = [load_config(p).train_rows for p in arm]
        assert len(set(sizes)) == len(arm)
    assert {load_config(p).train_rows for p in hard if p.name.startswith("key")} == {32, 512}
    assert len(sweep) == 13 + 7


def test_hard_sweep_runs_end_to_end_on_smallest_budget(tmp_path: Path) -> None:
    from cipher_experiments.cli import main
    path = tmp_path / "k64.json"
    path.write_text(json.dumps(config(train_rows=64, train_pool=512).to_dict()))
    out = tmp_path / "run"
    assert main(["train", "--config", str(path), "--output", str(out)]) == 0
    result = json.loads((out / "results.json").read_text())
    assert result["status"] == "completed" and result["gradient_updates"] == 0
    assert result["metrics"]["test"]["rows"] == 32

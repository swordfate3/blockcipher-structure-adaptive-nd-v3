"""Frozen small confirmation panel; no scientific fitting in config tests."""
from pathlib import Path

from cipher_experiments.config import load_config


def test_ste_confirmation_changes_only_predeclared_factors() -> None:
    root = Path(__file__).resolve().parents[1] / "configs/innovation2-ste-confirm"
    configs = {p.stem: load_config(p) for p in root.glob("*.json")}
    assert set(configs) == {"r4-panel", "r4-init18", "r4-init19", "r4-newkey", "r5", "r5-null"}
    base = configs["r4-panel"]
    assert base.data_seed == 192101 and base.init_seed == 17 and base.key_seed == 190502
    assert base.model_kind == "simon-ste" and base.device == "cpu" and base.rounds == 4
    assert (base.train_rows, base.dev_rows, base.test_rows) == (4096, 1024, 8192)
    assert base.epochs == 50 and base.batch_size == 256 and base.max_seconds == 120
    changes: dict[str, dict[str, int | str]] = {
        "r4-init18": {"init_seed": 18}, "r4-init19": {"init_seed": 19},
        "r4-newkey": {"key_seed": 190503}, "r5": {"rounds": 5},
        "r5-null": {"rounds": 5, "label_mode": "null"},
    }
    for name, patch in changes.items():
        assert configs[name].to_dict() == {**base.to_dict(), "name": configs[name].name, **patch}

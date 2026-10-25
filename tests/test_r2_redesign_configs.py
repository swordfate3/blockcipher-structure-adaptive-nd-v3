"""Freeze the four-arm two-round CPU diagnostic, not a cipher fit."""
from pathlib import Path

from cipher_experiments.config import load_config


def test_r2_redesign_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs/innovation2-r2-redesign").glob("*.json"))
    assert len(paths) == 4
    configs = [load_config(p) for p in paths]
    assert {(c.model_kind, c.label_mode) for c in configs} == {
        ("simon-r2-table", "cipher"), ("simon-r2-table", "null"),
        ("simon-r2-local", "cipher"), ("simon-r2-local", "null"),
    }
    for c in configs:
        assert c.dataset == "simon" and c.experiment == "prediction"
        assert c.rounds == 2 and c.target_bit == 0 and c.hidden_width == 16
        assert c.model_features == "full" and c.prediction_mode == "direct"
        assert (c.train_rows, c.dev_rows, c.test_rows) == (4096, 1024, 8192)
        assert (c.data_seed, c.null_seed, c.init_seed, c.key_seed) == (
            193101, 193102, 17, 190502)
        assert c.device == "cpu" and c.batch_size == 256 and c.epochs == 50
        assert c.learning_rate == .01 and c.max_seconds == 120
    for kind in ("table", "local"):
        c = load_config(root / "configs/innovation2-r2-redesign-smoke" / f"{kind}.json")
        assert c.model_kind == "simon-r2-" + kind and c.label_mode == "null"
        assert (c.train_rows, c.dev_rows, c.test_rows) == (8, 4, 4)
        assert c.batch_size == 4 and c.epochs == 1 and c.device == "cpu"
        assert (c.data_seed, c.null_seed) == (193111, 193112)

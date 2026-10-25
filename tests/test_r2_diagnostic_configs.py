"""Fixed two-init diagnostics budget, no scientific fits."""
from pathlib import Path

from cipher_experiments.config import load_config


def test_fixed_diagnostic_configs() -> None:
    base = Path(__file__).resolve().parents[1] / "configs"
    paths = list((base / "innovation2-r2-diagnostic").glob("*.json"))
    assert len(paths) == 4
    for path in paths:
        c = load_config(path)
        assert c.model_kind == "simon-r2-local" and c.hidden_width == 16
        assert c.rounds == 2 and c.target_bit == 0 and c.device == "cpu"
        assert c.data_seed == 193201 and c.null_seed == 193202 and c.key_seed == 190502
        assert c.init_seed in (17, 18)
        assert c.label_mode == ("null" if "null" in path.name else "cipher")
        assert (c.train_rows, c.dev_rows, c.test_rows) == (4096, 1024, 8192)
        assert c.epochs == 50 and c.batch_size == 256 and c.learning_rate == .01
        assert c.max_seconds == 120 and c.prediction_mode == "direct"
    smoke = load_config(base / "innovation2-r2-diagnostic-smoke.json")
    assert smoke.label_mode == "null" and smoke.train_rows == 8
    assert smoke.epochs == 1 and smoke.batch_size == 4

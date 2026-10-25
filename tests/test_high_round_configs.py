"""Fixed 6/8-round bit prediction batch budgets, no scientific fitting."""
from pathlib import Path

from cipher_experiments.config import load_config


def test_high_round_batch_configs() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    files = list((root / "innovation2-high-round").glob("*.json"))
    assert len(files) == 4
    for path in files:
        c = load_config(path)
        assert c.model_kind == "simon-unrolled" and c.hidden_width == 32
        assert c.rounds in (6, 8) and c.target_bit == 0
        assert c.label_mode == ("null" if path.stem.endswith("null") else "cipher")
        assert c.dataset == "simon" and c.experiment == "prediction"
        assert c.model_features == "full" and c.prediction_mode == "direct"
        assert c.data_seed == 193401 and c.null_seed == 193402
        assert c.init_seed == 17 and c.key_seed == 190502
        assert (c.train_rows, c.dev_rows, c.test_rows) == (4096, 1024, 8192)
        assert c.batch_size == 256 and c.epochs == 50 and c.learning_rate == .001
        assert c.device == "cpu" and c.max_seconds == 120
    smoke = load_config(root / "innovation2-high-round-smoke.json")
    assert smoke.rounds == 8 and smoke.label_mode == "null"
    assert smoke.train_rows == 8 and smoke.batch_size == 4 and smoke.epochs == 1

"""Bound the single-path ablation trial; no fitting in this test."""
from pathlib import Path

from cipher_experiments.config import load_config


def test_final_low_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs/innovation2-final-low").glob("*.json"))
    assert len(paths) == 4
    configs = [load_config(p) for p in paths]
    assert {(c.prediction_mode, c.label_mode) for c in configs} == {
        ("plaintext-xor", "cipher"), ("plaintext-xor", "null"),
        ("plaintext-xor-no-final-low", "cipher"), ("plaintext-xor-no-final-low", "null"),
    }
    for c in configs:
        assert c.experiment == "prediction" and c.dataset == "simon"
        assert c.model_kind == "simon-unrolled" and c.model_features == "full"
        assert c.rounds == 2 and c.target_bit == 0 and c.hidden_width == 32
        assert (c.train_rows, c.dev_rows, c.test_rows) == (4096, 1024, 8192)
        assert (c.data_seed, c.key_seed, c.init_seed, c.null_seed) == (
            193001, 190502, 17, 193002)
        assert c.batch_size == 256 and c.epochs == 50 and c.device == "cpu"
        assert c.max_seconds == 120 and c.learning_rate == .001
    smoke = load_config(root / "configs/innovation2-final-low-smoke.json")
    assert smoke.prediction_mode == "plaintext-xor-no-final-low" and smoke.label_mode == "null"
    assert (smoke.train_rows, smoke.dev_rows, smoke.test_rows) == (8, 4, 4)
    assert smoke.batch_size == 4 and smoke.epochs == 1
    assert smoke.device == "cpu" and smoke.data_seed == 193011 and smoke.null_seed == 193012

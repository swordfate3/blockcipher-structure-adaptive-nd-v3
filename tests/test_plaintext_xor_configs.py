"""Freeze the authorized four-arm, two-round CPU comparison."""
from pathlib import Path

import pytest

from cipher_experiments.config import load_config


@pytest.mark.parametrize(("directory", "data_seed", "null_seed"), [
    ("innovation2-plaintext-xor", 192801, 192802),
    ("innovation2-plaintext-xor-panel", 192901, 192902),
])
def test_plaintext_xor_comparison_configs(
        directory: str, data_seed: int, null_seed: int) -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs" / directory).glob("*.json"))
    assert len(paths) == 4
    configs = [load_config(path) for path in paths]
    assert {(c.prediction_mode, c.label_mode) for c in configs} == {
        ("direct", "cipher"), ("direct", "null"),
        ("plaintext-xor", "cipher"), ("plaintext-xor", "null")}
    for c in configs:
        assert c.dataset == "simon" and c.experiment == "prediction"
        assert c.model_kind == "simon-unrolled" and c.model_features == "full"
        assert c.rounds == 2 and c.target_bit == 0 and c.hidden_width == 32
        assert (c.train_rows, c.dev_rows, c.test_rows) == (4096, 1024, 8192)
        assert (c.data_seed, c.key_seed, c.init_seed, c.null_seed) == (
            data_seed, 190502, 17, null_seed)
        assert c.device == "cpu" and c.epochs == 50 and c.batch_size == 256
        assert c.learning_rate == .001 and c.max_seconds == 120
    smoke = load_config(root / "configs/innovation2-plaintext-xor-smoke.json")
    assert smoke.prediction_mode == "plaintext-xor" and smoke.label_mode == "null"
    assert smoke.train_rows == 8 and smoke.batch_size == 4 and smoke.epochs == 1

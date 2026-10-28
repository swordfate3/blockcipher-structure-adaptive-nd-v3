"""Frozen CNN adjacency comparison proposal, not authorization to train."""
from pathlib import Path

from cipher_experiments.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_conv_proposal_changes_only_adjacency_and_label_mode() -> None:
    paths = sorted((ROOT / "configs/innovation2-conv-structure").glob("*.json"))
    assert len(paths) == 4
    configs = [load_config(p) for p in paths]
    assert {(c.model_kind, c.label_mode) for c in configs} == {
        (kind, mode) for kind in ("simon-cnn", "simon-rotation-cnn") for mode in ("cipher", "null")
    }
    assert all(c.dataset == "simon" and c.rounds == 6 and c.target_bit == 0
               and c.hidden_width == 64 and c.model_features == "full"
               and c.prediction_mode == "direct" and c.train_rows == 65536
               and c.dev_rows == 8192 and c.test_rows == 65536 and c.train_pool_rows == 65536
               and c.epochs == 32 and c.batch_size == 256 and c.dev_every_epochs == 4
               and c.learning_rate == .001 and c.device == "cuda:0" and c.max_seconds == 3600
               and c.data_seed == 194201 and c.null_seed == 194202 and c.key_seed == 190502
               and c.init_seed == 17 for c in configs)
    signatures = []
    for c in configs:
        raw = c.to_dict()
        for k in ("name", "model_kind", "label_mode"):
            raw.pop(k)
        signatures.append(raw)
    assert all(s == signatures[0] for s in signatures)
    assert sum(c.epochs * c.train_rows // c.batch_size for c in configs) == 32768
    assert sum(c.epochs * c.train_rows for c in configs) == 8388608
    assert sum(c.train_rows + c.dev_rows + c.test_rows
               for c in configs if c.label_mode == "cipher") == 278528


def test_conv_smokes_are_tiny_cpu_null_not_scientific_trials() -> None:
    paths = sorted((ROOT / "configs/innovation2-conv-structure-smoke").glob("*.json"))
    assert len(paths) == 2
    for p in paths:
        c = load_config(p)
        assert c.device == "cpu" and c.hidden_width == 8 and c.label_mode == "null"
        assert c.rounds == 6 and c.train_rows == 8 and c.dev_rows == 4 and c.test_rows == 4
        assert c.epochs == 2 and c.batch_size == 4 and c.dev_every_epochs == 2
        assert c.epochs * c.train_rows // c.batch_size == 4

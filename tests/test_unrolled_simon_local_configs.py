"""Frozen shallow-round diagnostic, not a new fit in the test suite."""
from pathlib import Path

from cipher_experiments.config import load_config


def test_shallow_probe_configs_keep_information_and_budget_fixed() -> None:
    root = Path(__file__).resolve().parents[1] / "configs/innovation2-unrolled-local"
    paths = sorted(root.glob("*.json"))
    assert len(paths) == 4
    configs = [load_config(path) for path in paths]
    assert {(c.rounds, c.label_mode) for c in configs} == {
        (1, "cipher"), (1, "null"), (2, "cipher"), (2, "null")}
    for c in configs:
        assert c.model_kind == "simon-unrolled" and c.dataset == "simon"
        assert c.model_features == "full" and c.prediction_mode == "direct"
        assert c.target_bit == 0 and c.hidden_width == 32 and c.device == "cpu"
        assert (c.train_rows, c.dev_rows, c.test_rows) == (4096, 1024, 8192)
        assert (c.data_seed, c.key_seed, c.init_seed, c.null_seed) == (192701, 190502, 17, 192702)
        assert (c.batch_size, c.epochs, c.learning_rate, c.max_seconds) == (256, 50, .001, 120)
    for r in (1, 2):
        pair = [c.to_dict() for c in configs if c.rounds == r]
        for mapping in pair:
            mapping.pop("name")
            mapping.pop("label_mode")
        assert pair[0] == pair[1]

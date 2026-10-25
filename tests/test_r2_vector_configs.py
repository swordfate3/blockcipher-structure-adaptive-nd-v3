"""Fixed scalar/vector four-arm budgets, no fitting."""
from pathlib import Path

from cipher_experiments.config import load_config


def test_vector_batch_budget_and_pairing() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    paths = list((root / "innovation2-r2-vector").glob("*.json"))
    assert len(paths) == 4
    configs = {p.stem: load_config(p) for p in paths}
    for name, c in configs.items():
        assert c.model_kind == ("simon-r2-vector" if name.startswith("vector")
                                else "simon-r2-local")
        assert c.label_mode == ("null" if name.endswith("null") else "cipher")
        assert c.data_seed == 193301 and c.null_seed == 193302
        assert c.init_seed == 17 and c.key_seed == 190502
        assert (c.train_rows, c.dev_rows, c.test_rows) == (4096, 1024, 8192)
        assert c.batch_size == 256 and c.epochs == 50 and c.learning_rate == .01
        assert c.rounds == 2 and c.target_bit == 0 and c.device == "cpu"
        assert c.hidden_width == 16 and c.model_features == "full"
        assert c.prediction_mode == "direct" and c.max_seconds == 120
    for kind in ("scalar", "vector"):
        smoke = load_config(root / f"innovation2-r2-vector-smoke/{kind}.json")
        assert smoke.label_mode == "null" and smoke.train_rows == 8
        assert smoke.batch_size == 4 and smoke.epochs == 1

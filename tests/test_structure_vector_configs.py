"""Fixed two-arm proposal; parsing these configurations never trains."""
from pathlib import Path

from cipher_experiments.config import load_config


def test_structure_vector_fixed_matrix() -> None:
    root = Path(__file__).resolve().parents[1]
    real = load_config(root / "configs/innovation2-structure-vector/real.json")
    null = load_config(root / "configs/innovation2-structure-vector/null.json")
    assert real.label_mode == "cipher" and null.label_mode == "null"
    a, b = real.to_dict(), null.to_dict()
    for field in ("name", "label_mode"):
        a.pop(field)
        b.pop(field)
    assert a == b
    assert (real.rounds, real.target_bit, real.hidden_width) == (6, 0, 32)
    assert real.model_kind == "simon-vector" and real.device == "cuda:0"
    assert real.train_rows == real.train_pool_rows == 65536
    assert (real.dev_rows, real.test_rows) == (8192, 65536)
    assert real.train_rows // real.batch_size * real.epochs == 8192
    assert real.train_rows * real.epochs == 2097152
    assert real.epochs // real.dev_every_epochs == 8
    assert real.learning_rate == .001 and real.data_seed == 194401
    assert real.key_seed == 190502 and real.init_seed == 17


def test_structure_vector_smoke_is_null_cpu_only() -> None:
    root = Path(__file__).resolve().parents[1]
    c = load_config(root / "configs/innovation2-structure-vector-smoke.json")
    assert c.device == "cpu" and c.label_mode == "null" and c.model_kind == "simon-vector"
    assert (c.train_rows, c.dev_rows, c.test_rows, c.batch_size, c.epochs) == (8, 4, 4, 4, 1)
    assert c.rounds == 6 and c.target_bit == 0 and c.hidden_width == 32

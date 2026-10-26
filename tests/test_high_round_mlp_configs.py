"""Preparation-only contracts for the high-round architecture comparison."""
from pathlib import Path

import torch

from cipher_experiments.config import load_config
from cipher_experiments.models.prediction import build_predictor


def test_high_round_mlp_fixed_config_budget_and_parameter_counts() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    files = list((root / "innovation2-high-round-mlp").glob("*.json"))
    assert len(files) == 4
    for path in files:
        c = load_config(path)
        is_mlp = path.stem.startswith("mlp")
        assert c.model_kind == ("mlp" if is_mlp else "simon-unrolled")
        assert c.hidden_width == (128 if is_mlp else 32)
        assert c.label_mode == ("null" if path.stem.endswith("null") else "cipher")
        assert c.experiment == "prediction" and c.dataset == "simon"
        assert c.rounds == 6 and c.target_bit == 0 and c.model_features == "full"
        assert c.prediction_mode == "direct" and c.device == "cuda:0"
        assert (c.train_rows, c.dev_rows, c.test_rows) == (65536, 8192, 65536)
        assert c.train_pool_rows == 65536
        assert c.batch_size == 256 and c.epochs == 32 and c.dev_every_epochs == 1
        assert c.learning_rate == .001 and c.max_seconds == 1800
        assert (c.data_seed, c.null_seed, c.key_seed, c.init_seed) == (
            193601, 193602, 190502, 17)
        # Local engineering model construction only, no labels or model fitting.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.init_seed)
            model = build_predictor(c.hidden_width, model_kind=c.model_kind,
                                    rounds=c.rounds, target_bit=c.target_bit,
                                    feature_mode=c.model_features,
                                    prediction_mode=c.prediction_mode)
        assert sum(p.numel() for p in model.parameters()) == (24961 if is_mlp else 1089)
    for kind in ("unrolled", "mlp"):
        c = load_config(root / f"innovation2-high-round-mlp-smoke/{kind}.json")
        assert c.label_mode == "null" and c.device == "cpu" and c.rounds == 6
        assert (c.train_rows, c.dev_rows, c.test_rows) == (8, 4, 4)
        assert c.train_pool_rows == 16 and c.batch_size == 4
        assert c.epochs == 2 and c.dev_every_epochs == 2

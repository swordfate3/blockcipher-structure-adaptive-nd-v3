"""Six-round existing unrolled optimization contracts; no formal cipher labels."""
from pathlib import Path

import torch

from cipher_experiments.config import load_config
from cipher_experiments.models.prediction import build_predictor

ROOT = Path(__file__).resolve().parents[1] / "configs"
RATES = {"baseline": .001, "medium": .0003, "small": .0001}


def test_existing_unrolled_six_arm_fixed_budget_and_initial_parameters() -> None:
    paths = sorted((ROOT / "innovation2-unrolled-optimization").glob("*.json"))
    assert len(paths) == 6
    normalized, states = [], []
    for path in paths:
        condition, arm = path.stem.split("-")
        c = load_config(path)
        assert (c.dataset, c.experiment, c.model_kind) == ("simon", "prediction", "simon-unrolled")
        assert (c.rounds, c.target_bit, c.hidden_width) == (6, 0, 32)
        assert (c.model_features, c.prediction_mode) == ("full", "direct")
        assert (c.train_rows, c.dev_rows, c.test_rows, c.train_pool_rows) == (
            65536, 8192, 65536, 65536)
        assert (c.data_seed, c.null_seed, c.key_seed, c.init_seed) == (
            194101, 194102, 190502, 17)
        assert (c.batch_size, c.epochs, c.dev_every_epochs) == (256, 128, 4)
        assert c.learning_rate == RATES[condition]
        assert c.label_mode == ("cipher" if arm == "real" else "null")
        assert c.device == "cuda:0" and c.max_seconds == 1800
        assert c.train_rows // c.batch_size * c.epochs == 32768
        assert list(range(4 * 256, 129 * 256, 4 * 256)) == list(range(1024, 32769, 1024))
        raw = c.to_dict()
        for k in ("name", "learning_rate", "label_mode"):
            raw.pop(k)
        normalized.append(raw)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.init_seed)
            model = build_predictor(c.hidden_width, model_kind=c.model_kind, rounds=c.rounds,
                                    target_bit=c.target_bit, prediction_mode=c.prediction_mode)
        assert sum(p.numel() for p in model.parameters()) == 1089
        states.append(model.state_dict())
    assert all(d == normalized[0] for d in normalized)
    assert all(torch.equal(s[k], states[0][k]) for s in states for k in states[0])
    assert 6 * 32768 == 196608 and 6 * 32768 * 256 == 50331648


def test_null_engineering_configs_use_same_structure_and_small_budget() -> None:
    paths = sorted((ROOT / "innovation2-unrolled-optimization-smoke").glob("*.json"))
    assert len(paths) == 3
    for path in paths:
        c = load_config(path)
        assert c.learning_rate == RATES[path.stem]
        assert c.model_kind == "simon-unrolled" and c.rounds == 6 and c.hidden_width == 32
        assert c.device == "cpu" and c.label_mode == "null"
        assert (c.train_rows, c.dev_rows, c.test_rows, c.train_pool_rows) == (8, 4, 4, 8)
        assert (c.batch_size, c.epochs, c.dev_every_epochs, c.max_seconds) == (4, 4, 1, 60)
        assert c.data_seed == 194111 and c.null_seed == 194112
        assert c.train_rows // c.batch_size * c.epochs == 8

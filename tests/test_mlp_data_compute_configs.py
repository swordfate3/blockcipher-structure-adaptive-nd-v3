"""MLP size/compute proposal checks; no formal cipher fitting."""
from pathlib import Path

import torch

from cipher_experiments.config import load_config
from cipher_experiments.data.prediction import make_inputs, make_labels
from cipher_experiments.models.prediction import build_predictor

ROOT = Path(__file__).resolve().parents[1] / "configs"


def test_six_mlp_data_compute_budgets_and_identical_initial_parameters() -> None:
    paths = sorted((ROOT / "innovation2-mlp-data-compute").glob("*.json"))
    assert len(paths) == 6
    states = []
    updates = presentations = 0
    for path in paths:
        c = load_config(path)
        arm, label = path.stem.split("-")
        assert (c.dataset, c.experiment, c.model_kind) == ("simon", "prediction", "mlp")
        assert c.prediction_mode == "direct" and c.model_features == "full"
        assert c.rounds == 6 and c.target_bit == 0 and c.hidden_width == 128
        assert c.label_mode == ("cipher" if label == "real" else "null")
        assert c.train_rows == (65536 if arm == "a" else 262144)
        assert (c.dev_rows, c.test_rows, c.train_pool_rows) == (8192, 65536, 262144)
        assert (c.data_seed, c.null_seed, c.key_seed, c.init_seed) == (
            193801, 193802, 190502, 17)
        assert c.epochs == (8 if arm == "b" else 32)
        assert c.dev_every_epochs == (1 if arm == "b" else 4)
        assert c.epochs // c.dev_every_epochs == 8
        assert c.batch_size == 256 and c.learning_rate == .001
        assert c.device == "cuda:0" and c.max_seconds == 1800
        steps = c.train_rows // c.batch_size * c.epochs
        assert steps == (32768 if arm == "c" else 8192)
        grid = [e * c.train_rows // c.batch_size for e in range(
            c.dev_every_epochs, c.epochs + 1, c.dev_every_epochs)]
        assert grid == [i * (4096 if arm == "c" else 1024) for i in range(1, 9)]
        updates += steps
        presentations += steps * c.batch_size
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.init_seed)
            model = build_predictor(c.hidden_width, rounds=c.rounds, target_bit=c.target_bit)
        assert sum(p.numel() for p in model.parameters()) == 24961
        states.append(model.state_dict())
    assert (updates, presentations) == (98304, 25165824)
    assert all(torch.equal(s[k], states[0][k]) for s in states for k in states[0])


def test_mlp_data_compute_null_smoke_prefix_and_selection_grids() -> None:
    panels = {}
    for arm in ("a", "b", "c"):
        c = load_config(ROOT / f"innovation2-mlp-data-compute-smoke/{arm}.json")
        assert c.device == "cpu" and c.label_mode == "null" and c.max_seconds == 60
        assert (c.train_pool_rows, c.dev_rows, c.test_rows) == (32, 4, 4)
        assert c.train_rows == (8 if arm == "a" else 32)
        assert c.epochs == c.dev_every_epochs == (1 if arm == "b" else 4)
        assert c.batch_size == 4 and c.data_seed == 193811 and c.null_seed == 193812
        assert c.epochs * c.train_rows // c.batch_size == (32 if arm == "c" else 8)
        panels[arm] = make_inputs(c.train_rows, c.dev_rows, c.test_rows,
                                 seed=c.data_seed, train_pool_rows=c.train_pool_rows)
    assert torch.equal(panels["a"]["train"], panels["b"]["train"][:8])
    for split in ("train", "dev", "test"):
        assert torch.equal(panels["b"][split], panels["c"][split])
        small, large = (make_labels(panels[k][split], dataset="simon", rounds=6,
                                   target_bit=0, key_seed=19, label_mode="null",
                                   null_seed=193812, split=split) for k in ("a", "b"))
        assert torch.equal(small, large[:len(small)])
        if split != "train":
            assert torch.equal(panels["a"][split], panels["b"][split])

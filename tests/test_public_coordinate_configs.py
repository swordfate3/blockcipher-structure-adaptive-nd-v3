"""Fixed proposal budget and paired MLP parameters; not scientific fitting."""
from pathlib import Path

import torch

from cipher_experiments.config import load_config
from cipher_experiments.models.prediction import build_predictor


def test_public_coordinate_proposal_budget_and_initialization() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    paths = list((root / "innovation2-public-coordinate").glob("*.json"))
    assert len(paths) == 4
    states = []
    for path in paths:
        c = load_config(path)
        assert c.model_kind == "mlp" and c.hidden_width == 128
        assert c.prediction_mode == ("direct" if path.stem.startswith("direct")
                                      else "public-coordinate")
        assert c.label_mode == ("null" if path.stem.endswith("null") else "cipher")
        assert c.dataset == "simon" and c.experiment == "prediction"
        assert c.rounds == 6 and c.target_bit == 0 and c.model_features == "full"
        assert (c.train_rows, c.dev_rows, c.test_rows) == (65536, 8192, 65536)
        assert c.train_pool_rows == 65536 and c.device == "cuda:0"
        assert c.batch_size == 256 and c.epochs == 32 and c.dev_every_epochs == 1
        assert c.learning_rate == .001 and c.max_seconds == 1800
        assert (c.data_seed, c.null_seed, c.key_seed, c.init_seed) == (
            193701, 193702, 190502, 17)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.init_seed)
            model = build_predictor(c.hidden_width, rounds=c.rounds,
                                    target_bit=c.target_bit, prediction_mode=c.prediction_mode)
        assert sum(p.numel() for p in model.parameters()) == 24961
        states.append(model.state_dict())
    assert all(torch.equal(s[k], states[0][k]) for s in states for k in states[0])
    for kind in ("direct", "coordinate"):
        c = load_config(root / f"innovation2-public-coordinate-smoke/{kind}.json")
        assert c.label_mode == "null" and c.device == "cpu"
        assert (c.train_rows, c.dev_rows, c.test_rows) == (8, 4, 4)
        assert c.train_pool_rows == 16 and c.batch_size == 4
        assert c.epochs == 2 and c.dev_every_epochs == 2

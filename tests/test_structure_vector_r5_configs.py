"""Five-round fixed proposal; parsing these configurations never trains."""
from __future__ import annotations

import json
from pathlib import Path

from cipher_experiments.config import load_config
from cipher_experiments.models.prediction import build_predictor
from cipher_experiments.models.simon_vector import SimonVectorPredictor

# Compiled public-graph facts for rounds5 (verified in tests/test_simon_vector.py
# against the untouched simon_cone compiler): 266 nodes, depth20, 46 plaintext
# leaves, 44 symbolic leaves and 8d^2+(symbolic+3)d+1 = 9697 parameters.
FIVE_PARAMETERS = 9697


def test_r5_fixed_matrix_matches_frozen_plan_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    real = load_config(root / "configs/innovation2-structure-vector-r5/real.json")
    null = load_config(root / "configs/innovation2-structure-vector-r5/null.json")
    assert real.label_mode == "cipher" and null.label_mode == "null"
    a, b = real.to_dict(), null.to_dict()
    for field in ("name", "label_mode"):
        a.pop(field)
        b.pop(field)
    assert a == b
    for config in (real, null):
        assert config.rounds == 5 and config.target_bit == 0
        assert config.model_kind == "simon-vector" and config.hidden_width == 32
        assert config.model_features == "full" and config.prediction_mode == "direct"
        assert config.device == "cuda:0" and config.max_seconds == 3600
        assert config.train_rows == config.train_pool_rows == 65536
        assert (config.dev_rows, config.test_rows) == (8192, 65536)
        assert config.batch_size == 256 and config.epochs == 32
        assert config.dev_every_epochs == 4 and config.learning_rate == .001
        # Each arm: 8192 updates / 2097152 presentations / 8 DEV evaluations.
        assert config.train_rows % config.batch_size == 0
        assert config.train_rows // config.batch_size * config.epochs == 8192
        assert config.train_rows * config.epochs == 2097152
        assert config.epochs % config.dev_every_epochs == 0
        assert config.epochs // config.dev_every_epochs == 8
    # Two arms combined: 16384 updates / 4194304 presentations.
    assert real.train_rows // real.batch_size * real.epochs * 2 == 16384
    assert real.train_rows * real.epochs * 2 == 4194304
    assert real.name == "r5-structure-vector-real" and null.name == "r5-structure-vector-null"
    # Same plaintext panel/seeds as the six-round batch: this reuses inputs for a
    # rounds contrast, not an independent confirmation with new seeds.
    assert (real.data_seed, real.null_seed, real.key_seed, real.init_seed) == (
        194401, 194402, 190502, 17)
    for old, new in (("real", "real"), ("null", "null")):
        six = json.loads((root / f"configs/innovation2-structure-vector/{old}.json")
                         .read_text(encoding="utf-8"))
        five = json.loads((root / f"configs/innovation2-structure-vector-r5/{new}.json")
                          .read_text(encoding="utf-8"))
        assert {k: v for k, v in six.items() if k not in ("rounds", "name")} == {
            k: v for k, v in five.items() if k not in ("rounds", "name")}
        assert six["rounds"] == 6 and five["rounds"] == 5
    # The existing six-round configurations stay untouched.
    assert load_config(root / "configs/innovation2-structure-vector/real.json").rounds == 6
    assert load_config(root / "configs/innovation2-structure-vector/null.json").rounds == 6


def test_r5_configs_build_five_round_graph_with_fewer_symbolic_leaves() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/innovation2-structure-vector-r5/real.json")
    model = build_predictor(config.hidden_width, model_kind=config.model_kind,
                            rounds=config.rounds, target_bit=config.target_bit,
                            feature_mode=config.model_features,
                            prediction_mode=config.prediction_mode)
    assert isinstance(model, SimonVectorPredictor)
    assert model.rounds == 5
    assert sum(p.numel() for p in model.parameters()) == FIVE_PARAMETERS
    assert (model.plan.node_count, model.plan.depth) == (266, 20)
    assert (len(model.plan.plaintext_indices), len(model.plan.symbolic_coordinates)) == (46, 44)
    # Fewer rounds shrink the symbolic leaf count and parameter total; this is a
    # different target DAG, not the six-round 10529-parameter capacity.
    six = build_predictor(32, model_kind="simon-vector", rounds=6, target_bit=0)
    assert isinstance(six, SimonVectorPredictor)
    assert sum(p.numel() for p in six.parameters()) == 10529


def test_r5_smokes_are_null_cpu_and_cuda_only() -> None:
    root = Path(__file__).resolve().parents[1]
    cpu = load_config(root / "configs/innovation2-structure-vector-r5-smoke.json")
    cuda = load_config(root / "configs/innovation2-structure-vector-r5-cuda-smoke.json")
    for config, device, name in ((cpu, "cpu", "structure-vector-r5-null-engineering"),
                                 (cuda, "cuda:0", "structure-vector-r5-null-cuda-engineering")):
        assert config.device == device and config.label_mode == "null"
        assert config.model_kind == "simon-vector" and config.rounds == 5
        assert config.name == name
        assert (config.train_rows, config.dev_rows, config.test_rows) == (8, 4, 4)
        assert (config.train_pool_rows, config.batch_size, config.epochs) == (8, 4, 1)
        assert config.dev_every_epochs == 1 and config.max_seconds == 60
        # 2 updates / 8 presentations per smoke; the CUDA config is only parsed.
        assert config.train_rows // config.batch_size * config.epochs == 2
        assert config.train_rows * config.epochs == 8
        assert (config.data_seed, config.null_seed, config.key_seed,
                config.init_seed) == (194401, 194402, 190502, 17)
    for old, new in (("innovation2-structure-vector-smoke.json",
                      "innovation2-structure-vector-r5-smoke.json"),
                     ("innovation2-structure-vector-cuda-smoke.json",
                      "innovation2-structure-vector-r5-cuda-smoke.json")):
        six = json.loads((root / "configs" / old).read_text(encoding="utf-8"))
        five = json.loads((root / "configs" / new).read_text(encoding="utf-8"))
        assert {k: v for k, v in six.items() if k not in ("rounds", "name")} == {
            k: v for k, v in five.items() if k not in ("rounds", "name")}
        assert six["rounds"] == 6 and five["rounds"] == 5

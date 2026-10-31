"""Public fixed-background inputs only: no labels, encryption, forward or fit."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from cipher_experiments import checkpoints, training
from cipher_experiments.artifacts import file_sha256
from cipher_experiments.ciphers import simon
from cipher_experiments.config import ExperimentConfig, load_config
from cipher_experiments.data import prediction, synthetic
from cipher_experiments.models.prediction import build_predictor

ROOT = Path(__file__).resolve().parents[1]
SPACES = ("high32-zero", "low32-zero")
FIXED = 0xA5A5A5A5


@pytest.fixture(autouse=True)
def forbid_scientific_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("encryption, labels, forward and training forbidden")
    for module, names in (
        (prediction, ("_encrypt", "make_labels")),
        (simon, ("encrypt",)),
        (training, ("train", "labels_for", "reference_panels", "evaluate_model")),
        (torch.nn.Module, ("_call_impl",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


def sha(x: Tensor) -> str:
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()


def test_default_dictionary_sha_and_subspace_serialization() -> None:
    # Fixed pre-additive-field representative from d6c3c5d3b89afe5dd494a948f6b313b43c1f6d39.
    # No Git/HEAD dependency: source-only copies can run this test.
    expected: dict[str, Any] = {
        "name": "legacy-default", "dataset": "synthetic", "experiment": "prediction",
        "train_rows": 64, "dev_rows": 32, "test_rows": 32, "data_seed": 1,
        "key_seed": 2, "init_seed": 3, "rounds": 3, "target_bit": 0,
        "hidden_width": 32, "model_kind": "mlp", "model_features": "full",
        "prediction_mode": "direct", "label_mode": "cipher", "null_seed": 191302,
        "batch_size": 16, "epochs": 2, "learning_rate": 0.001, "device": "cpu",
        "max_seconds": 60, "train_pool_rows": None, "dev_every_epochs": 1,
    }
    config = ExperimentConfig(name="legacy-default")
    assert config.to_dict() == expected
    assert config.sha256() == "07be17e9124079c8e8d554395a9fcab3b59c5cbacc988fccbbbabb723855844a"
    for space in ("full", *SPACES):
        old_mapping = expected | {"dataset": "simon"}
        if space != "full":
            old_mapping["plaintext_space"] = space
        default = ExperimentConfig(**old_mapping)
        explicit = ExperimentConfig(**old_mapping, plaintext_fixed_value=0)
        old_sha = hashlib.sha256(json.dumps(old_mapping, sort_keys=True).encode()).hexdigest()
        assert default.to_dict() == explicit.to_dict() == old_mapping
        assert default.sha256() == explicit.sha256() == old_sha
        if space != "full":
            nonzero = replace(default, plaintext_fixed_value=FIXED)
            assert nonzero.to_dict() == old_mapping | {"plaintext_fixed_value": FIXED}
            assert nonzero.sha256() != old_sha
            assert ExperimentConfig(**nonzero.to_dict()) == nonzero


@pytest.mark.parametrize("pool", [None, 7, 19])
@pytest.mark.parametrize("space", ["full", *SPACES])
def test_zero_default_retains_input_bytes(space: str, pool: int | None) -> None:
    panels = prediction.make_inputs(7, 4, 3, seed=17, train_pool_rows=pool,
                                    plaintext_space=space)
    explicit = prediction.make_inputs(7, 4, 3, seed=17, train_pool_rows=pool,
                                      plaintext_space=space, plaintext_fixed_value=0)
    rng = random.Random(17)
    values: list[int] = []
    while len(values) < (pool or 7) + 7:
        value = rng.getrandbits(64 if space == "full" else 32)
        if value not in values:
            values.append(value)
    blocks = [v << 32 if space == "low32-zero" else v for v in values]
    legacy = synthetic._binary_rows(blocks)
    start = pool or 7
    expected = {"train": legacy[:7], "dev": legacy[start:start + 4],
                "test": legacy[start + 4:]}
    for split in panels:
        assert sha(panels[split]) == sha(explicit[split]) == sha(expected[split])
    if space == "full" and pool is None:
        assert {s: sha(x) for s, x in panels.items()} == {
            "train": "d0ad791ff0db478550d2188faa73cf4443160df4304ed1f6a71de81761a6f536",
            "dev": "2ad26af922e82291e16f2a0f07645f0ed71e310b2e8ed4f18ffe1d0d143f1a8f",
            "test": "0aa64f36f3011d35d22a80a8b250b5cd12eced62fa1e2ff5a710993d12321051",
        }


@pytest.mark.parametrize("fixed", [1, FIXED, (1 << 32) - 1])
@pytest.mark.parametrize("pool", [None, 11, 23])
def test_placement_active_sequence_uniqueness_and_pool(fixed: int, pool: int | None) -> None:
    state = random.getstate()
    panels = {space: prediction.make_inputs(
        11, 7, 5, seed=195601, train_pool_rows=pool, plaintext_space=space,
        plaintext_fixed_value=fixed) for space in SPACES}
    assert random.getstate() == state
    for space in SPACES:
        zero = prediction.make_inputs(11, 7, 5, seed=195601, train_pool_rows=pool,
                                      plaintext_space=space)
        active = slice(32, 64) if space == "high32-zero" else slice(0, 32)
        for split, x in panels[space].items():
            assert x.dtype == torch.float32 and x.device.type == "cpu"
            assert x.is_contiguous() and not x.requires_grad
            assert torch.equal(x[:, active], zero[split][:, active])
            assert torch.equal(panels[SPACES[0]][split][:, 32:],
                               panels[SPACES[1]][split][:, :32])
            original = prediction._plaintexts(zero[split])
            expected = [v | (fixed << 32) if space == "high32-zero" else v | fixed
                        for v in original]
            assert prediction._plaintexts(x) == expected
        packed = prediction._plaintexts(torch.cat(list(panels[space].values())))
        assert len(packed) == len(set(packed)) == 23
        if pool is not None:
            larger = prediction.make_inputs(pool, 7, 5, seed=195601, train_pool_rows=pool,
                                            plaintext_space=space, plaintext_fixed_value=fixed)
            assert torch.equal(larger["train"][:11], panels[space]["train"])
            for split in ("dev", "test"):
                assert torch.equal(larger[split], panels[space][split])
            all_rows = prediction._plaintexts(torch.cat(list(larger.values())))
            assert len(all_rows) == len(set(all_rows)) == pool + 12


@pytest.mark.parametrize("space", SPACES)
def test_uint32_endpoints_and_collision_budget(monkeypatch: pytest.MonkeyPatch, space: str) -> None:
    class SequenceRng:
        def __init__(self, seed: int) -> None:
            self.values = iter((0, 0, (1 << 32) - 1, 1 << 31))

        def getrandbits(self, bits: int) -> int:
            assert bits == 32
            return next(self.values)

    monkeypatch.setattr(random, "Random", SequenceRng)
    panels = prediction.make_inputs(1, 1, 1, seed=0, plaintext_space=space,
                                    plaintext_fixed_value=FIXED)
    assert prediction._plaintexts(torch.cat(list(panels.values()))) == [
        (FIXED << 32) | v if space == "high32-zero" else (v << 32) | FIXED
        for v in (0, (1 << 32) - 1, 1 << 31)]
    calls: list[int] = []

    class StuckRng:
        def __init__(self, seed: int) -> None:
            pass

        def getrandbits(self, bits: int) -> int:
            calls.append(bits)
            return 0

    monkeypatch.setattr(random, "Random", StuckRng)
    with pytest.raises(RuntimeError, match="bounded draw budget"):
        prediction.make_inputs(1, 1, 1, seed=0, plaintext_space=space,
                               plaintext_fixed_value=FIXED)
    assert calls == [32] * (3 * synthetic._DRAWS_PER_SAMPLE)


@pytest.mark.parametrize("value", [True, False, None, -1, 1 << 32, 1.0, "1", [], {}])
@pytest.mark.parametrize("space", ["full", *SPACES])
def test_invalid_fixed_values_before_rng_or_allocation(
        monkeypatch: pytest.MonkeyPatch, value: Any, space: str) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("validation must precede RNG or allocation")
    monkeypatch.setattr(random, "Random", forbidden)
    monkeypatch.setattr(prediction, "_binary_rows", forbidden)
    with pytest.raises(ValueError, match="plaintext_fixed_value"):
        ExperimentConfig(name="invalid", dataset="simon", plaintext_space=space,
                         plaintext_fixed_value=value)
    with pytest.raises(ValueError, match="plaintext_fixed_value"):
        prediction.make_inputs(1, 1, 1, seed=1, plaintext_space=space,
                               plaintext_fixed_value=value)


def test_full_nonzero_rejected_before_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("full/nonzero validation must precede RNG")
    monkeypatch.setattr(random, "Random", forbidden)
    with pytest.raises(ValueError, match="full plaintext_space requires"):
        ExperimentConfig(name="bad", plaintext_fixed_value=FIXED)
    with pytest.raises(ValueError, match="full plaintext_space requires"):
        prediction.make_inputs(1, 1, 1, seed=0, plaintext_fixed_value=FIXED)


@pytest.mark.parametrize("changes", [
    {"dataset": "synthetic"}, {"dataset": "circuit-synthetic"},
    {"experiment": "innovation1", "dataset": "differential"},
    {"model_kind": "logic16"}, {"model_features": "support"},
    {"prediction_mode": "public-residual"}, {"prediction_mode": "public-coordinate"},
])
@pytest.mark.parametrize("space", SPACES)
def test_nonzero_preserves_subspace_restrictions(changes: dict[str, Any], space: str) -> None:
    parameters: dict[str, Any] = dict(name="bad", dataset="simon", plaintext_space=space,
                                      plaintext_fixed_value=FIXED) | changes
    with pytest.raises(ValueError, match="non-full plaintext_space requires"):
        ExperimentConfig(**parameters)


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("options", [
    {"train_pool_rows": True}, {"train_pool_rows": 0}, {"train_pool_rows": 1_000_001},
    {"train_pool_rows": 1_000_000, "dev": 1_000_000, "test": 1},
])
def test_pool_limits_before_sampling(
        monkeypatch: pytest.MonkeyPatch, space: str, options: dict[str, Any]) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("invalid pool must not sample or allocate")
    monkeypatch.setattr(random, "Random", forbidden)
    monkeypatch.setattr(prediction, "_binary_rows", forbidden)
    parameters: dict[str, Any] = dict(train=1, dev=1, test=1, seed=0,
                                      plaintext_space=space,
                                      plaintext_fixed_value=FIXED) | options
    with pytest.raises((TypeError, ValueError)):
        prediction.make_inputs(**parameters)


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("fixed", [0, FIXED, (1 << 32) - 1])
def test_training_passes_only_nondefault_options_and_public_sampling(
        monkeypatch: pytest.MonkeyPatch, space: str, fixed: int) -> None:
    config = ExperimentConfig(name="public", dataset="simon", plaintext_space=space,
                              plaintext_fixed_value=fixed, train_rows=7, dev_rows=4,
                              test_rows=3, train_pool_rows=19)
    real_make_inputs = prediction.make_inputs
    calls: list[dict[str, Any]] = []

    def capture(train: int, dev: int, test: int, **options: Any) -> dict[str, Tensor]:
        calls.append(options)
        return real_make_inputs(train, dev, test, **options)

    monkeypatch.setattr(prediction, "make_inputs", capture)
    original = training.input_panels(config)
    changed = training.input_panels(replace(config, key_seed=777, null_seed=888,
                                            init_seed=999, target_bit=63, label_mode="null"))
    assert all(torch.equal(x, changed[s]) for s, x in original.items())
    expected = {"seed": 1, "train_pool_rows": 19, "plaintext_space": space}
    if fixed:
        expected["plaintext_fixed_value"] = fixed
    assert calls == [expected, expected]


@pytest.mark.parametrize("space,fixed", [
    ("full", 0), *[(space, fixed) for space in SPACES for fixed in (0, FIXED)],
])
def test_input_metadata_pure_and_precise(space: str, fixed: int) -> None:
    config = ExperimentConfig(name="metadata", dataset="simon", plaintext_space=space,
                              plaintext_fixed_value=fixed)
    semantics = training._input_semantics(config)
    bits = list(range(32)) if space == SPACES[0] else (
        list(range(32, 64)) if space == SPACES[1] else [])
    assert semantics == {
        "plaintext_space": space,
        "active_plaintext_bits": 32 if bits else 64,
        "fixed_plaintext_bits": bits,
        "fixed_plaintext_value": fixed if bits else None,
        "plaintext_bit_order": "64-bit plaintext MSB=0, LSB=63",
        "plaintext_sampling": ("unique uniform uint32 draws; local random.Random(data_seed); "
                               f"placed in active half, other half fixed to 0x{fixed:08X}"
                               if bits else
                               "unique uniform uint64 draws; local random.Random(data_seed)"),
    }
    assert "fixed to zero" not in semantics["plaintext_sampling"]
    for dataset in ("synthetic", "circuit-synthetic"):
        assert training._input_semantics(ExperimentConfig(name="fixture", dataset=dataset)) == {}


@pytest.mark.parametrize("smoke", [False, True])
def test_configs_exact_two_field_changes_and_budgets(smoke: bool) -> None:
    suffix = "-smoke" if smoke else ""
    folder = ROOT / f"configs/innovation2-nonzero-background{suffix}"
    assert {p.name for p in folder.iterdir()} == {"xor2.json", "null.json"}
    for arm in ("xor2", "null"):
        source = ROOT / f"configs/innovation2-plaintext-subspace{suffix}/high32-zero-{arm}.json"
        expected = json.loads(source.read_text()) | {
            "name": f"r3-mlp-nonzero-background-{arm}{suffix}", "plaintext_fixed_value": FIXED}
        path = folder / f"{arm}.json"
        assert json.loads(path.read_text()) == expected
        config = load_config(path)
        assert config.plaintext_fixed_value == 2779096485
        assert (config.rounds, config.hidden_width, config.plaintext_space) == (
            3, 128, "high32-zero")
        assert config.epochs * (config.train_rows // config.batch_size) == (2 if smoke else 8192)
        assert config.train_rows * config.epochs == (8 if smoke else 2097152)
        assert config.label_mode == ("null" if smoke or arm == "null" else "cipher")


def test_json_strict_fixed_value_validation(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    for raw in (
        '{"name":"bad","plaintext_fixed_value":true}',
        '{"name":"bad","plaintext_fixed_value":4294967296}',
        '{"name":"bad","plaintext_fixed_value":1}',
        '{"name":"bad","plaintext_fixed_value":0,"plaintext_fixed_value":0}',
    ):
        path.write_text(raw)
        with pytest.raises(ValueError):
            load_config(path)
    path.write_text(json.dumps(dict(name="valid", dataset="simon", plaintext_space=SPACES[0],
                                    plaintext_fixed_value=FIXED)))
    assert load_config(path).plaintext_fixed_value == FIXED


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("fixed", [0, FIXED, (1 << 32) - 1])
def test_strict_checkpoint_identity_without_training(
        tmp_path: Path, space: str, fixed: int) -> None:
    config = ExperimentConfig(name="identity", dataset="simon", plaintext_space=space,
                              plaintext_fixed_value=fixed)
    path = tmp_path / "model.pt"
    checkpoints.save_checkpoint(path, build_predictor(config.hidden_width), config, 1, 0, 0, .5)
    model, loaded, _, _ = checkpoints.load_checkpoint(
        path, file_sha256(path), device=torch.device("cpu"))
    assert loaded == config and model.checkpoint_config_sha256 == config.sha256()
    assert all(torch.equal(x, training.input_panels(loaded)[s])
               for s, x in training.input_panels(config).items())
    payload = torch.load(path, weights_only=True)
    assert ("plaintext_fixed_value" in payload["config"]) == bool(fixed)
    # Validating file bytes alone is insufficient: mutate to another legal background
    # and recompute the file SHA, but not the embedded configuration SHA.
    payload["config"]["plaintext_fixed_value"] = 1 if not fixed else 0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="configuration SHA"):
        checkpoints.load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))

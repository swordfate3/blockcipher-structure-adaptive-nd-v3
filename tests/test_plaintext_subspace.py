"""Plaintext-space contracts: fake encryption and bounded CPU/null fitting only."""
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
ARMS = ("bit", "xor2", "xor3", "xor-and", "null")


@pytest.fixture(autouse=True)
def forbid_encrypt(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> int:
        raise AssertionError("real encryption/labels forbidden in plaintext-space tests")
    monkeypatch.setattr(prediction, "_encrypt", forbidden)
    monkeypatch.setattr(simon, "encrypt", forbidden)


def sha(x: Tensor) -> str:
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("pool", [None, 7, 19])
def test_default_inputs_exact_legacy_bytes(pool: int | None) -> None:
    panels = prediction.make_inputs(7, 4, 3, seed=17, train_pool_rows=pool)
    explicit = prediction.make_inputs(7, 4, 3, seed=17, train_pool_rows=pool,
                                      plaintext_space="full")
    x = synthetic._binary_rows(synthetic._unique_blocks((pool or 7) + 7, 17))
    expected = {"train": x[:7], "dev": x[(pool or 7):(pool or 7) + 4],
                "test": x[(pool or 7) + 4:]}
    for split in panels:
        assert sha(panels[split]) == sha(expected[split]) == sha(explicit[split])
    if pool is None:
        assert {s: sha(x) for s, x in panels.items()} == {
            "train": "d0ad791ff0db478550d2188faa73cf4443160df4304ed1f6a71de81761a6f536",
            "dev": "2ad26af922e82291e16f2a0f07645f0ed71e310b2e8ed4f18ffe1d0d143f1a8f",
            "test": "0aa64f36f3011d35d22a80a8b250b5cd12eced62fa1e2ff5a710993d12321051",
        }


def test_fixed_legacy_full_bit_xor2_dict_and_sha() -> None:
    # Fixture version: d6c3c5d3b89afe5dd494a948f6b313b43c1f6d39 (pre-subspace).
    # These dictionaries and SHA values were captured from that version's config.py,
    # not computed from current defaults. Representative coverage, not all configs:
    # minimal defaults plus configs/innovation2-cipher-expression-low-round/r3-{bit,xor2}.json.
    # Keep fixtures in this test so source copies need neither Git nor historical code.
    default_expected: dict[str, Any] = {
        "name": "legacy-default",
        "dataset": "synthetic",
        "experiment": "prediction",
        "train_rows": 64,
        "dev_rows": 32,
        "test_rows": 32,
        "data_seed": 1,
        "key_seed": 2,
        "init_seed": 3,
        "rounds": 3,
        "target_bit": 0,
        "hidden_width": 32,
        "model_kind": "mlp",
        "model_features": "full",
        "prediction_mode": "direct",
        "label_mode": "cipher",
        "null_seed": 191302,
        "batch_size": 16,
        "epochs": 2,
        "learning_rate": 0.001,
        "device": "cpu",
        "max_seconds": 60,
        "train_pool_rows": None,
        "dev_every_epochs": 1,
    }
    bit_expected: dict[str, Any] = {
        "name": "r3-mlp-cipher-expression-bit",
        "dataset": "simon",
        "experiment": "prediction",
        "train_rows": 65536,
        "dev_rows": 8192,
        "test_rows": 65536,
        "data_seed": 195601,
        "key_seed": 190502,
        "init_seed": 17,
        "rounds": 3,
        "target_bit": 0,
        "hidden_width": 128,
        "model_kind": "mlp",
        "model_features": "full",
        "prediction_mode": "direct",
        "label_mode": "cipher",
        "null_seed": 195602,
        "batch_size": 256,
        "epochs": 32,
        "learning_rate": 0.001,
        "device": "cuda:0",
        "max_seconds": 1800,
        "train_pool_rows": 65536,
        "dev_every_epochs": 1,
    }
    xor2_expected = bit_expected | {
        "name": "r3-mlp-cipher-expression-xor2",
        "label_expression": "xor2",
        "label_bits": (0, 8),
    }
    fixtures = (
        ({"name": "legacy-default"}, default_expected,
         "07be17e9124079c8e8d554395a9fcab3b59c5cbacc988fccbbbabb723855844a"),
        (bit_expected | {"label_expression": "bit", "label_bits": []}, bit_expected,
         "744b38ad5eff4fd1dbe1b792efff9467fff43ededac9fa64ba96dea202a060f7"),
        (xor2_expected | {"label_bits": [0, 8]}, xor2_expected,
         "74ae9d7c5a9c4c02963f4cf3a4f082e819acd9979599f261520879237749da99"),
    )
    for raw, expected, expected_sha in fixtures:
        current = ExperimentConfig(**raw)
        explicit_full = ExperimentConfig(**raw, plaintext_space="full")
        for config in (current, explicit_full, replace(current, plaintext_space="full")):
            assert config.to_dict() == expected, raw["name"]
            assert config.sha256() == expected_sha, raw["name"]


@pytest.mark.parametrize("value", [None, True, 32, 32.0, [], {}, "", "high32", "FULL"])
def test_invalid_space_types_and_values(value: Any) -> None:
    with pytest.raises(ValueError, match="plaintext_space"):
        ExperimentConfig(name="invalid", plaintext_space=value)
    with pytest.raises(ValueError, match="plaintext_space"):
        prediction.make_inputs(1, 1, 1, seed=1, plaintext_space=value)


@pytest.mark.parametrize("changes", [
    {"dataset": "synthetic"}, {"dataset": "circuit-synthetic"},
    {"experiment": "innovation1", "dataset": "differential"},
    {"model_features": "support"}, {"prediction_mode": "public-residual"},
    {"prediction_mode": "public-coordinate"}, {"prediction_mode": "plaintext-xor"},
    {"prediction_mode": "plaintext-xor-no-final-low"},
    *[{"model_kind": kind} for kind in (
        "logic16", "circuit-aligned", "circuit-random", "simon-structure", "simon-unrolled",
        "simon-r2-table", "simon-r2-local", "simon-r2-vector", "simon-cnn",
        "simon-rotation-cnn", "simon-vector")],
])
@pytest.mark.parametrize("space", SPACES)
def test_subspace_rejects_incompatible_combinations(changes: dict[str, Any], space: str) -> None:
    with pytest.raises(ValueError, match="non-full plaintext_space requires"):
        ExperimentConfig(**(dict(name="bad", dataset="simon", plaintext_space=space) | changes))


@pytest.mark.parametrize("pool", [None, 23])
def test_subspaces_paired_unique_split_and_pool_prefix(pool: int | None) -> None:
    state = random.getstate()
    high = prediction.make_inputs(11, 7, 5, seed=195601, train_pool_rows=pool,
                                  plaintext_space="high32-zero")
    low = prediction.make_inputs(11, 7, 5, seed=195601, train_pool_rows=pool,
                                 plaintext_space="low32-zero")
    assert random.getstate() == state
    rng = random.Random(195601)
    values: list[int] = []
    while len(values) < (pool or 11) + 12:
        value = rng.getrandbits(32)
        if value not in values:
            values.append(value)
    expected = {"train": values[:11], "dev": values[(pool or 11):(pool or 11) + 7],
                "test": values[(pool or 11) + 7:]}
    for split in high:
        a, b = high[split], low[split]
        assert a.device.type == b.device.type == "cpu"
        assert a.dtype == b.dtype == torch.float32
        assert a.shape == b.shape == (len(expected[split]), 64)
        assert a.is_contiguous() and b.is_contiguous() and not a.requires_grad
        assert not a[:, :32].any() and not b[:, 32:].any()
        assert torch.equal(a[:, 32:], b[:, :32])
        assert prediction._plaintexts(a) == expected[split]
        assert prediction._plaintexts(b) == [v << 32 for v in expected[split]]
        assert set(a.flatten().tolist()) <= {0., 1.}
        for panels in (high, low):
            assert torch.equal(panels[split], prediction.make_inputs(
                11, 7, 5, seed=195601, train_pool_rows=pool,
                plaintext_space="high32-zero" if panels is high else "low32-zero")[split])
    assert len(set(prediction._plaintexts(torch.cat(list(high.values()))))) == 23
    if pool is not None:
        for space, panels in zip(SPACES, (high, low), strict=True):
            larger = prediction.make_inputs(23, 7, 5, seed=195601, train_pool_rows=pool,
                                            plaintext_space=space)
            assert torch.equal(panels["train"], larger["train"][:11])
            assert torch.equal(panels["dev"], larger["dev"])
            assert torch.equal(panels["test"], larger["test"])
            all_rows = prediction._plaintexts(torch.cat(list(larger.values())))
            assert len(set(all_rows)) == len(all_rows) == 35


@pytest.mark.parametrize("options", [
    {"train": True}, {"dev": 0}, {"test": 1_000_001}, {"seed": True}, {"seed": -1},
    {"train_pool_rows": True}, {"train_pool_rows": 0}, {"train_pool_rows": 1_000_001},
    {"train": 1_000_000, "dev": 1_000_000, "test": 1},
    {"train_pool_rows": 1_000_000, "dev": 1_000_000, "test": 1},
])
def test_invalid_sampling_parameters_before_rng_or_allocation(
        monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("validation must precede RNG and tensor allocation")
    monkeypatch.setattr(random, "Random", forbidden)
    monkeypatch.setattr(prediction, "_binary_rows", forbidden)
    with pytest.raises((TypeError, ValueError)):
        parameters: dict[str, Any] = dict(train=1, dev=1, test=1, seed=0,
                                          plaintext_space="low32-zero") | options
        prediction.make_inputs(**parameters)


def test_uint32_endpoints_collisions_and_finite_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    class SequenceRng:
        def __init__(self, seed: int) -> None:
            self.values = iter([0, 0, (1 << 32) - 1, (1 << 31)])

        def getrandbits(self, bits: int) -> int:
            assert bits == 32
            return next(self.values)

    monkeypatch.setattr(random, "Random", SequenceRng)
    for space in SPACES:
        panels = prediction.make_inputs(1, 1, 1, seed=0, plaintext_space=space)
        assert prediction._plaintexts(torch.cat(list(panels.values()))) == [
            v if space == "high32-zero" else v << 32 for v in (0, (1 << 32) - 1, 1 << 31)]

    calls: list[int] = []

    class StuckRng:
        def __init__(self, seed: int) -> None:
            pass

        def getrandbits(self, bits: int) -> int:
            calls.append(bits)
            return 0

    monkeypatch.setattr(random, "Random", StuckRng)
    with pytest.raises(RuntimeError, match="bounded draw budget"):
        prediction.make_inputs(1, 1, 1, seed=0, plaintext_space="high32-zero")
    assert calls == [32] * (3 * synthetic._DRAWS_PER_SAMPLE)


def test_default_make_inputs_retains_old_fake_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    def old_fake(train: int, dev: int, test: int, *, seed: int,
                 train_pool_rows: int | None = None) -> dict[str, Tensor]:
        assert (train, dev, test, seed, train_pool_rows) == (64, 32, 32, 1, None)
        return {"train": torch.zeros(train, 64)}
    monkeypatch.setattr(prediction, "make_inputs", old_fake)
    assert len(training.input_panels(ExperimentConfig(name="old"))["train"]) == 64


@pytest.mark.parametrize("space", SPACES)
def test_null_stream_same_between_spaces_and_nonbit_references_empty(space: str) -> None:
    panels = prediction.make_inputs(8, 4, 4, seed=195601, plaintext_space=space)
    other = prediction.make_inputs(8, 4, 4, seed=195601,
                                   plaintext_space=SPACES[1] if space == SPACES[0] else SPACES[0])
    for arm, bits in (("bit", ()), ("xor2", (0, 8)), ("xor3", (0, 8, 16)),
                      ("xor-and", (0, 8, 16))):
        config = ExperimentConfig(name="null", dataset="simon", plaintext_space=space,
                                  label_mode="null", null_seed=195602,
                                  label_expression=arm, label_bits=bits)
        if arm != "bit":
            assert training.reference_panels(config, panels) == {}
        for split in panels:
            assert torch.equal(training.labels_for(config, panels[split], split),
                               training.labels_for(config, other[split], split))


@pytest.mark.parametrize("smoke", [False, True])
def test_twenty_configs_exact_source_changes_and_budgets(smoke: bool) -> None:
    suffix = "-smoke" if smoke else ""
    folder = ROOT / f"configs/innovation2-plaintext-subspace{suffix}"
    assert {p.name for p in folder.iterdir()} == {
        f"{space}-{arm}.json" for space in SPACES for arm in ARMS}
    for space in SPACES:
        for arm in ARMS:
            source = ROOT / f"configs/innovation2-cipher-expression-low-round{suffix}/r3-{arm}.json"
            raw = json.loads(source.read_text())
            path = folder / f"{space}-{arm}.json"
            expected_name = f"r3-mlp-plaintext-subspace-{space}-{arm}{suffix}"
            assert json.loads(path.read_text()) == raw | {
                "name": expected_name, "plaintext_space": space}
            config = load_config(path)
            assert config.rounds == 3 and config.hidden_width == 128
            assert (config.key_seed, config.data_seed, config.init_seed, config.null_seed) == (
                190502, 195601, 17, 195602)
            assert config.label_mode == ("null" if smoke or arm == "null" else "cipher")
            updates = config.epochs * (config.train_rows // config.batch_size)
            assert updates == (2 if smoke else 8192)
            assert config.train_rows * config.epochs == (8 if smoke else 2097152)
            assert (config.train_pool_rows, config.device, config.max_seconds) == (
                (8, "cpu", 60) if smoke else (65536, "cuda:0", 1800))
            model = build_predictor(config.hidden_width, rounds=config.rounds)
            assert sum(p.numel() for p in model.parameters()) == 24961


def test_strict_json_fields_and_sampling_limits(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    for text in (
        '{"name":"bad","plaintext_space":null}',
        '{"name":"bad","plaintext_space":"high32-zero","plaintext_mask":0}',
        '{"name":"bad","plaintext_space":"full","plaintext_space":"high32-zero"}',
    ):
        path.write_text(text)
        with pytest.raises(ValueError):
            load_config(path)
    for changes in (
        {"train_rows": 1_000_000, "dev_rows": 1_000_000, "test_rows": 1},
        {"train_pool_rows": 1_000_000, "dev_rows": 1_000_000, "test_rows": 1},
    ):
        parameters: dict[str, Any] = dict(name="limit", dataset="simon",
                                          plaintext_space="high32-zero") | changes
        with pytest.raises(ValueError, match="2,000,000"):
            ExperimentConfig(**parameters)
    # Effective capacity is still the existing 2M cap, below the 2^32 space.
    assert 2_000_000 < (1 << 32)


@pytest.mark.parametrize("space", SPACES)
def test_smoke_inputs_and_fake_cipher_labels_only(
        monkeypatch: pytest.MonkeyPatch, space: str) -> None:
    calls: list[int] = []

    def fake(plaintext: int, key: int, rounds: int) -> int:
        calls.append(plaintext)
        assert rounds == 3
        return plaintext ^ 0xA55AA55A5AA55AA5

    monkeypatch.setattr(prediction, "_encrypt", fake)
    for arm in ARMS:
        config = load_config(ROOT / "configs/innovation2-plaintext-subspace-smoke"
                             / f"{space}-{arm}.json")
        panels = training.input_panels(config)
        assert all(x.shape == (rows, 64) for x, rows in zip(
            panels.values(), (8, 4, 4), strict=True))
        # Only the explicit fake is permitted; no real cipher or key value enters sampling.
        cipher_config = replace(config, label_mode="cipher")
        x = panels["train"]
        y = training.labels_for(cipher_config, x, "train")
        bits = config.label_bits or (config.target_bit,)
        expected = []
        for plaintext in prediction._plaintexts(x):
            selected = [((plaintext ^ 0xA55AA55A5AA55AA5) >> (63 - b)) & 1 for b in bits]
            if config.label_expression == "bit":
                value = selected[0]
            elif config.label_expression == "xor2":
                value = selected[0] ^ selected[1]
            elif config.label_expression == "xor3":
                value = selected[0] ^ selected[1] ^ selected[2]
            else:
                value = selected[0] ^ (selected[1] & selected[2])
            expected.append([float(value)])
        assert y.tolist() == expected
    assert len(calls) == 8 * len(ARMS)


@pytest.mark.parametrize("space", SPACES)
def test_checkpoint_identity_binds_space(tmp_path: Path, space: str) -> None:
    config = ExperimentConfig(name="identity", dataset="simon", plaintext_space=space)
    assert config.to_dict()["plaintext_space"] == space
    assert len({replace(config, plaintext_space=s).sha256() for s in (*SPACES, "full")}) == 3
    path = tmp_path / "model.pt"
    checkpoints.save_checkpoint(path, build_predictor(config.hidden_width), config, 1, 0, 0, .5)
    model, loaded, _, _ = checkpoints.load_checkpoint(
        path, file_sha256(path), device=torch.device("cpu"))
    assert loaded == config and model.checkpoint_config_sha256 == config.sha256()
    assert all(torch.equal(x, training.input_panels(loaded)[s])
               for s, x in training.input_panels(config).items())
    payload = torch.load(path, weights_only=True)
    payload["config"]["plaintext_space"] = "full"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="configuration SHA"):
        checkpoints.load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))


@pytest.mark.parametrize("space", [*SPACES, "full", "circuit-fixture", "synthetic-fixture"])
def test_bounded_cpu_null_metadata_and_strict_load_before_test(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, space: str) -> None:
    # Three SIMON-null fits x two updates + two fixture fits x one = eight total.
    circuit = space == "circuit-fixture"
    fixture = space.endswith("-fixture")
    dataset = "circuit-synthetic" if circuit else "synthetic" if fixture else "simon"
    config = ExperimentConfig(name="metadata", dataset=dataset,
                              plaintext_space="full" if fixture else space,
                              train_rows=8, dev_rows=4, test_rows=4,
                              batch_size=8 if fixture else 4, epochs=1,
                              rounds=6, label_mode="cipher" if fixture else "null")
    if circuit:
        # No cipher or real labels: retain the circuit input distribution but use
        # the existing null stream as an explicit test double for labels.
        from cipher_experiments.data import circuit as circuit_data
        monkeypatch.setattr(circuit_data, "circuit_labels", lambda x: prediction.make_labels(
            x, dataset="simon", rounds=6, target_bit=0, key_seed=1,
            label_mode="null", split="train"))
    events: list[str] = []
    original_load = checkpoints.load_checkpoint
    original_labels = training.labels_for

    def load(*args: Any, **kwargs: Any) -> Any:
        result = original_load(*args, **kwargs)
        events.append("strict-load")
        return result

    def labels(cfg: ExperimentConfig, x: Tensor, split: str) -> Tensor:
        if split == "test":
            assert events == ["train", "dev", "strict-load"]
        events.append(split)
        if fixture and not circuit:
            return prediction.make_labels(x, dataset="simon", rounds=6, target_bit=0,
                                          key_seed=1, label_mode="null", split=split)
        return original_labels(cfg, x, split)

    monkeypatch.setattr(training, "load_checkpoint", load)
    monkeypatch.setattr(training, "labels_for", labels)
    output = tmp_path / "run"
    result = training.train(config, output)
    assert result["updates"] == (1 if fixture else 2) and result["presentations"] == 8
    assert events == ["train", "dev", "strict-load", "test"]
    manifest = json.loads((output / "manifest.json").read_text())
    saved_result = json.loads((output / "results.json").read_text())
    assert saved_result == result
    keys = ("plaintext_space", "active_plaintext_bits", "fixed_plaintext_bits",
            "fixed_plaintext_value", "plaintext_sampling", "plaintext_bit_order")
    if fixture:
        assert all(key not in artifact for artifact in (manifest, result) for key in keys)
    else:
        fixed = list(range(32)) if space == SPACES[0] else (
            list(range(32, 64)) if space == SPACES[1] else [])
        for artifact in (manifest, result):
            assert artifact["plaintext_space"] == space
            assert artifact["active_plaintext_bits"] == (32 if fixed else 64)
            assert artifact["fixed_plaintext_bits"] == fixed
            assert artifact["fixed_plaintext_value"] == (0 if fixed else None)
            assert ("uint32" if fixed else "uint64") in artifact["plaintext_sampling"]
        assert {k: manifest[k] for k in keys} == {k: result[k] for k in keys}
    data = json.loads((output / "data.json").read_text())
    for split, x in training.input_panels(config).items():
        assert set(data[split]) == {"rows", "x_sha256", "y_sha256"}
        assert data[split]["x_sha256"] == sha(x)
    assert json.loads((output / "selection.json").read_text())["test_labels_generated"] is False
    assert json.loads((output / "exit.json").read_text())["status"] == "completed"

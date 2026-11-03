"""Artificial-only config/factory/report contracts; no cipher, data, fit or optimizer.

Frozen identities were reviewed at 4596264d; fixtures are self-contained.
Tests catch ignored research fields, identity drift, wrong dispatch and report side effects.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from cipher_experiments.config import ExperimentConfig, load_config
from cipher_experiments.models.prediction import (
    PlaintextPredictor,
    build_predictor,
    public_coordinate_inverse,
    public_coordinate_transform,
    public_reference,
)
from cipher_experiments.models.public_prediction import PublicPredictor
from cipher_experiments.prediction_reports import experiment_kind, model_metadata

# Copied once from the reviewed baseline, never read from private files at runtime.
LEGACY: list[dict[str, Any]] = [{'raw_config': {'name': 'legacy-synthetic',
                 'dataset': 'synthetic',
                 'experiment': 'prediction',
                 'train_rows': 64,
                 'dev_rows': 32,
                 'test_rows': 32,
                 'data_seed': 1,
                 'key_seed': 2,
                 'init_seed': 3,
                 'rounds': 3,
                 'target_bit': 0,
                 'hidden_width': 32,
                 'model_kind': 'mlp',
                 'model_features': 'full',
                 'prediction_mode': 'direct',
                 'label_mode': 'cipher',
                 'null_seed': 191302,
                 'batch_size': 16,
                 'epochs': 2,
                 'learning_rate': 0.001,
                 'device': 'cpu',
                 'max_seconds': 60,
                 'train_pool_rows': None,
                 'dev_every_epochs': 1},
  'config_sha256': '7cd4b17dd92377e1103ece1153d80d80af7c7dd8b1120c4fa5378c2a6df151ed',
  'state_names': ['layers.0.weight',
                  'layers.0.bias',
                  'layers.2.weight',
                  'layers.2.bias',
                  'layers.4.weight',
                  'layers.4.bias'],
  'state_sha256': '30c44d80cf0b0084a12a8e4d8be1ee1d704f74c17e894ae19ef5fa03dd4c706a',
  'rng_sha256': '1eca0107af5afee44873106a69d4050b4e93b42aa38ac2baafb56b927f503016',
  'parameters': 3169},
 {'raw_config': {'name': 'legacy-full',
                 'dataset': 'simon',
                 'experiment': 'prediction',
                 'train_rows': 64,
                 'dev_rows': 32,
                 'test_rows': 32,
                 'data_seed': 1,
                 'key_seed': 2,
                 'init_seed': 3,
                 'rounds': 6,
                 'target_bit': 0,
                 'hidden_width': 128,
                 'model_kind': 'mlp',
                 'model_features': 'full',
                 'prediction_mode': 'direct',
                 'label_mode': 'cipher',
                 'null_seed': 191302,
                 'batch_size': 16,
                 'epochs': 2,
                 'learning_rate': 0.001,
                 'device': 'cpu',
                 'max_seconds': 60,
                 'train_pool_rows': None,
                 'dev_every_epochs': 1},
  'config_sha256': '1b34641dd90137842daeee59b8bfebe53585cd13a89409e59069813251f45942',
  'state_names': ['layers.0.weight',
                  'layers.0.bias',
                  'layers.2.weight',
                  'layers.2.bias',
                  'layers.4.weight',
                  'layers.4.bias'],
  'state_sha256': '793e30f65789fc521b6a146fa8af5ea6bdc4b06febb77642ebbb045c3d37482f',
  'rng_sha256': '8b312242e4312675ee2f4b3bea9c3678de42499f012a7ef53e7cc5f3c13c8a16',
  'parameters': 24961},
 {'raw_config': {'name': 'legacy-coordinate',
                 'dataset': 'simon',
                 'experiment': 'prediction',
                 'train_rows': 64,
                 'dev_rows': 32,
                 'test_rows': 32,
                 'data_seed': 1,
                 'key_seed': 2,
                 'init_seed': 3,
                 'rounds': 6,
                 'target_bit': 0,
                 'hidden_width': 128,
                 'model_kind': 'mlp',
                 'model_features': 'full',
                 'prediction_mode': 'public-coordinate',
                 'label_mode': 'cipher',
                 'null_seed': 191302,
                 'batch_size': 16,
                 'epochs': 2,
                 'learning_rate': 0.001,
                 'device': 'cpu',
                 'max_seconds': 60,
                 'train_pool_rows': None,
                 'dev_every_epochs': 1},
  'config_sha256': '65238e3680e5d8e68874f29071cb6e9efba644156492cc1e65ba2a655fdbe5f5',
  'state_names': ['layers.0.weight',
                  'layers.0.bias',
                  'layers.2.weight',
                  'layers.2.bias',
                  'layers.4.weight',
                  'layers.4.bias'],
  'state_sha256': '793e30f65789fc521b6a146fa8af5ea6bdc4b06febb77642ebbb045c3d37482f',
  'rng_sha256': '8b312242e4312675ee2f4b3bea9c3678de42499f012a7ef53e7cc5f3c13c8a16',
  'parameters': 24961},
 {'raw_config': {'name': 'legacy-background',
                 'dataset': 'simon',
                 'experiment': 'prediction',
                 'train_rows': 64,
                 'dev_rows': 32,
                 'test_rows': 32,
                 'data_seed': 1,
                 'key_seed': 2,
                 'init_seed': 3,
                 'rounds': 3,
                 'target_bit': 0,
                 'hidden_width': 128,
                 'model_kind': 'mlp',
                 'model_features': 'full',
                 'prediction_mode': 'direct',
                 'label_mode': 'cipher',
                 'null_seed': 191302,
                 'batch_size': 16,
                 'epochs': 2,
                 'learning_rate': 0.001,
                 'device': 'cpu',
                 'max_seconds': 60,
                 'train_pool_rows': None,
                 'dev_every_epochs': 1,
                 'label_expression': 'xor2',
                 'label_bits': [0, 8],
                 'plaintext_space': 'high32-zero',
                 'plaintext_fixed_value': 2779096485},
  'config_sha256': '02ba1afd441dee76d9b4fafa2cd0d553081db82130ba77f4035dbf1080657e7a',
  'state_names': ['layers.0.weight',
                  'layers.0.bias',
                  'layers.2.weight',
                  'layers.2.bias',
                  'layers.4.weight',
                  'layers.4.bias'],
  'state_sha256': '793e30f65789fc521b6a146fa8af5ea6bdc4b06febb77642ebbb045c3d37482f',
  'rng_sha256': '8b312242e4312675ee2f4b3bea9c3678de42499f012a7ef53e7cc5f3c13c8a16',
  'parameters': 24961}]

RESEARCH: dict[str, Any] = dict(public_input="identity", public_readout="direct",
                              background_init="standard",
                l1_penalty=0.0, record_predictions=False)
OLD_KINDS = ("mlp", "logic16", "circuit-aligned", "circuit-random", "simon-structure",
             "simon-unrolled", "simon-r2-table", "simon-r2-local", "simon-r2-vector",
             "simon-cnn", "simon-rotation-cnn", "simon-vector")
NONDEFAULT = (dict(public_input="simon-interactions"),
              dict(public_readout="zero-round-constant-residual"),
              dict(background_init="match-direct"), dict(l1_penalty=1e-4))


class StringSubclass(str):
    pass


def config(**overrides: Any) -> ExperimentConfig:
    options: dict[str, Any] = dict(name="public-test", dataset="simon", model_kind="public-mlp",
                                   rounds=6, hidden_width=128)
    options.update(overrides)
    return ExperimentConfig(**options)


def n1(**overrides: Any) -> ExperimentConfig:
    return config(**dict(dict(rounds=3, public_input="background-zero", label_expression="xor2",
                              label_bits=(0, 8), plaintext_space="high32-zero",
                              plaintext_fixed_value=0xA5A5A5A5), **overrides))


def linear(**overrides: Any) -> ExperimentConfig:
    return config(**dict(dict(model_kind="public-linear", public_input="simon-interactions",
                              public_readout="zero-round-constant-residual", l1_penalty=1e-4),
                         **overrides))


def factory_options(cfg: ExperimentConfig) -> dict[str, Any]:
    return dict(hidden_width=cfg.hidden_width, model_kind=cfg.model_kind, rounds=cfg.rounds,
                target_bit=cfg.target_bit, feature_mode=cfg.model_features,
                prediction_mode=cfg.prediction_mode, public_input=cfg.public_input,
                public_readout=cfg.public_readout, background_init=cfg.background_init,
                l1_penalty=cfg.l1_penalty, plaintext_space=cfg.plaintext_space,
                plaintext_fixed_value=cfg.plaintext_fixed_value)


@pytest.fixture(autouse=True)
def no_grad() -> Any:
    with torch.no_grad():
        yield


@pytest.mark.parametrize("record", LEGACY, ids=lambda r: r["raw_config"]["name"])
def test_frozen_legacy_mapping_sha_state_registration_rng(record: dict[str, Any]) -> None:
    cfg = ExperimentConfig(**record["raw_config"])
    # JSON normalizes the legacy label_bits list vs dataclass tuple only.
    assert json.loads(json.dumps(cfg.to_dict())) == record["raw_config"]
    assert cfg.sha256() == record["config_sha256"]
    assert replace(cfg, **RESEARCH).to_dict() == cfg.to_dict()
    assert replace(cfg, **RESEARCH).sha256() == record["config_sha256"]
    torch.manual_seed(17)
    model = build_predictor(cfg.hidden_width, model_kind=cfg.model_kind, rounds=cfg.rounds,
                            target_bit=cfg.target_bit, feature_mode=cfg.model_features,
                            prediction_mode=cfg.prediction_mode)
    assert list(model.state_dict()) == record["state_names"]
    digest = hashlib.sha256(b"".join(
        name.encode() + value.cpu().contiguous().numpy().tobytes()
        for name, value in model.state_dict().items())).hexdigest()
    assert digest == record["state_sha256"]
    rng_sha = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    assert rng_sha == record["rng_sha256"]
    assert sum(p.numel() for p in model.parameters()) == record["parameters"]


@pytest.mark.parametrize("rounds", [6, 8, 10])
@pytest.mark.parametrize("public_input", ["identity", "simon-interactions"])
@pytest.mark.parametrize("public_readout", ["direct", "zero-round-constant-residual"])
def test_n2_valid_combinations_roundtrip(tmp_path: Path, rounds: int, public_input: str,
                                         public_readout: str) -> None:
    cfg = config(rounds=rounds, public_input=public_input, public_readout=public_readout)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert load_config(path) == cfg
    assert isinstance(build_predictor(**factory_options(cfg)), PublicPredictor)


@pytest.mark.parametrize("fixed", [0, 0xA5A5A5A5, 0xFFFFFFFF])
@pytest.mark.parametrize("init", ["standard", "match-direct"])
def test_n1_valid_background_slice(fixed: int, init: str) -> None:
    cfg = n1(plaintext_fixed_value=fixed, background_init=init)
    assert isinstance(build_predictor(**factory_options(cfg)), PublicPredictor)


@pytest.mark.parametrize("rounds", [6, 8, 10])
def test_linear_valid_identity_and_parameter_count(rounds: int) -> None:
    cfg = linear(rounds=rounds)
    model = build_predictor(**factory_options(cfg))
    assert isinstance(model, PublicPredictor)
    assert sum(p.numel() for p in model.parameters()) == 97
    assert cfg.to_dict()["l1_penalty"] == 1e-4


@pytest.mark.parametrize("builder,changes", [
    (config, dict(dataset="synthetic")), (config, dict(experiment="innovation1")),
    (config, dict(model_features="support")), (config, dict(prediction_mode="public-residual")),
    (config, dict(prediction_mode="public-coordinate")), (config, dict(rounds=3)),
    (config, dict(rounds=7)), (config, dict(target_bit=1)), (config, dict(target_bit=False)),
    (config, dict(rounds=6.0)), (config, dict(hidden_width=True)),
    (config, dict(plaintext_space="high32-zero")), (config, dict(plaintext_fixed_value=1)),
    (config, dict(background_init="match-direct")), (config, dict(l1_penalty=1e-4)),
    (config, dict(label_expression="xor2", label_bits=(0, 8))),
    (n1, dict(rounds=6)), (n1, dict(target_bit=1)), (n1, dict(public_input="identity")),
    (n1, dict(label_expression="bit", label_bits=())), (n1, dict(label_bits=(0, 9))),
    (n1, dict(label_expression="xor3", label_bits=(0, 8, 9))),
    (n1, dict(label_expression="xor-and", label_bits=(0, 8, 9))),
    (n1, dict(plaintext_space="low32-zero")), (n1, dict(plaintext_space="full")),
    (n1, dict(public_readout="zero-round-constant-residual")),
    (n1, dict(plaintext_fixed_value=True)), (n1, dict(plaintext_fixed_value=-1)),
    (n1, dict(plaintext_fixed_value=2**32)),
    (linear, dict(hidden_width=127)), (linear, dict(public_input="identity")),
    (linear, dict(public_readout="direct")), (linear, dict(l1_penalty=0.0)),
    (linear, dict(background_init="match-direct")), (linear, dict(rounds=3)),
])
def test_config_rejects_nonprotocol_combinations(builder: Any, changes: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        builder(**changes)


@pytest.mark.parametrize("key,values", [
    ("public_input", [None, 0, True, "phi", "IDENTITY", StringSubclass("identity")]),
    ("public_readout", [None, False, "residual", StringSubclass("direct")]),
    ("background_init", [None, 1, "matched", StringSubclass("standard")]),
    ("l1_penalty", [True, False, float("nan"), float("inf"), -float("inf"), "0", None, -1]),
    ("record_predictions", [0, 1, "false", None]),
])
def test_config_exact_new_field_types(key: str, values: list[Any]) -> None:
    for value in values:
        with pytest.raises((TypeError, ValueError)):
            config(**{key: value})


@pytest.mark.parametrize("kind", OLD_KINDS)
@pytest.mark.parametrize("changes", NONDEFAULT)
def test_every_old_kind_rejects_research_options_before_factory_dispatch(
    kind: str, changes: dict[str, Any],
) -> None:
    # Defaults intentionally invalid for some branches: match research error to
    # ensure rejection is before any branch-specific initialization/validation.
    with pytest.raises(ValueError, match="research|public_"):
        build_predictor(model_kind=kind, **changes)
    with pytest.raises(ValueError, match="research|public_"):
        ExperimentConfig(name="old", model_kind=kind, **changes)


@pytest.mark.parametrize("changes", NONDEFAULT)
def test_innovation1_rejects_research_controls(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(name="innovation1", experiment="innovation1", dataset="differential",
                         **changes)


def test_record_predictions_is_report_only_and_prediction_only() -> None:
    base = ExperimentConfig(name="record")
    recorded = replace(base, record_predictions=True)
    assert recorded.to_dict() == dict(base.to_dict(), record_predictions=True)
    assert recorded.sha256() != base.sha256()
    assert "record_predictions" not in inspect.signature(build_predictor).parameters
    for name in ("key_seed", "labels", "config"):
        assert name not in inspect.signature(build_predictor).parameters
    with pytest.raises(TypeError):
        build_predictor(record_predictions=True)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        ExperimentConfig(name="innovation1", experiment="innovation1", dataset="differential",
                         record_predictions=True)


def test_every_nondefault_research_field_is_in_identity() -> None:
    variants = [config(), config(public_input="simon-interactions"),
                config(public_readout="zero-round-constant-residual"), n1(),
                n1(background_init="match-direct"), linear(), config(record_predictions=True)]
    assert len({cfg.sha256() for cfg in variants}) == len(variants)
    for cfg in variants:
        raw = cfg.to_dict()
        for key, default in RESEARCH.items():
            if getattr(cfg, key) == default:
                assert key not in raw
            else:
                assert raw[key] == getattr(cfg, key)
        expected = hashlib.sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode())
        assert cfg.sha256() == expected.hexdigest()


@pytest.mark.parametrize("key,value", [("public_input", StringSubclass("identity")),
                                       ("public_readout", StringSubclass("direct")),
                                       ("background_init", StringSubclass("standard")),
                                       ("public_input", None), ("public_readout", False),
                                       ("background_init", 0), ("l1_penalty", "0"),
                                       ("l1_penalty", False), ("l1_penalty", True),
                                       ("l1_penalty", float("nan")),
                                       ("l1_penalty", float("inf"))])
def test_factory_rejects_malformed_default_like_research_options(key: str, value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_predictor(**{key: value})


@pytest.mark.parametrize("changes", [dict(feature_mode="support"),
                                     dict(prediction_mode="public-residual"),
                                     dict(prediction_mode="public-coordinate")])
def test_factory_public_models_reject_old_feature_controls(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        build_predictor(model_kind="public-mlp", rounds=6, **changes)


@pytest.mark.parametrize("public_input", ["identity", "simon-interactions"])
def test_factory_initialization_pairs_preserve_state_rng_and_parameter_sha(
    public_input: str,
) -> None:
    def make(readout: str) -> tuple[nn.Module, Tensor]:
        torch.manual_seed(17)
        m = build_predictor(**factory_options(config(public_input=public_input,
                                                     public_readout=readout)))
        return m, torch.get_rng_state().clone()
    direct, rng = make("direct")
    residual, residual_rng = make("zero-round-constant-residual")
    assert torch.equal(rng, residual_rng)
    assert list(direct.state_dict()) == list(residual.state_dict())
    for name, value in direct.state_dict().items():
        assert torch.equal(value, residual.state_dict()[name])
    if public_input == "identity":
        frozen = LEGACY[1]
        digest = hashlib.sha256(b"".join(
            name.encode() + value.numpy().tobytes()
            for name, value in direct.state_dict().items())).hexdigest()
        assert digest == frozen["state_sha256"]
        assert hashlib.sha256(rng.numpy().tobytes()).hexdigest() == frozen["rng_sha256"]


def test_factory_n1_r_z_identity_and_m_equivalence() -> None:
    models = []
    rngs = []
    for cfg in [replace(n1(), model_kind="mlp", public_input="identity"), n1(),
                n1(background_init="match-direct")]:
        torch.manual_seed(17)
        models.append(build_predictor(**factory_options(cfg)))
        rngs.append(torch.get_rng_state().clone())
    r, z, m = models
    for name, value in r.state_dict().items():
        assert torch.equal(value, z.state_dict()[name])
        if name != "layers.0.bias":
            assert torch.equal(value, m.state_dict()[name])
    assert all(torch.equal(rngs[0], rng) for rng in rngs[1:])
    fixed_bits = [(0xA5A5A5A5 >> shift) & 1 for shift in range(31, -1, -1)]
    x = torch.tensor([fixed_bits + [0, 1] * 16, fixed_bits + [1, 0] * 16]).float()
    torch.testing.assert_close(r(x), m(x), atol=1e-5, rtol=0)


@pytest.mark.parametrize("builder", [config, n1, linear,
                                     lambda: n1(background_init="match-direct"),
                                     lambda: config(public_input="simon-interactions"),
                                     lambda: config(public_readout="zero-round-constant-residual")])
@pytest.mark.parametrize("training", [False, True])
def test_metadata_actual_public_contract_no_forward_no_input_access_no_mutation(
    builder: Any, training: bool,
) -> None:
    cfg = builder()
    model = build_predictor(**factory_options(cfg))
    model.train(training)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()

    def forbid_forward(module: nn.Module, args: Any) -> None:
        raise AssertionError("metadata must not forward")
    handle = model.register_forward_pre_hook(forbid_forward)
    try:
        # Poison objects: public metadata must not inspect even TRAIN shapes.
        metadata = model_metadata(model, model_kind="not-authoritative", prediction_mode="direct",
                                  rounds=44, target_bit=63,
                                  x_train=None, y_train=None,  # type: ignore[arg-type]
                                  device=torch.device("cpu"))
    finally:
        handle.remove()
    assert metadata is not None
    assert metadata["kind"] == cfg.model_kind
    assert metadata["rounds"] == cfg.rounds and metadata["target_bit"] == 0
    assert metadata["public_input"] == cfg.public_input
    assert metadata["public_readout"] == cfg.public_readout
    assert metadata["background_init"] == cfg.background_init
    assert metadata["l1_penalty"] == cfg.l1_penalty
    assert metadata["input_width"] == (96 if cfg.public_input == "simon-interactions" else 64)
    reference = metadata["public_reference"]
    assert "zero master key" in reference or reference == "none"
    assert metadata["extra_labels"] == "none; absolute final target-bit labels only"
    assert metadata["selection"] == "absolute DEV BCE"
    assert metadata["representation"]
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [module.training for module in model.modules()]
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])
    for p, grad in zip(model.parameters(), grads, strict=True):
        assert p.grad is not None
        assert torch.equal(p.grad, grad)
    assert experiment_kind("simon", cfg.model_kind) == "simon-public-final-bit-supervision"


def test_legacy_phi_inverse_and_reference_modes_keep_exact_logits() -> None:
    # Independent literal MSB recurrence, no cipher/production helpers as oracle.
    x = torch.tensor([[((i * 37 + row * 11) % 13) < 6 for i in range(64)]
                      for row in range(7)]).float()
    bits = x.bool()
    high, low = bits[:, :32], bits[:, 32:]
    def f(word: Tensor) -> Tensor:
        return (word[:, [(i + 1) % 32 for i in range(32)]]
                & word[:, [(i + 8) % 32 for i in range(32)]]) ^ word[:, [
                    (i + 2) % 32 for i in range(32)]]
    phi = torch.cat((low ^ f(high), high), dim=1).float()
    assert torch.equal(public_coordinate_transform(x), phi)
    assert torch.equal(public_coordinate_inverse(phi), x)
    for rounds in range(1, 5):
        high, low = low ^ f(high), high
        for bit in (0, 17, 31, 32, 63):
            expected = torch.cat((high, low), dim=1)[:, bit:bit + 1].float()
            assert torch.equal(public_reference(x, rounds, bit), expected)
            model = PlaintextPredictor(8, rounds=rounds, target_bit=bit,
                                       prediction_mode="public-residual")
            raw = model.layers(x)
            assert torch.equal(model(x), (1 - 2 * expected) * raw)
    model = PlaintextPredictor(8, rounds=6, prediction_mode="public-coordinate")
    assert torch.equal(model(x), model.layers(phi))


@pytest.mark.parametrize("kind,options", [
    ("mlp", {}), ("logic16", dict(hidden_width=64)),
    ("circuit-aligned", dict(dataset="circuit-synthetic", hidden_width=64)),
    ("circuit-random", dict(dataset="circuit-synthetic", hidden_width=64)),
    ("simon-structure", dict(hidden_width=64)), ("simon-unrolled", {}),
    ("simon-r2-table", dict(rounds=2, hidden_width=16)),
    ("simon-r2-local", dict(rounds=2, hidden_width=16)),
    ("simon-r2-vector", dict(rounds=2, hidden_width=16)),
    ("simon-cnn", dict(rounds=6)), ("simon-rotation-cnn", dict(rounds=6)),
    ("simon-vector", dict(rounds=6)),
])
def test_record_predictions_available_to_every_legacy_prediction_kind(
    kind: str, options: dict[str, Any],
) -> None:
    cfg = ExperimentConfig(**dict(dict(name="old-valid", dataset="simon", model_kind=kind),
                                  **options))
    assert replace(cfg, **RESEARCH).to_dict() == cfg.to_dict()
    assert replace(cfg, record_predictions=True).to_dict() == dict(
        cfg.to_dict(), record_predictions=True)
    for changes in NONDEFAULT:
        with pytest.raises(ValueError, match="research|public_"):
            replace(cfg, **changes)


@pytest.mark.parametrize("kind", ["logic16", "simon-unrolled", "public-linear"])
@pytest.mark.parametrize("changes", [dict(plaintext_space="high32-zero"),
                                     dict(label_expression="xor2", label_bits=(0, 8))])
def test_nonfull_nonbit_exception_does_not_relax_other_models(
    kind: str, changes: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        if kind == "public-linear":
            linear(**changes)
        else:
            ExperimentConfig(name="restricted", dataset="simon", model_kind=kind,
                             hidden_width=64, **changes)


@pytest.mark.parametrize("changes", [dict(public_input="simon-interactions"),
                                     dict(public_readout="zero-round-constant-residual"),
                                     dict(background_init="match-direct"),
                                     dict(l1_penalty=float("nan"))])
def test_legacy_factory_rejection_consumes_no_rng(changes: dict[str, Any]) -> None:
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        build_predictor(**changes)
    assert torch.equal(before, torch.get_rng_state())


def test_legacy_report_default_and_coordinate_metadata_stay_unchanged() -> None:
    model = PlaintextPredictor(8, rounds=6)
    options: dict[str, Any] = dict(model_kind="mlp", prediction_mode="direct", rounds=6,
                                   target_bit=0, x_train=None, y_train=None,
                                   device=torch.device("cpu"))
    rng = torch.get_rng_state().clone()
    assert model_metadata(model, **options) is None
    metadata = model_metadata(model, **dict(options, prediction_mode="public-coordinate"))
    assert metadata == {
        "kind": "mlp", "prediction_mode": "public-coordinate",
        "coordinates": "(Y xor ((ROL1(X) & ROL8(X)) xor ROL2(X)), X)",
        "bit_order": "64-bit plaintext MSB index 0..63; X=0..31, Y=32..63",
        "interpretation": "public bijective coordinates only; not actual cipher state",
        "learned_constants": "none",
        "extra_labels": "none; absolute final target-bit labels only",
        "input": "plaintext only; no actual key, trace, ciphertext, or other labels",
        "parameter_learning": "ordinary MLP weights from TRAIN labels; no key injection",
        "selection": "absolute DEV BCE",
    }
    assert experiment_kind("simon", "mlp") == "simon-output-bit-baseline"
    assert experiment_kind("synthetic", "mlp") == "synthetic-engineering"
    assert torch.equal(rng, torch.get_rng_state())

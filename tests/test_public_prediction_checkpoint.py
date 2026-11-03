"""Artificial state only: strict public reconstruction, identity and corruption guards."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from cipher_experiments import checkpoints
from cipher_experiments.artifacts import file_sha256
from cipher_experiments.config import ExperimentConfig
from cipher_experiments.models.prediction import PlaintextPredictor, build_predictor
from cipher_experiments.models.public_prediction import PublicPredictor


def config(**changes: Any) -> ExperimentConfig:
    options: dict[str, Any] = dict(name="manual-public", dataset="simon", rounds=6,
                                   hidden_width=128, model_kind="public-mlp", label_mode="null")
    options.update(changes)
    return ExperimentConfig(**options)


def background(**changes: Any) -> ExperimentConfig:
    return config(**dict(dict(rounds=3, public_input="background-zero",
                              plaintext_space="high32-zero", plaintext_fixed_value=0xA5A5A5A5,
                              label_expression="xor2", label_bits=(0, 8)), **changes))


def make_model(cfg: ExperimentConfig) -> PublicPredictor | PlaintextPredictor:
    model = build_predictor(
        cfg.hidden_width, model_kind=cfg.model_kind, rounds=cfg.rounds, target_bit=cfg.target_bit,
        feature_mode=cfg.model_features, prediction_mode=cfg.prediction_mode,
        public_input=cfg.public_input, public_readout=cfg.public_readout,
        background_init=cfg.background_init, l1_penalty=cfg.l1_penalty,
        plaintext_space=cfg.plaintext_space, plaintext_fixed_value=cfg.plaintext_fixed_value)
    assert isinstance(model, (PublicPredictor, PlaintextPredictor))
    return model


def save(path: Path, cfg: ExperimentConfig) -> PublicPredictor | PlaintextPredictor:
    torch.manual_seed(19)
    model = make_model(cfg)
    checkpoints.save_checkpoint(path, model, cfg, epoch=1, updates=0,
                                majority=1, probability=0.6)
    return model


CANDIDATES = [
    background(), background(background_init="match-direct"), config(),
    config(public_input="simon-interactions"),
    config(public_readout="zero-round-constant-residual"),
    config(public_input="simon-interactions", public_readout="zero-round-constant-residual"),
    config(model_kind="public-linear", public_input="simon-interactions",
           public_readout="zero-round-constant-residual", l1_penalty=1e-4),
]


@pytest.mark.parametrize("cfg", CANDIDATES)
def test_public_save_strict_load_preserves_state_config_and_logits(
    tmp_path: Path, cfg: ExperimentConfig,
) -> None:
    # Dropping any factory option breaks a supported candidate or its saved semantics.
    path = tmp_path / "manual.pt"
    original = save(path, cfg)
    bits = [(cfg.plaintext_fixed_value >> shift) & 1 for shift in range(31, -1, -1)]
    x = torch.tensor([bits + [0, 1] * 16, bits + [1, 0] * 16]).float()
    if cfg.plaintext_space == "full":
        x[1, :32] = 1 - x[1, :32]
    for _ in range(2):
        loaded, restored, majority, probability = checkpoints.load_checkpoint(
            path, file_sha256(path), device=torch.device("cpu"))
        assert isinstance(loaded, PublicPredictor)
        assert restored == cfg
        assert loaded.checkpoint_config_sha256 == cfg.sha256()
        assert (majority, probability) == (1, 0.6)
        assert not loaded.training
        assert loaded.public_input == cfg.public_input
        assert loaded.public_readout == cfg.public_readout
        assert loaded.background_init == cfg.background_init
        assert loaded.l1_penalty == cfg.l1_penalty
        assert loaded.plaintext_fixed_value == cfg.plaintext_fixed_value
        for name, value in original.state_dict().items():
            assert torch.equal(loaded.state_dict()[name], value)
        # Exact equality: adding match-direct compensation after load is forbidden.
        assert torch.equal(loaded(x), original(x))
        checkpoints.save_checkpoint(path, loaded, cfg, epoch=1, updates=0,
                                    majority=majority, probability=probability)


def test_legacy_r_and_matched_m_remain_function_equivalent_on_repeated_load(tmp_path: Path) -> None:
    cfg = background()
    r_cfg = config(**dict(cfg.to_dict(), model_kind="mlp", public_input="identity"))
    m_cfg = background(background_init="match-direct")
    r = save(tmp_path / "r.pt", r_cfg)
    m = save(tmp_path / "m.pt", m_cfg)
    fixed = [(0xA5A5A5A5 >> shift) & 1 for shift in range(31, -1, -1)]
    x = torch.tensor([fixed + [0, 1] * 16, fixed + [1, 0] * 16]).float()
    torch.testing.assert_close(r(x), m(x), atol=1e-5, rtol=0)
    for _ in range(2):
        for name, original in (("r", r), ("m", m)):
            path = tmp_path / f"{name}.pt"
            loaded, _, _, _ = checkpoints.load_checkpoint(
                path, file_sha256(path), device=torch.device("cpu"))
            assert torch.equal(loaded(x), original(x))


@pytest.mark.parametrize("change", [
    {"plaintext_fixed_value": 0x5A5A5A5A}, {"background_init": "match-direct"},
    {"public_input": "simon-interactions"},
    {"public_readout": "zero-round-constant-residual"},
])
def test_changed_public_config_cannot_reuse_recorded_raw_sha(
    tmp_path: Path, change: dict[str, Any],
) -> None:
    cfg = (background() if "plaintext_fixed_value" in change or "background_init" in change
           else config())
    path = tmp_path / "identity.pt"
    save(path, cfg)
    payload = torch.load(path, weights_only=True)
    payload["config"].update(change)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="configuration SHA"):
        checkpoints.load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))


@pytest.mark.filterwarnings("ignore:Validating sparse tensor invariants:UserWarning")
@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "dtype", "nan", "layout"])
def test_public_state_corruption_rejected_before_strict_load(tmp_path: Path, damage: str) -> None:
    path = tmp_path / "damaged.pt"
    save(path, CANDIDATES[-1])
    payload = torch.load(path, weights_only=True)
    state = payload["state_dict"]
    key = "layers.0.weight"
    if damage == "missing":
        del state[key]
    elif damage == "extra":
        state["unknown"] = torch.zeros(1)
    elif damage == "shape":
        state[key] = state[key][:, :-1]
    elif damage == "dtype":
        state[key] = state[key].double()
    elif damage == "nan":
        state[key][0, 0] = float("nan")
    else:
        state[key] = state[key].to_sparse()
    torch.save(payload, path)
    with pytest.raises(ValueError, match="state"):
        checkpoints.load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))


def test_checkpoint_file_sha_schema_and_weights_only_stay_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "strict.pt"
    save(path, config())
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        checkpoints.load_checkpoint(path, "0" * 64, device=torch.device("cpu"))
    real_load = torch.load
    calls: list[dict[str, Any]] = []

    def tracked_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    checkpoints.load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))
    assert calls == [{"weights_only": True, "map_location": "cpu"}]
    payload = real_load(path, weights_only=True)
    assert payload["schema"] == "cipher-experiments-checkpoint-v1"
    assert set(payload) == {"schema", "config", "config_sha256", "state_dict", "epoch",
                            "updates", "majority", "probability"}
    payload["schema"] = "cipher-experiments-checkpoint-v2"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="schema"):
        checkpoints.load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))


def test_legacy_sparse_raw_mapping_sha_is_not_replaced_by_additive_defaults(tmp_path: Path) -> None:
    # A genuinely pre-additive mapping, not a freshly serialized current config.
    raw = {"name": "sparse-legacy", "hidden_width": 8}
    raw_sha = hashlib.sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode()).hexdigest()
    model = PlaintextPredictor(8)
    payload = dict(schema="cipher-experiments-checkpoint-v1", config=raw,
                   config_sha256=raw_sha, state_dict=dict(model.state_dict()), epoch=0, updates=0,
                   majority=0, probability=0.5)
    path = tmp_path / "old.pt"
    torch.save(payload, path)
    loaded, cfg, _, _ = checkpoints.load_checkpoint(
        path, file_sha256(path), device=torch.device("cpu"))
    assert isinstance(loaded, PlaintextPredictor)
    assert loaded.checkpoint_config_sha256 == raw_sha
    assert cfg.sha256() != raw_sha
    x = torch.zeros(2, 64)
    assert torch.equal(loaded(x), model(x))
    assert all(isinstance(value, Tensor) for value in loaded.state_dict().values())

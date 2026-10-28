"""Serialization and strict loading for prediction checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from .artifacts import file_sha256
from .config import ExperimentConfig
from .models.circuit import CircuitPredictor
from .models.logic import LogicPredictor
from .models.prediction import PlaintextPredictor, build_predictor
from .models.simon_conv import SimonConvPredictor
from .models.simon_two_round import (
    TwoRoundLocalPredictor,
    TwoRoundTablePredictor,
    TwoRoundVectorPredictor,
)
from .models.structured_simon import StructuredSimonPredictor
from .models.unrolled_simon import UnrolledSimonPredictor

CHECKPOINT_SCHEMA = "cipher-experiments-checkpoint-v1"


def save_checkpoint(path: Path, model: nn.Module, config: ExperimentConfig,
                    epoch: int, updates: int, majority: int, probability: float) -> None:
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    payload = {"schema": CHECKPOINT_SCHEMA, "config": config.to_dict(),
               "config_sha256": config.sha256(), "state_dict": state,
               "epoch": epoch, "updates": updates,
               "majority": majority, "probability": probability}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, expected_sha256: str, *, device: torch.device
                    ) -> tuple[
                        PlaintextPredictor | LogicPredictor | CircuitPredictor
                        | StructuredSimonPredictor | UnrolledSimonPredictor
                        | TwoRoundTablePredictor | TwoRoundLocalPredictor
                        | TwoRoundVectorPredictor | SimonConvPredictor,
                        ExperimentConfig, int, float]:
    if len(expected_sha256) != 64 or file_sha256(path) != expected_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("checkpoint exceeds 256 MiB")
    payload = torch.load(path, weights_only=True, map_location="cpu")
    expected = {"schema", "config", "config_sha256", "state_dict", "epoch", "updates",
                "majority", "probability"}
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("checkpoint schema fields mismatch")
    if payload["schema"] != CHECKPOINT_SCHEMA or type(payload["config"]) is not dict:
        raise ValueError("checkpoint schema mismatch")
    config = ExperimentConfig(**payload["config"])
    # Bind the saved raw mapping, so checkpoints predating additive default fields
    # remain loadable without inventing a different recorded configuration SHA.
    raw_sha = hashlib.sha256(json.dumps(payload["config"], sort_keys=True,
                                        allow_nan=False).encode()).hexdigest()
    if raw_sha != payload["config_sha256"]:
        raise ValueError("checkpoint configuration SHA mismatch")
    if config.experiment != "prediction":
        raise ValueError("not a prediction checkpoint")
    majority, probability = payload["majority"], payload["probability"]
    if type(majority) is not int or majority not in (0, 1):
        raise ValueError("invalid saved TRAIN majority")
    if type(probability) is not float or not 0 < probability < 1:
        raise ValueError("invalid saved TRAIN probability")
    model = build_predictor(config.hidden_width, model_kind=config.model_kind,
                            rounds=config.rounds,
                               target_bit=config.target_bit, feature_mode=config.model_features,
                               prediction_mode=config.prediction_mode)
    template = model.state_dict()
    state = payload["state_dict"]
    if type(state) is not dict or set(state) != set(template):
        raise ValueError("checkpoint state names mismatch")
    for key, target in template.items():
        value = state[key]
        if (type(value) is not Tensor or value.shape != target.shape
                or value.dtype != target.dtype or value.layout != torch.strided
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"checkpoint state mismatch: {key}")
    model.load_state_dict(state, strict=True)
    if isinstance(model, (LogicPredictor, CircuitPredictor, SimonConvPredictor)):
        model.validate_wiring()
    model.checkpoint_config_sha256 = raw_sha
    model.to(device)
    model.eval()
    return model, config, majority, probability

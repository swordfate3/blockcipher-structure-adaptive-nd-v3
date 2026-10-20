"""Binary original-bit metrics and strict checkpoint inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .artifacts import file_sha256
from .config import ExperimentConfig
from .models.logic import LogicPredictor
from .models.prediction import PlaintextPredictor, build_predictor

CHECKPOINT_SCHEMA = "cipher-experiments-checkpoint-v1"


@torch.no_grad()
def evaluate_model(model: nn.Module, x: Tensor, y: Tensor, *, device: torch.device,
                   batch_size: int, majority: int, probability: float,
                   reference: Tensor | None = None, residual_majority: int = 0,
                   residual_probability: float = .5) -> dict[str, Any]:
    correct, count, loss = 0, len(y), 0.0
    constant_correct = 0
    public_correct = 0
    public_loss = 0.0
    probability = min(max(probability, 1e-12), 1 - 1e-12)
    import math
    constant_loss = 0.0
    model.eval()
    for start in range(0, count, batch_size):
        targets = y[start:start + batch_size].to(device)
        logits = model(x[start:start + batch_size].to(device))
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite evaluation logits")
        loss += float(nn.functional.binary_cross_entropy_with_logits(logits, targets,
                                                                    reduction="sum"))
        correct += int(((logits > 0) == targets.bool()).sum())
        ones = float(targets.sum())
        constant_correct += int((targets == majority).sum())
        constant_loss -= ones * math.log(probability) + (len(targets) - ones) * math.log1p(
            -probability)
        if reference is not None:
            ref = reference[start:start + batch_size].to(device)
            public_correct += int(((ref.bool() ^ bool(residual_majority)) == targets.bool()).sum())
            p = torch.where(ref.bool(), 1 - residual_probability, residual_probability).double()
            public_loss += float(torch.nn.functional.binary_cross_entropy(
                p, targets.double(), reduction="sum"))
    result = {"rows": count, "correct": correct, "accuracy": correct / count,
              "bce": loss / count,
              "train_constant": {"bit": majority, "probability": probability,
                                 "accuracy": constant_correct / count,
                                 "bce": constant_loss / count}}
    if reference is not None:
        result["public_reference_constant"] = {
            "residual_bit": residual_majority, "residual_probability": residual_probability,
            "accuracy": public_correct / count, "bce": public_loss / count}
    return result


def load_checkpoint(path: Path, expected_sha256: str, *, device: torch.device
                    ) -> tuple[PlaintextPredictor | LogicPredictor, ExperimentConfig, int, float]:
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
    if isinstance(model, LogicPredictor):
        model.validate_wiring()
    model.checkpoint_config_sha256 = raw_sha
    model.to(device)
    model.eval()
    return model, config, majority, probability

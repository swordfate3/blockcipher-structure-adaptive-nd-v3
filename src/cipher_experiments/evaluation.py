"""Binary original-bit metrics and strict checkpoint inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .artifacts import file_sha256
from .config import ExperimentConfig
from .models.mlp import BitMLP

CHECKPOINT_SCHEMA = "cipher-experiments-checkpoint-v1"


@torch.no_grad()
def evaluate_model(model: nn.Module, x: Tensor, y: Tensor, *, device: torch.device,
                   batch_size: int, majority: int, probability: float) -> dict[str, Any]:
    correct, count, loss = 0, len(y), 0.0
    constant_correct = 0
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
    return {"rows": count, "correct": correct, "accuracy": correct / count,
            "bce": loss / count,
            "train_constant": {"bit": majority, "probability": probability,
                               "accuracy": constant_correct / count,
                               "bce": constant_loss / count}}


def load_checkpoint(path: Path, expected_sha256: str, *, device: torch.device
                    ) -> tuple[BitMLP, ExperimentConfig, int, float]:
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
    if config.sha256() != payload["config_sha256"]:
        raise ValueError("checkpoint configuration SHA mismatch")
    majority, probability = payload["majority"], payload["probability"]
    if type(majority) is not int or majority not in (0, 1):
        raise ValueError("invalid saved TRAIN majority")
    if type(probability) is not float or not 0 < probability < 1:
        raise ValueError("invalid saved TRAIN probability")
    model = BitMLP(config.hidden_width)
    template = model.state_dict()
    state = payload["state_dict"]
    if type(state) is not dict or set(state) != set(template):
        raise ValueError("checkpoint state names mismatch")
    for key, target in template.items():
        value = state[key]
        if (type(value) is not Tensor or value.shape != target.shape
                or value.dtype != target.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError(f"checkpoint state mismatch: {key}")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, config, majority, probability

"""Binary original-bit metrics, with legacy checkpoint import compatibility."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .checkpoints import CHECKPOINT_SCHEMA as CHECKPOINT_SCHEMA
from .checkpoints import load_checkpoint as load_checkpoint
from .models.structured_simon import StructuredSimonPredictor


@torch.no_grad()
def evaluate_model(model: nn.Module, x: Tensor, y: Tensor, *, device: torch.device,
                   batch_size: int, majority: int, probability: float,
                   reference: Tensor | None = None, residual_majority: int = 0,
                   residual_probability: float = .5) -> dict[str, Any]:
    correct, count, loss = 0, len(y), 0.0
    constant_correct = 0
    public_correct = 0
    public_loss = 0.0
    hard_correct, hard_loss = 0, 0.0
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
        if isinstance(model, StructuredSimonPredictor):
            hard_logits = model.hard_logits(x[start:start + batch_size].to(device))
            hard_correct += int(((hard_logits > 0) == targets.bool()).sum())
            hard_loss += float(nn.functional.binary_cross_entropy_with_logits(
                hard_logits, targets, reduction="sum"))
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
    if isinstance(model, StructuredSimonPredictor):
        result["hard_constants"] = {"rows": count, "correct": hard_correct,
                                    "accuracy": hard_correct / count, "bce": hard_loss / count}
    if reference is not None:
        result["public_reference_constant"] = {
            "residual_bit": residual_majority, "residual_probability": residual_probability,
            "accuracy": public_correct / count, "bce": public_loss / count}
    return result

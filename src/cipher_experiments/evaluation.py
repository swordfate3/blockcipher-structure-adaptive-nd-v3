"""Binary original-bit metrics, with legacy checkpoint import compatibility."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .checkpoints import CHECKPOINT_SCHEMA as CHECKPOINT_SCHEMA
from .checkpoints import load_checkpoint as load_checkpoint
from .models.structured_simon import StructuredSimonPredictor
from .prediction_evidence import PredictionCapture


@torch.no_grad()
def evaluate_model(model: nn.Module, x: Tensor, y: Tensor, *, device: torch.device,
                   batch_size: int, majority: int, probability: float,
                   reference: Tensor | None = None, residual_majority: int = 0,
                   residual_probability: float = .5,
                   capture: PredictionCapture | None = None,
                   zero_constant_reference: Tensor | None = None,
                   zero_constant_majority: int = 0,
                   zero_constant_probability: float = .5) -> dict[str, Any]:
    if zero_constant_reference is not None:
        ref = zero_constant_reference
        if not isinstance(ref, Tensor) or ref.dtype != torch.float32:
            raise TypeError("zero_constant_reference must be a float32 tensor")
        if ref.layout != torch.strided or ref.shape != (len(y), 1):
            raise ValueError("zero_constant_reference must have shape [rows,1]")
        if not bool(torch.isfinite(ref).all()) or not bool(((ref == 0) | (ref == 1)).all()):
            raise ValueError("zero_constant_reference must be finite and binary")
        if type(zero_constant_majority) is not int:
            raise TypeError("zero_constant_majority must be a built-in int")
        if zero_constant_majority not in (0, 1):
            raise ValueError("zero_constant_majority must be binary")
        if type(zero_constant_probability) not in (int, float):
            raise TypeError("zero_constant_probability must be a built-in number")
        if not math.isfinite(zero_constant_probability) or not 0 < zero_constant_probability < 1:
            raise ValueError("zero_constant_probability must be finite and strictly inside (0,1)")
    zero_correct, zero_loss = 0, 0.0
    correct, count, loss = 0, len(y), 0.0
    constant_correct = 0
    public_correct = 0
    public_loss = 0.0
    hard_correct, hard_loss = 0, 0.0
    probability = min(max(probability, 1e-12), 1 - 1e-12)
    constant_loss = 0.0
    model.eval()
    for start in range(0, count, batch_size):
        targets = y[start:start + batch_size].to(device)
        logits = model(x[start:start + batch_size].to(device))
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite evaluation logits")
        if capture is not None:
            capture.append(logits, targets)
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
        if zero_constant_reference is not None:
            zero_ref = zero_constant_reference[start:start + batch_size].to(device).bool()
            zero_correct += int(((zero_ref ^ bool(zero_constant_majority)) == targets.bool()).sum())
            # Float64 probabilities only affect BCE, never reconstruct the majority decision.
            zero_p = torch.full(zero_ref.shape, zero_constant_probability,
                                dtype=torch.float64, device=device)
            zero_p = torch.where(zero_ref, 1 - zero_p, zero_p)
            zero_loss += float(nn.functional.binary_cross_entropy(
                zero_p, targets.double(), reduction="sum"))
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
    if zero_constant_reference is not None:
        result["zero_round_constant_reference_constant"] = {
            "residual_bit": zero_constant_majority,
            "residual_probability": zero_constant_probability,
            "accuracy": zero_correct / count, "bce": zero_loss / count}
    return result

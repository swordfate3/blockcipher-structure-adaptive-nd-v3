"""Read-only prediction metadata from explicit model and TRAIN inputs.

No filesystem, label generation, optimizer, selection or TEST access belongs here.
The caller decides when to collect the report and where to persist it.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .models.simon_two_round import (
    TwoRoundLocalPredictor,
    TwoRoundTablePredictor,
    TwoRoundVectorPredictor,
)
from .models.structured_simon import StructuredSimonPredictor
from .models.unrolled_simon import UnrolledSimonPredictor


def experiment_kind(dataset: str, model_kind: str) -> str:
    """Describe the existing experiment role without dispatching execution."""
    if dataset in ("synthetic", "circuit-synthetic"):
        return "synthetic-engineering"
    if model_kind == "simon-structure":
        return "simon-structure-unknown-constants"
    if model_kind == "simon-unrolled":
        return "simon-unrolled-final-bit-supervision"
    if model_kind == "simon-r2-table":
        return "simon-r2-table-positive-control"
    if model_kind == "simon-r2-local":
        return "simon-r2-local-neural-candidate"
    if model_kind == "simon-r2-vector":
        return "simon-r2-vector-neural-candidate"
    return "simon-output-bit-baseline"


@torch.no_grad()
def _table_coverage(model: TwoRoundTablePredictor, x_train: Tensor,
                    y_train: Tensor, device: torch.device) -> dict[str, Any]:
    cells, public_t = model.public_table_features(x_train.to(device))
    residual = torch.logical_xor(
        y_train.to(device).bool().squeeze(1), public_t.bool().squeeze(1)
    )
    cell_counts: list[int] = []
    residual_ones: list[int] = []
    conflicts: list[bool] = []
    for cell in range(4):
        cell_mask = cells == cell
        count = int(cell_mask.sum())
        ones = int((residual & cell_mask).sum())
        cell_counts.append(count)
        residual_ones.append(ones)
        conflicts.append(ones not in (0, count))
    return {
        "counts": cell_counts,
        "residual_ones": residual_ones,
        "covered_all_cells": sum(count > 0 for count in cell_counts) == 4,
        "conflicting_cells": [cell for cell, conflict in enumerate(conflicts) if conflict],
    }


def model_metadata(model: nn.Module, *, model_kind: str, prediction_mode: str,
                   rounds: int, target_bit: int, x_train: Tensor, y_train: Tensor,
                   device: torch.device) -> dict[str, Any] | None:
    """Return existing report fields, or None for models without a metadata file.

    Only the table control inspects the supplied TRAIN tensors. This function
    does not change parameters, gradients, module modes or random generators.
    """
    if prediction_mode == "public-coordinate":
        return {
            "kind": "mlp",
            "prediction_mode": "public-coordinate",
            "coordinates": "(Y xor ((ROL1(X) & ROL8(X)) xor ROL2(X)), X)",
            "bit_order": "64-bit plaintext MSB index 0..63; X=0..31, Y=32..63",
            "interpretation": "public bijective coordinates only; not actual cipher state",
            "learned_constants": "none",
            "extra_labels": "none; absolute final target-bit labels only",
            "input": "plaintext only; no actual key, trace, ciphertext, or other labels",
            "parameter_learning": "ordinary MLP weights from TRAIN labels; no key injection",
            "selection": "absolute DEV BCE",
        }
    if isinstance(model, StructuredSimonPredictor):
        return {
            "kind": "simon-structure", "rounds": rounds,
            "target_bit": target_bit,
            "unknowns": "independent learned round-constant effects; not true key input",
            "initialization": "random signs and noise from init_seed, no data/key access",
            "key_schedule_constraint": False,
            "selection": "soft absolute DEV BCE; hard constants report-only",
            "gate_counts": model.cone.gate_counts,
            "symbolic_coordinates": list(model.cone.round_key_coordinates),
        }
    if isinstance(model, (TwoRoundTablePredictor, TwoRoundLocalPredictor,
                          TwoRoundVectorPredictor)):
        metadata: dict[str, Any] = {
            "kind": model_kind,
            "role": "positive-control" if isinstance(model, TwoRoundTablePredictor)
                    else "neural-candidate",
            "rounds": 2, "target_bit": target_bit,
            "supervision": "absolute final target-bit BCE only",
            "key_free": True,
            "input": "public plaintext bits only; no key or intermediate states; "
                     "final absolute bit labels only",
            "selection": "absolute DEV BCE",
        }
        if isinstance(model, TwoRoundTablePredictor):
            metadata.update({
                "parameters": 4,
                "formula": "U=Y XOR F(X); A=U[j+1], B=U[j+8], "
                           "T=X[j] XOR U[j+2]; logit=(1-2T)*table_logits[2A+B]",
                "train_table": _table_coverage(model, x_train, y_train, device),
            })
        elif isinstance(model, TwoRoundVectorPredictor):
            metadata.update({
                "parameters": 557,
                "modules": "three independent 4->16->4 local modules with Tanh hidden "
                           "activation only, then 12->8->1 with Tanh hidden activation",
                "local_features": "centered [P_Y[q],P_X[q+1],P_X[q+8],P_X[q+2]], "
                                  "q=j+1,j+8,j+2",
                "hidden_channels": "sigmoid per local logit then center; concatenate 12",
                "output_sign": "(1-2*P_X[j])",
                "parameter_sharing": "independent local module weights; no sharing",
            })
        else:
            metadata.update({
                "parameters": 332,
                "modules": "three independent 4->16->1 Tanh MLPs, then 3->8->1 Tanh",
                "local_features": "centered [P_Y[q],P_X[q+1],P_X[q+8],P_X[q+2]], "
                                  "q=j+1,j+8,j+2",
                "output_sign": "(1-2*P_X[j])",
                "parameter_sharing": "independent local module weights; no sharing",
            })
        return metadata
    if isinstance(model, UnrolledSimonPredictor):
        return {
            "kind": "simon-unrolled", "rounds": rounds,
            "target_bit": target_bit, "prediction_mode": prediction_mode,
            "supervision": "final-only absolute target-bit BCE; no real intermediate states",
            "public_reference": (
                "plaintext target bit at target_bit in the high half (indices 0..31); "
                "sign factor (1-2*x[target_bit])"
                if prediction_mode in ("plaintext-xor", "plaintext-xor-no-final-low")
                else "none (direct mode)"),
            "final_low_input": (
                "zero after centering for every position in the final round only"
                if prediction_mode == "plaintext-xor-no-final-low"
                else "centered local low-state value in every round"),
            "input": "plaintext only; no true key or round-key input",
            "public_prior": "Feistel copy and rotations 1,8,2; local neural update",
            "state": "continuous latent scalars, not verified cipher round states",
            "cell": "shared 8->hidden_width->1 Tanh MLP, sigmoid hidden updates",
            "conditions": "learned [round,position,4], randomly initialized",
            "key_schedule_constraint": False,
            "selection": "absolute DEV BCE; no discrete parameter search",
        }
    return None

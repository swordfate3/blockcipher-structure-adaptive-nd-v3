"""Key-free diagnostics for selected two-round local predictors.

Only caller-supplied TRAIN and DEV panels are accepted. The reports aggregate
public-feature groups and model errors; they never expose individual examples.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models.simon_two_round import TwoRoundLocalPredictor, TwoRoundVectorPredictor


@dataclass(slots=True)
class _Counts:
    rows: int = 0
    errors: int = 0
    zero_labels: int = 0
    one_labels: int = 0

    def add(self, *, error: bool, label: int) -> None:
        self.rows += 1
        self.errors += int(error)
        if label:
            self.one_labels += 1
        else:
            self.zero_labels += 1


def _group_summary(groups: Mapping[tuple[int, ...], _Counts]) -> dict[str, int]:
    return {
        "groups": len(groups),
        "rows": sum(count.rows for count in groups.values()),
        "errors": sum(count.errors for count in groups.values()),
        "label_zero_rows": sum(count.zero_labels for count in groups.values()),
        "label_one_rows": sum(count.one_labels for count in groups.values()),
        "unique_label_groups": sum(
            count.zero_labels == 0 or count.one_labels == 0 for count in groups.values()
        ),
        "conflicting_label_groups": sum(
            count.zero_labels > 0 and count.one_labels > 0 for count in groups.values()
        ),
    }


def _validate_panel(name: str, panel: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    if not isinstance(panel, tuple) or len(panel) != 2:
        raise TypeError(f"{name} panel must be an (inputs, labels) tuple")
    inputs, labels = panel
    for tensor, tensor_name, shape in (
        (inputs, "inputs", (None, 64)), (labels, "labels", (None, 1))
    ):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} {tensor_name} must be a torch.Tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} {tensor_name} must be on CPU")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} {tensor_name} must have dtype torch.float32")
        if tensor.requires_grad:
            raise ValueError(f"{name} {tensor_name} must be detached")
        if tensor.ndim != 2 or tensor.shape[1] != shape[1]:
            raise ValueError(f"{name} {tensor_name} must have shape [N,{shape[1]}]")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} {tensor_name} must contain only finite values")
        if not bool(((tensor == 0) | (tensor == 1)).all()):
            raise ValueError(f"{name} {tensor_name} must contain only binary values")
    if inputs.shape[0] == 0 or inputs.shape[0] != labels.shape[0]:
        raise ValueError(f"{name} inputs and labels must have the same nonzero row count")
    return inputs, labels


def _take_local_logits(
    captured: dict[int, Tensor], index: int, rows: int, channels: int
) -> Tensor:
    logits = captured.pop(index, None)
    if (logits is None or logits.shape != (rows, channels)
            or not bool(torch.isfinite(logits).all())):
        raise ValueError(f"local module did not produce finite [N,{channels}] logits")
    return logits.reshape(rows, channels)


def _corner_summaries(
    model: TwoRoundLocalPredictor | TwoRoundVectorPredictor,
    captured: dict[int, Tensor],
    device: torch.device,
) -> list[dict[str, Any]]:
    channels = 1 if isinstance(model, TwoRoundLocalPredictor) else 4
    summaries: list[dict[str, Any]] = []
    corners = torch.tensor(
        list(itertools.product((0.0, 1.0), repeat=4)), dtype=torch.float32, device=device
    )
    centered = corners.mul(2.0).sub(1.0)
    for index, module in enumerate(model.local_modules):
        module(centered)
        logits = _take_local_logits(captured, index, 16, channels)
        probabilities = torch.sigmoid(logits).tolist()
        channel_summaries: list[dict[str, float | int]] = []
        for channel in range(channels):
            values = [float(row[channel]) for row in probabilities]
            summary: dict[str, float | int] = {
                "corner_count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
            if not all(math.isfinite(float(value)) for value in summary.values()):
                raise ValueError("nonfinite local corner diagnostic")
            channel_summaries.append(summary)
        if channels == 1:
            summaries.append(channel_summaries[0])
        else:
            summaries.append({
                "channel_count": channels,
                "channels": channel_summaries,
            })
    return summaries


def two_round_local_diagnostics(
    model: TwoRoundLocalPredictor | TwoRoundVectorPredictor,
    panels: Mapping[str, tuple[Tensor, Tensor]],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Summarize selected-model TRAIN/DEV behavior without reading TEST or cipher state.

    The local sigmoid corner values are learned activations, not verified cipher
    states. Model parameters, RNG state, input panels, and each module's original
    training flag are preserved. Hooks are removed even if inference fails.
    """
    if not isinstance(model, (TwoRoundLocalPredictor, TwoRoundVectorPredictor)):
        raise TypeError("diagnostics require a two-round local or vector predictor")
    if set(panels) != {"train", "dev"}:
        raise ValueError("diagnostics accept exactly TRAIN and DEV panels; TEST is forbidden")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive built-in int")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    parameters = tuple(model.parameters())
    if not parameters or any(parameter.device != device for parameter in parameters):
        raise ValueError("model parameters and diagnostics device must match")
    if model.indices.device != device:
        raise ValueError("model public indices and diagnostics device must match")

    validated = {split: _validate_panel(split, panels[split]) for split in ("train", "dev")}
    indices = [int(index) for index in model.indices.detach().cpu().tolist()]
    if (not indices or len(indices) != len(set(indices))
            or any(not 0 <= index < 64 for index in indices)):
        raise ValueError("model.indices must contain unique public input positions")

    target_bit = model.target_bit
    q_positions = tuple((target_bit + offset) % 32 for offset in (1, 8, 2))
    train_signatures: set[tuple[int, ...]] = set()
    signature_groups: dict[str, dict[str, dict[tuple[int, ...], _Counts]]] = {
        "train": {"seen": {}, "unseen": {}},
        "dev": {"seen": {}, "unseen": {}},
    }
    u_groups: dict[str, dict[tuple[int, ...], _Counts]] = {
        split: {bits: _Counts() for bits in itertools.product((0, 1), repeat=3)}
        for split in ("train", "dev")
    }
    residual_labels: dict[str, dict[tuple[int, ...], list[int]]] = {
        split: {bits: [0, 0] for bits in itertools.product((0, 1), repeat=3)}
        for split in ("train", "dev")
    }
    overall: dict[str, dict[str, int | float]] = {
        split: {"rows": 0, "correct": 0, "errors": 0, "bce_sum": 0.0}
        for split in ("train", "dev")
    }

    original_modes = [(module, module.training) for module in model.modules()]
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    handles: list[torch.utils.hooks.RemovableHandle] = []
    captured: dict[int, Tensor] = {}

    def capture(index: int) -> Any:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, Tensor):
                raise TypeError("local module output must be a tensor")
            captured[index] = output.detach()

        return hook

    try:
        for index, module in enumerate(model.local_modules):
            handles.append(module.register_forward_hook(capture(index)))
        model.eval()
        with torch.no_grad():
            corner_summaries = _corner_summaries(model, captured, device)
            for split in ("train", "dev"):
                inputs, labels = validated[split]
                split_totals = overall[split]
                for start in range(0, len(inputs), batch_size):
                    stop = min(start + batch_size, len(inputs))
                    batch_cpu = inputs[start:stop]
                    targets_cpu = labels[start:stop]
                    batch = batch_cpu.to(device)
                    targets = targets_cpu.to(device)
                    captured.clear()
                    logits = model(batch)
                    if logits.shape != targets.shape or not bool(torch.isfinite(logits).all()):
                        raise ValueError("model produced invalid diagnostic logits")
                    local_logits = [
                        _take_local_logits(
                            captured, index, stop - start,
                            1 if isinstance(model, TwoRoundLocalPredictor) else 4,
                        )
                        for index in range(len(model.local_modules))
                    ]
                    if captured:
                        raise RuntimeError("unexpected local module hook output")
                    if not all(bool(torch.isfinite(value).all()) for value in local_logits):
                        raise ValueError("model produced nonfinite local logits")

                    target_bits = targets.bool().reshape(-1)
                    predictions = (logits > 0).reshape(-1)
                    errors = predictions != target_bits
                    count = stop - start
                    split_totals["rows"] = int(split_totals["rows"]) + count
                    split_totals["correct"] = int(split_totals["correct"]) + int((~errors).sum())
                    split_totals["errors"] = int(split_totals["errors"]) + int(errors.sum())
                    split_totals["bce_sum"] = float(split_totals["bce_sum"]) + float(
                        F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
                    )

                    x_bits = batch_cpu[:, :32].bool()
                    y_bits = batch_cpu[:, 32:].bool()
                    u_columns: list[Tensor] = []
                    for q in q_positions:
                        f_q = (x_bits[:, (q + 1) % 32] & x_bits[:, (q + 8) % 32]) ^ x_bits[
                            :, (q + 2) % 32
                        ]
                        u_columns.append(y_bits[:, q] ^ f_q)
                    u_rows = torch.stack(u_columns, dim=1).tolist()
                    signature_rows = batch_cpu[:, indices].tolist()
                    labels_list = target_bits.cpu().tolist()
                    error_list = errors.cpu().tolist()
                    x_target = x_bits[:, target_bit].tolist()
                    for u_row, signature_row, label_value, error_value, x_value in zip(
                        u_rows, signature_rows, labels_list, error_list, x_target, strict=True
                    ):
                        u_signature = tuple(int(value) for value in u_row)
                        model_signature = tuple(int(value) for value in signature_row)
                        label = int(label_value)
                        error = bool(error_value)
                        residual = label ^ int(x_value)
                        u_groups[split][u_signature].add(error=error, label=residual)
                        residual_labels[split][u_signature][residual] += 1
                        if split == "train":
                            train_signatures.add(model_signature)
                            signature_groups[split]["seen"].setdefault(
                                model_signature, _Counts()
                            ).add(error=error, label=label)
                        else:
                            category = "seen" if model_signature in train_signatures else "unseen"
                            signature_groups[split][category].setdefault(
                                model_signature, _Counts()
                            ).add(error=error, label=label)
                    captured.clear()
    finally:
        for handle in handles:
            handle.remove()
        for module, was_training in original_modes:
            module.training = was_training
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)

    split_reports: dict[str, Any] = {}
    for split in ("train", "dev"):
        totals = overall[split]
        rows = int(totals["rows"])
        bce = float(totals["bce_sum"]) / rows
        if not math.isfinite(bce):
            raise ValueError("nonfinite diagnostic BCE")
        u_report: list[dict[str, Any]] = []
        for bits in itertools.product((0, 1), repeat=3):
            u_count = u_groups[split][bits]
            residual_count = residual_labels[split][bits]
            observed = u_count.rows > 0
            u_report.append({
                "u_bits": list(bits),
                "rows": u_count.rows,
                "errors": u_count.errors,
                "residual_labels": {
                    "definition": "absolute_label XOR public X[target_bit]",
                    "zero": residual_count[0],
                    "one": residual_count[1],
                    "consistent": (
                        residual_count[0] == 0 or residual_count[1] == 0
                    ) if observed else None,
                },
            })
        split_reports[split] = {
            "overall": {
                "rows": rows,
                "correct": int(totals["correct"]),
                "errors": int(totals["errors"]),
                "bce": bce,
            },
            "public_u_groups": {
                "bit_order": [f"U[{q}]" for q in q_positions],
                "cells": u_report,
            },
            "model_input_signatures": {
                "bit_indices": indices,
                "label_definition": "absolute output label Y, not public-Xj residual",
                "target_bit_in_signature": target_bit in indices,
                "train_seen_definition": (
                    "exact complete model.indices bit signature occurs in TRAIN"
                ),
                "seen": _group_summary(signature_groups[split]["seen"]),
                "unseen": _group_summary(signature_groups[split]["unseen"]),
            },
        }
    is_scalar = isinstance(model, TwoRoundLocalPredictor)
    return {
        "schema": "cipher-experiments-two-round-local-diagnostics-v1",
        "model": "simon-r2-local" if is_scalar else "simon-r2-vector",
        "target_bit": target_bit,
        "scope": ["train", "dev"],
        "state_interpretation": (
            "Local sigmoid activations are learned hidden scalars; no hidden value was "
            "compared with or verified as a cipher state."
        ),
        "local_hidden_sigmoid_corners": {
            "definition": (
                "sigmoid(local module scalar logit) over all 16 public 4-bit input corners"
                if is_scalar else
                "sigmoid(local module logits) over all 16 public 4-bit input corners; "
                "vector channels summarized independently"
            ),
            "modules": corner_summaries,
        },
        "splits": split_reports,
    }

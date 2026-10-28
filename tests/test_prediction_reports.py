"""Report-only boundaries: no model, RNG, input or optimization side effects."""
from __future__ import annotations

import ast
import inspect
import random
from typing import Any

import pytest
import torch

from cipher_experiments import prediction_reports, training
from cipher_experiments.models.prediction import build_predictor
from cipher_experiments.prediction_reports import experiment_kind, model_metadata


@pytest.mark.parametrize(("kind", "mode", "width", "has_metadata"), [
    ("simon-cnn", "direct", 8, True),
    ("simon-rotation-cnn", "direct", 8, True),
    ("mlp", "direct", 8, False),
    ("mlp", "public-coordinate", 8, True),
    ("logic16", "direct", 64, False),
    ("simon-structure", "direct", 64, True),
    ("simon-unrolled", "direct", 8, True),
    ("simon-unrolled", "plaintext-xor", 8, True),
    ("simon-unrolled", "plaintext-xor-no-final-low", 8, True),
    ("simon-r2-table", "direct", 16, True),
    ("simon-r2-local", "direct", 16, True),
    ("simon-r2-vector", "direct", 16, True),
])
def test_reports_preserve_parameters_gradients_inputs_rng_and_modes(
        kind: str, mode: str, width: int, has_metadata: bool) -> None:
    rounds = 6 if kind in ("simon-cnn", "simon-rotation-cnn") else 2
    model = build_predictor(width, model_kind=kind, rounds=rounds, prediction_mode=mode)
    model.train()
    # Preserve heterogeneous mode flags too; report generation must not call eval/train.
    children = list(model.children())
    if children:
        children[0].eval()
    x = torch.randint(0, 2, (12, 64), generator=torch.Generator().manual_seed(97)).float()
    y = x[:, 5:6].clone()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    gradients = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    modes = [m.training for m in model.modules()]
    original_x, original_y = x.clone(), y.clone()
    cpu_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    metadata = model_metadata(model, model_kind=kind, prediction_mode=mode, rounds=rounds,
                              target_bit=0, x_train=x, y_train=y, device=torch.device("cpu"))
    assert (metadata is not None) == has_metadata
    assert torch.equal(cpu_rng, torch.get_rng_state()) and random.getstate() == python_rng
    assert modes == [m.training for m in model.modules()]
    assert torch.equal(x, original_x) and torch.equal(y, original_y)
    assert all(torch.equal(state[n], t) for n, t in model.state_dict().items())
    assert all(p.grad is not None and torch.equal(p.grad, g)
               for p, g in zip(model.parameters(), gradients, strict=True))


def test_table_coverage_aggregates_only_explicit_train_labels() -> None:
    model = build_predictor(16, model_kind="simon-r2-table", rounds=2)
    # X=0 => U=Y; j=0: A=Y[1], B=Y[8], T=Y[2]. Two rows/cell.
    x = torch.zeros(8, 64)
    labels = []
    for i in range(8):
        cell = i // 2
        x[i, 33], x[i, 40] = cell // 2, cell % 2
        x[i, 34] = i % 2
        residual = 1 if cell == 1 else (i % 2 if cell == 2 else 0)
        labels.append([float(residual ^ (i % 2))])
    report = model_metadata(model, model_kind="simon-r2-table", prediction_mode="direct",
                            rounds=2, target_bit=0, x_train=x, y_train=torch.tensor(labels),
                            device=torch.device("cpu"))
    assert report is not None
    assert report["train_table"] == {
        "counts": [2, 2, 2, 2], "residual_ones": [0, 2, 1, 0],
        "covered_all_cells": True, "conflicting_cells": [2],
    }
    assert report["parameters"] == 4 and report["role"] == "positive-control"


@pytest.mark.parametrize(("dataset", "kind", "expected"), [
    ("synthetic", "mlp", "synthetic-engineering"),
    ("circuit-synthetic", "circuit-aligned", "synthetic-engineering"),
    ("simon", "simon-cnn", "simon-residual-cnn-final-bit-supervision"),
    ("simon", "simon-rotation-cnn", "simon-residual-cnn-final-bit-supervision"),
    ("simon", "mlp", "simon-output-bit-baseline"),
    ("simon", "logic16", "simon-output-bit-baseline"),
    ("simon", "simon-structure", "simon-structure-unknown-constants"),
    ("simon", "simon-unrolled", "simon-unrolled-final-bit-supervision"),
    ("simon", "simon-r2-table", "simon-r2-table-positive-control"),
    ("simon", "simon-r2-local", "simon-r2-local-neural-candidate"),
    ("simon", "simon-r2-vector", "simon-r2-vector-neural-candidate"),
])
def test_manifest_roles_preserve_existing_contract(dataset: str, kind: str, expected: str) -> None:
    assert experiment_kind(dataset, kind) == expected


def _imports(module: Any) -> set[str]:
    return {n.module or "" for n in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(n, ast.ImportFrom)}


def test_report_dependencies_and_common_training_responsibility() -> None:
    assert not (_imports(prediction_reports) & {
        "training", "evaluation", "checkpoints", "config", "artifacts", "data.prediction",
        "data.simon", "ciphers.simon",
    })
    source = inspect.getsource(prediction_reports)
    assert "write_json" not in source and "torch.save" not in source
    assert "y_train" in inspect.signature(model_metadata).parameters
    assert not ({"config", "key_seed", "test", "labels_for"}
                & set(inspect.signature(model_metadata).parameters))
    common = inspect.getsource(training.train)
    assert "conflicting_cells" not in common and "hidden_channels" not in common
    assert "public_table_features" not in common

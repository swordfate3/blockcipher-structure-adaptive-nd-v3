"""Synthetic contracts for the hard-forward straight-through SIMON predictor."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from torch import Tensor

from cipher_experiments.models.ste_simon import STESimonPredictor
from cipher_experiments.models.structured_simon import HardSimonPredictor


def _inputs(count: int = 32) -> Tensor:
    generator = torch.Generator().manual_seed(7781)
    return torch.randint(0, 2, (count, 64), generator=generator).float()


def _interpret(model: STESimonPredictor, inputs: Tensor) -> Tensor:
    """Evaluate the public DAG with Python booleans, independently of Torch gates."""
    result: list[list[float]] = []
    key_bits = (model.key_logits.detach() > 0).tolist()
    for row in inputs.to(dtype=torch.int64).tolist():
        values: list[int] = []
        for node in model.cone.nodes:
            if node.kind == "plaintext":
                (coordinate,) = node.coordinate
                values.append(row[coordinate])
            elif node.kind == "round_key":
                key_coordinate = (node.coordinate[0], node.coordinate[1])
                values.append(int(key_bits[model._key_positions[key_coordinate]]))
            else:
                left, right = (values[index] for index in node.args)
                if node.kind == "and":
                    values.append(left & right)
                else:
                    assert node.kind == "xor"
                    values.append(left ^ right)
        result.append([float(values[model.cone.output])])
    return torch.tensor(result, dtype=torch.float32)


@pytest.mark.parametrize("rounds,target_bit", [(1, 0), (2, 31), (3, 32), (4, 63), (5, 0), (6, 63)])
def test_forward_is_exact_hard_dag_for_all_supported_rounds(
    rounds: int, target_bit: int
) -> None:
    model = STESimonPredictor(rounds=rounds, target_bit=target_bit)
    inputs = _inputs()

    expected_bits = _interpret(model, inputs)
    logits = model(inputs)
    hard_logits = model.hard_logits(inputs)

    assert logits.shape == (len(inputs), 1)
    assert logits.dtype == torch.float32
    assert bool(torch.isfinite(logits).all())
    assert torch.equal(logits, 2.0 * (2.0 * expected_bits - 1.0))
    assert torch.equal(hard_logits, logits)
    assert set(logits.unique().tolist()) == {-2.0, 2.0}

    hard = HardSimonPredictor(rounds=rounds, target_bit=target_bit)
    with torch.no_grad():
        hard.key_logits.copy_(torch.where(model.key_logits > 0, 1.0, -1.0))
    assert torch.equal(hard.hard_logits(inputs) > 0, logits > 0)


def test_initialization_scale_and_only_key_logits_are_trainable() -> None:
    torch.manual_seed(137)
    model = STESimonPredictor(rounds=4, target_bit=0)
    assert list(model.named_parameters()) == [("key_logits", model.key_logits)]
    assert model.key_logits.requires_grad
    assert model.key_logits.dtype == torch.float32
    assert model.key_logits.numel() == len(model.key_coordinates)
    assert bool((model.key_logits.abs() < 0.11).all())
    assert bool((model.key_logits.abs() > 0.09).all())
    assert model.key_coordinates == tuple(sorted(model.cone.round_key_coordinates))


def test_seeded_initialization_is_reproducible() -> None:
    torch.manual_seed(9201)
    first = STESimonPredictor(rounds=3, target_bit=0)
    torch.manual_seed(9201)
    second = STESimonPredictor(rounds=3, target_bit=0)
    assert torch.equal(first.key_logits, second.key_logits)


def test_straight_through_gradients_are_finite_and_nonzero_at_wrong_constant() -> None:
    model = STESimonPredictor(rounds=1, target_bit=0)
    with torch.no_grad():
        model.key_logits.fill_(0.1)
    inputs = _inputs(16)
    targets = torch.zeros((16, 1), dtype=torch.float32)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(inputs), targets)
    loss.backward()  # type: ignore[no-untyped-call]

    gradient = model.key_logits.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0.0


def test_artificial_unknown_xor_constant_can_flip_under_adam() -> None:
    # This is a four-input synthetic Boolean truth table, not cipher data or key fitting.
    model = STESimonPredictor(rounds=1, target_bit=0)
    with torch.no_grad():
        model.key_logits.fill_(0.1)  # starts at constant 1; synthetic target uses constant 0

    inputs = torch.zeros((16, 64), dtype=torch.float32)
    assignments = ((torch.arange(16)[:, None] >> torch.arange(3, -1, -1)) & 1).float()
    coordinates = [32, 1, 8, 2]
    inputs[:, coordinates] = assignments
    fixed_output = assignments[:, 0].bool() ^ (
        assignments[:, 1].bool() & assignments[:, 2].bool()
    ) ^ assignments[:, 3].bool()
    targets = fixed_output.float().unsqueeze(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(inputs), targets)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    assert float(model.key_logits[0].detach()) < 0.0
    assert torch.equal(model(inputs) > 0, targets.bool())


def test_eval_mode_and_zero_threshold_do_not_change_hard_forward() -> None:
    model = STESimonPredictor(rounds=2, target_bit=0)
    inputs = _inputs(20)
    with torch.no_grad():
        model.key_logits[0] = 0.0
    saved_logits = model.key_logits.detach().clone()

    model.train()
    train_logits = model(inputs)
    model.eval()
    eval_logits = model(inputs)

    assert torch.equal(train_logits, eval_logits)
    assert torch.equal(model.key_logits, saved_logits)
    assert torch.equal(train_logits, model.hard_logits(inputs))


def test_module_has_no_cipher_label_solver_or_key_schedule_imports() -> None:
    source = Path(__file__).resolve().parents[1] / "src/cipher_experiments/models/ste_simon.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imported <= {"torch", "representations", "prediction", "structured_simon", "__future__"}


def test_round_and_target_validation_are_inherited() -> None:
    with pytest.raises(ValueError, match="rounds"):
        STESimonPredictor(rounds=7)
    with pytest.raises(ValueError, match="target_bit"):
        STESimonPredictor(target_bit=64)
    with pytest.raises(ValueError, match="no learnable round-key"):
        STESimonPredictor(rounds=1, target_bit=32)

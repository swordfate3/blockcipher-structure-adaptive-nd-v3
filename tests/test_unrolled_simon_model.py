"""Synthetic contracts for the differentiable unrolled SIMON-shaped predictor."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from cipher_experiments.models.prediction import support_indices
from cipher_experiments.models.unrolled_simon import UnrolledSimonPredictor


def _plaintexts(count: int = 12) -> Tensor:
    generator = torch.Generator().manual_seed(31415)
    return torch.randint(0, 2, (count, 64), generator=generator).to(torch.float32)


@pytest.mark.parametrize("rounds", range(1, 9))
def test_round_counts_return_finite_single_bit_logits(rounds: int) -> None:
    model = UnrolledSimonPredictor(hidden_width=5, rounds=rounds, target_bit=0)
    result = model(_plaintexts())
    assert result.shape == (12, 1)
    assert result.dtype is torch.float32
    assert bool(torch.isfinite(result).all())

    if rounds >= 2:
        low_half = UnrolledSimonPredictor(hidden_width=5, rounds=rounds, target_bit=63)
        assert low_half(_plaintexts()).shape == (12, 1)


def test_metadata_cell_and_strict_state_dict_round_trip() -> None:
    model = UnrolledSimonPredictor(7, rounds=3, target_bit=12)
    assert model.rounds == 3
    assert model.target_bit == 12
    assert model.feature_mode == "full"
    assert model.prediction_mode == "direct"
    assert model.checkpoint_config_sha256 == ""
    assert model.conditions.shape == (3, 32, 4)
    assert model.cell[0].in_features == 8
    assert model.cell[0].out_features == 7
    assert isinstance(model.cell[1], torch.nn.Tanh)
    assert model.cell[2].out_features == 1
    assert set(model.state_dict()) == {
        "conditions",
        "cell.0.weight",
        "cell.0.bias",
        "cell.2.weight",
        "cell.2.bias",
    }

    restored = UnrolledSimonPredictor(7, rounds=3, target_bit=12)
    restored.load_state_dict(model.state_dict(), strict=True)
    inputs = _plaintexts(4)
    assert torch.equal(model(inputs), restored(inputs))


def test_constructor_and_public_signature_are_fixed_and_validated() -> None:
    assert tuple(inspect.signature(UnrolledSimonPredictor).parameters) == (
        "hidden_width",
        "rounds",
        "target_bit",
        "prediction_mode",
    )
    assert tuple(inspect.signature(UnrolledSimonPredictor.forward).parameters) == (
        "self",
        "inputs",
    )

    for name, valid_kwargs in (
        ("hidden_width", {"hidden_width": 0}),
        ("hidden_width", {"hidden_width": 4097}),
        ("rounds", {"rounds": 0}),
        ("rounds", {"rounds": 9}),
        ("target_bit", {"target_bit": -1}),
        ("target_bit", {"target_bit": 64}),
    ):
        with pytest.raises(ValueError, match=name):
            cast(Any, UnrolledSimonPredictor)(**valid_kwargs)

    invalid_integer_kwargs: tuple[dict[str, object], ...] = (
        {"hidden_width": True},
        {"rounds": True},
        {"target_bit": 0.0},
    )
    for invalid_case in invalid_integer_kwargs:
        with pytest.raises(TypeError, match="built-in int"):
            cast(Any, UnrolledSimonPredictor)(**invalid_case)
    with pytest.raises(ValueError, match="no trainable output"):
        UnrolledSimonPredictor(rounds=1, target_bit=32)
    with pytest.raises(ValueError, match="prediction_mode"):
        UnrolledSimonPredictor(prediction_mode="public-residual")
    for kwargs in ({"rounds": 1}, {"rounds": 3}, {"target_bit": 32}):
        with pytest.raises(ValueError, match="plaintext-xor"):
            cast(Any, UnrolledSimonPredictor)(prediction_mode="plaintext-xor", **kwargs)


@pytest.mark.parametrize(
    "inputs,error,message",
    [
        ([[0.0] * 64], TypeError, "torch.Tensor"),
        (torch.zeros((0, 64)), ValueError, "shape"),
        (torch.zeros((2, 63)), ValueError, "shape"),
        (torch.zeros((1, 64), dtype=torch.float64), TypeError, "float32"),
        (torch.full((1, 64), float("nan")), ValueError, "finite"),
        (torch.full((1, 64), 0.5), ValueError, "binary"),
    ],
)
def test_forward_rejects_malformed_plaintexts(
    inputs: object, error: type[Exception], message: str
) -> None:
    model = UnrolledSimonPredictor()
    with pytest.raises(error, match=message):
        model(cast(Tensor, inputs))


def test_forward_requires_matching_device() -> None:
    model = UnrolledSimonPredictor(rounds=2)
    with pytest.raises(ValueError, match="same device"):
        model.to(device="meta")(_plaintexts(1))


def test_initialization_is_seeded_and_forward_has_no_random_state_side_effect() -> None:
    torch.manual_seed(839)
    first = UnrolledSimonPredictor(9, rounds=3)
    torch.manual_seed(839)
    repeated = UnrolledSimonPredictor(9, rounds=3)
    torch.manual_seed(840)
    different = UnrolledSimonPredictor(9, rounds=3)
    assert all(
        torch.equal(first.state_dict()[name], repeated.state_dict()[name])
        for name in first.state_dict()
    )
    assert not torch.equal(first.conditions, different.conditions)

    inputs = _plaintexts(8)
    state_before = torch.random.get_rng_state().clone()
    train_result = first(inputs)
    first.eval()
    eval_result = first(inputs)
    assert torch.equal(train_result, eval_result)
    assert torch.equal(state_before, torch.random.get_rng_state())


@pytest.mark.parametrize("target_bit", range(32))
def test_plaintext_xor_logit_sign_and_absolute_bce_equivalence(target_bit: int) -> None:
    torch.manual_seed(1000 + target_bit)
    direct = UnrolledSimonPredictor(32, rounds=2, target_bit=target_bit)
    transformed = UnrolledSimonPredictor(
        32, rounds=2, target_bit=target_bit, prediction_mode="plaintext-xor"
    )
    transformed.load_state_dict(direct.state_dict(), strict=True)
    assert set(direct.state_dict()) == set(transformed.state_dict())
    assert sum(parameter.numel() for parameter in direct.parameters()) == 577
    assert sum(parameter.numel() for parameter in transformed.parameters()) == 577
    assert all(torch.equal(direct.state_dict()[name], transformed.state_dict()[name])
               for name in direct.state_dict())

    inputs = torch.zeros((4, 64), dtype=torch.float32)
    inputs[1, target_bit] = 1.0
    inputs[2] = 1.0
    inputs[3] = 1.0
    inputs[3, target_bit] = 0.0
    labels = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    plain_logits = direct(inputs)
    absolute_logits = transformed(inputs)
    sign = 1.0 - 2.0 * inputs[:, target_bit : target_bit + 1]
    assert torch.equal(absolute_logits, sign * plain_logits)
    assert torch.equal(absolute_logits[inputs[:, target_bit] == 0],
                       plain_logits[inputs[:, target_bit] == 0])
    assert torch.equal(absolute_logits[inputs[:, target_bit] == 1],
                       -plain_logits[inputs[:, target_bit] == 1])
    bce_absolute = F.binary_cross_entropy_with_logits(absolute_logits, labels)
    equivalent_labels = torch.logical_xor(
        labels.bool(), inputs[:, target_bit : target_bit + 1].bool()
    ).float()
    bce_equivalent = F.binary_cross_entropy_with_logits(plain_logits, equivalent_labels)
    torch.testing.assert_close(bce_absolute, bce_equivalent, rtol=1e-6, atol=1e-7)


def test_plaintext_xor_mode_preserves_direct_initialization_and_buffers() -> None:
    torch.manual_seed(8675309)
    direct = UnrolledSimonPredictor(32, rounds=2, target_bit=13)
    torch.manual_seed(8675309)
    transformed = UnrolledSimonPredictor(
        32, rounds=2, target_bit=13, prediction_mode="plaintext-xor"
    )
    assert set(direct.state_dict()) == set(transformed.state_dict())
    assert dict(direct.named_buffers()).keys() == dict(transformed.named_buffers()).keys()
    assert sum(parameter.numel() for parameter in direct.parameters()) == 577
    assert sum(parameter.numel() for parameter in transformed.parameters()) == 577
    assert all(torch.equal(value, transformed.state_dict()[name])
               for name, value in direct.state_dict().items())


def test_loss_backpropagates_to_early_conditions_and_final_cell() -> None:
    torch.manual_seed(251)
    model = UnrolledSimonPredictor(8, rounds=3, target_bit=0)
    inputs = _plaintexts(20)
    labels = torch.randint(0, 2, (20, 1), generator=torch.Generator().manual_seed(98))
    loss = F.binary_cross_entropy_with_logits(model(inputs), labels.to(torch.float32))
    loss.backward()  # type: ignore[no-untyped-call]

    assert model.conditions.grad is not None
    assert bool(torch.isfinite(model.conditions.grad).all())
    assert float(model.conditions.grad[0].abs().sum()) > 0.0
    assert model.cell[0].weight.grad is not None
    final_cell_grad = cast(Tensor, model.cell[2].weight).grad
    assert final_cell_grad is not None
    assert float(torch.sum(torch.abs(final_cell_grad))) > 0.0


def test_input_gradient_is_confined_to_public_dependency_cone() -> None:
    torch.manual_seed(817)
    model = UnrolledSimonPredictor(8, rounds=3, target_bit=0)
    inputs = _plaintexts(16).requires_grad_()
    model(inputs).sum().backward()
    assert inputs.grad is not None
    assert bool(torch.isfinite(inputs.grad).all())
    support = support_indices(3, 0)
    outside = [index for index in range(64) if index not in support]
    assert float(inputs.grad[:, support].abs().sum()) > 0
    assert torch.count_nonzero(inputs.grad[:, outside]) == 0


def test_low_half_output_is_exact_previous_round_logit_copy() -> None:
    torch.manual_seed(725)
    full = UnrolledSimonPredictor(6, rounds=3, target_bit=32)
    two_rounds = UnrolledSimonPredictor(6, rounds=2, target_bit=0)
    with torch.no_grad():
        two_rounds.cell.load_state_dict(full.cell.state_dict(), strict=True)
        two_rounds.conditions.copy_(full.conditions[:2])

    inputs = _plaintexts(9)
    assert torch.equal(full(inputs), two_rounds(inputs))


def test_model_module_imports_no_cipher_data_or_experiment_modules() -> None:
    path = Path(__file__).resolve().parents[1] / "src/cipher_experiments/models/unrolled_simon.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_roots = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_roots.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imported_roots <= {"torch", "prediction", "__future__"}
    assert "key" not in inspect.signature(UnrolledSimonPredictor).parameters
    assert "key" not in inspect.signature(UnrolledSimonPredictor.forward).parameters

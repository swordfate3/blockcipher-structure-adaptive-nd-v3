"""CPU-only synthetic contracts for the differentiable soft-logic model."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import pytest
import torch
from torch.nn import functional as F

from cipher_experiments.models.logic import LogicPredictor, SoftGateLayer
from cipher_experiments.models.prediction import public_reference, support_indices


def test_truth_tables_match_all_boolean_corners_and_mask_order() -> None:
    corners = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    values = SoftGateLayer.gate_values(corners[:, 0], corners[:, 1])
    assert values.shape == (4, 16)
    expected = torch.tensor(
        [[float((mask >> corner) & 1) for mask in range(16)] for corner in range(4)]
    )
    assert torch.equal(values, expected)
    # Local little-endian corner-bit convention: these are intentionally this
    # module's A/B table IDs, not IDs copied from an external implementation.
    assert torch.equal(values[:, 12], corners[:, 0])  # A pass-through.
    assert torch.equal(values[:, 10], corners[:, 1])  # B pass-through.
    assert torch.equal(values[0], torch.tensor([float(mask & 1) for mask in range(16)]))
    assert torch.equal(values[1], torch.tensor([float((mask >> 1) & 1) for mask in range(16)]))


def test_truth_tables_use_bernoulli_corner_interpolation() -> None:
    a = torch.tensor([0.2, 0.75, 0.4], dtype=torch.float32)
    b = torch.tensor([0.7, 0.1, 0.6], dtype=torch.float32)
    actual = SoftGateLayer.gate_values(a, b)
    corner_probabilities = torch.stack(
        ((1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b), dim=-1
    )
    truth_tables = torch.tensor(
        [[float((mask >> corner) & 1) for corner in range(4)] for mask in range(16)]
    )
    expected = corner_probabilities @ truth_tables.T
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)
    assert bool(((actual >= 0) & (actual <= 1)).all())


def test_uniform_truth_table_mixture_has_zero_input_jacobian_and_gate_gradient() -> None:
    a = torch.tensor([0.2, 0.65], requires_grad=True)
    b = torch.tensor([0.7, 0.1], requires_grad=True)
    logits = torch.zeros((2, 16), requires_grad=True)
    table_values = SoftGateLayer.gate_values(a[:, None], b[:, None])
    output = (table_values * torch.softmax(logits, dim=-1)).sum(dim=-1)
    grad_a, grad_b, grad_logits = torch.autograd.grad(
        output.sum(), (a, b, logits)
    )
    assert torch.allclose(grad_a, torch.zeros_like(grad_a), atol=1e-7, rtol=0)
    assert torch.allclose(grad_b, torch.zeros_like(grad_b), atol=1e-7, rtol=0)
    assert bool(torch.isfinite(grad_logits).all())
    assert bool((grad_logits != 0).any())


def test_soft_gate_accepts_continuous_hidden_values_and_rejects_invalid_values() -> None:
    layer = SoftGateLayer(2, 3)
    continuous = torch.tensor([[0.2, 0.8], [1.0, 0.0]], dtype=torch.float32)
    outputs = layer(continuous)
    assert outputs.shape == (2, 3)
    assert bool(((outputs >= 0) & (outputs <= 1)).all())
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        layer(torch.tensor([[0.2, 1.1]], dtype=torch.float32))
    with pytest.raises(ValueError, match="finite"):
        layer(torch.tensor([[0.2, float("inf")]], dtype=torch.float32))
    with pytest.raises(TypeError, match="float32"):
        layer(torch.zeros((1, 2), dtype=torch.float64))


def test_seeded_wiring_is_persistent_covered_distinct_and_roundtrips() -> None:
    torch.manual_seed(9301)
    first = SoftGateLayer(7, 11)
    torch.manual_seed(9301)
    second = SoftGateLayer(7, 11)
    assert torch.equal(first.left_indices, second.left_indices)
    assert torch.equal(first.right_indices, second.right_indices)
    assert torch.equal(first.gate_logits, second.gate_logits)
    assert torch.equal(first.left_indices, torch.arange(11).remainder(7))
    assert bool((first.left_indices != first.right_indices).all())
    assert {"left_indices", "right_indices", "gate_logits"} <= set(first.state_dict())
    restored = SoftGateLayer(7, 11)
    restored.load_state_dict(first.state_dict())
    assert torch.equal(restored.left_indices, first.left_indices)
    assert torch.equal(restored.right_indices, first.right_indices)
    restored.validate_wiring()


def test_corrupted_wiring_is_rejected_and_single_wire_is_allowed() -> None:
    layer = SoftGateLayer(3, 2)
    layer.left_indices[0] = 3
    with pytest.raises(ValueError, match="out-of-range"):
        layer.validate_wiring()
    layer = SoftGateLayer(3, 2)
    layer.right_indices[0] = layer.left_indices[0]
    with pytest.raises(ValueError, match="not connect an input to itself"):
        layer.validate_wiring()
    single = SoftGateLayer(1, 2)
    single.validate_wiring()
    assert torch.equal(single.left_indices, single.right_indices)


def test_logic_predictor_initialization_is_finite_and_gradients_reach_all_gates() -> None:
    torch.manual_seed(845)
    model = LogicPredictor(hidden_width=64)
    inputs = torch.randint(0, 2, (9, 64), dtype=torch.int64).float().requires_grad_()
    logits = model(inputs)
    assert logits.shape == (9, 1)
    assert bool(torch.isfinite(logits).all())
    loss = logits.square().mean()
    loss.backward()
    assert inputs.grad is not None and float(inputs.grad.norm()) > 0
    for layer in model.gate_layers:
        assert layer.gate_logits.grad is not None
        assert bool(torch.isfinite(layer.gate_logits.grad).all())
        assert bool((layer.gate_logits.grad != 0).any())
    assert all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def test_logic_modes_and_nonpersistent_feature_indices() -> None:
    model = LogicPredictor(64, rounds=3, target_bit=0, feature_mode="support")
    assert model.input_width == len(support_indices(3, 0))
    assert torch.equal(model.indices, torch.tensor(support_indices(3, 0)))
    assert "indices" not in model.state_dict()
    assert model.checkpoint_config_sha256 == ""
    assert model(torch.randint(0, 2, (2, 64)).float()).shape == (2, 1)
    corrupt_state = model.state_dict()
    corrupt_state["first_gate.left_indices"][0] = model.input_width
    restored = LogicPredictor(64, rounds=3, target_bit=0, feature_mode="support")
    restored.load_state_dict(corrupt_state)
    with pytest.raises(ValueError, match="out-of-range"):
        restored.validate_wiring()

    residual = LogicPredictor(64, rounds=3, target_bit=7, prediction_mode="public-residual")
    inputs = torch.randint(0, 2, (3, 64)).float()
    with torch.no_grad():
        hidden = inputs.index_select(1, residual.indices)
        for layer in residual.gate_layers:
            hidden = layer(hidden)
        direct_logits = residual.head(hidden)
        expected = (1 - 2 * public_reference(inputs, 3, 7)) * direct_logits
        assert torch.equal(residual(inputs), expected)


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
def test_logic_predictor_rejects_invalid_plaintext_inputs(
    inputs: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        LogicPredictor(64)(cast(torch.Tensor, inputs))


def test_constructor_validates_modes_and_builtin_dimensions() -> None:
    with pytest.raises(TypeError, match="built-in int"):
        LogicPredictor(True)
    with pytest.raises(ValueError, match="hidden_width"):
        LogicPredictor(4097)
    with pytest.raises(ValueError, match="feature_mode"):
        LogicPredictor(64, feature_mode="unknown")
    with pytest.raises(ValueError, match="prediction_mode"):
        LogicPredictor(64, prediction_mode="unknown")
    with pytest.raises(ValueError, match="rounds 1..4"):
        LogicPredictor(64, rounds=5, prediction_mode="public-residual")
    with pytest.raises(ValueError, match="at least input_width"):
        LogicPredictor(63)
    with pytest.raises(TypeError, match="built-in int"):
        SoftGateLayer(cast(int, 1.0), 2)


def test_three_gate_layers_handwired_8bit_parity_on_all_binary_inputs() -> None:
    """Exact Boolean-circuit witness, not a training or cipher-prediction claim."""
    model = LogicPredictor(hidden_width=64)
    layers = model.gate_layers
    with torch.no_grad():
        layers[0].left_indices.copy_(torch.tensor([0, 2, 4, 6] + [0] * 60))
        layers[0].right_indices.copy_(torch.tensor([1, 3, 5, 7] + [1] * 60))
        layers[1].left_indices.copy_(torch.tensor([0, 2] + [0] * 62))
        layers[1].right_indices.copy_(torch.tensor([1, 3] + [1] * 62))
        layers[2].left_indices.zero_()
        layers[2].right_indices.fill_(1)
        # Under this implementation's corner ordering, XOR has table mask 6.
        for layer in layers:
            layer.gate_logits.fill_(-15.0)
            layer.gate_logits[:, 6] = 15.0
        model.head.weight.zero_()
        model.head.weight[0, 0] = 20.0
        model.head.bias.fill_(-10.0)
    model.validate_wiring()

    values = torch.arange(256, dtype=torch.int64)
    eight_bits = ((values[:, None] >> torch.arange(7, -1, -1)) & 1).float()
    inputs = torch.zeros((256, 64), dtype=torch.float32)
    inputs[:, :8] = eight_bits
    targets = (eight_bits.sum(dim=1, keepdim=True).remainder(2)).float()
    model.train()
    train_logits = model(inputs)
    model.eval()
    eval_logits = model(inputs)
    assert torch.equal(train_logits, eval_logits)
    assert float(((train_logits > 0) == targets.bool()).float().mean()) > 0.99
    assert bool(torch.isfinite(train_logits).all())


def test_one_gate_can_reduce_synthetic_xor_loss_in_bounded_updates() -> None:
    torch.manual_seed(111)
    layer = SoftGateLayer(2, 1)
    inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.15)
    with torch.no_grad():
        initial_loss = F.mse_loss(layer(inputs), targets)
    updates = 0
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(layer(inputs), targets)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        updates += 1
    with torch.no_grad():
        final_loss = F.mse_loss(layer(inputs), targets)
    assert updates == 150
    assert float(final_loss) < float(initial_loss) * 0.2


def test_logic_module_imports_only_torch_and_prediction_helpers() -> None:
    path = Path(__file__).resolve().parents[1] / "src/cipher_experiments/models/logic.py"
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
    assert imported_roots <= {"torch", "prediction", "typing", "__future__"}

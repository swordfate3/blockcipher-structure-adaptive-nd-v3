"""CPU-only synthetic contracts for the fixed-width circuit predictor."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import pytest
import torch

from cipher_experiments.models.circuit import CircuitPredictor


def test_constructor_has_fixed_interface_and_expected_trainable_parameter_count() -> None:
    torch.manual_seed(410)
    model = CircuitPredictor()

    assert model.indices.tolist() == list(range(8))
    assert "indices" not in model.state_dict()
    assert model.checkpoint_config_sha256 == ""
    assert [(layer.in_features, layer.out_features) for layer in model.gate_layers] == [
        (8, 4),
        (4, 2),
        (2, 1),
    ]
    assert model.head.in_features == model.head.out_features == 1
    assert torch.equal(model.head.weight, torch.tensor([[4.0]]))
    assert torch.equal(model.head.bias, torch.tensor([-2.0]))
    assert sum(parameter.numel() for parameter in model.parameters()) == 114
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_seeded_wiring_changes_only_the_wiring_not_trainable_initial_state() -> None:
    torch.manual_seed(9271)
    aligned = CircuitPredictor(wiring="aligned")
    torch.manual_seed(9271)
    random = CircuitPredictor(wiring="random")

    for aligned_parameter, random_parameter in zip(
        aligned.parameters(), random.parameters(), strict=True
    ):
        assert torch.equal(aligned_parameter, random_parameter)
    assert any(
        not torch.equal(aligned_layer.left_indices, random_layer.left_indices)
        or not torch.equal(aligned_layer.right_indices, random_layer.right_indices)
        for aligned_layer, random_layer in zip(aligned.gate_layers, random.gate_layers, strict=True)
    )
    assert [
        (layer.left_indices.tolist(), layer.right_indices.tolist()) for layer in aligned.gate_layers
    ] == [([0, 2, 4, 6], [1, 3, 5, 7]), ([0, 2], [1, 3]), ([0], [1])]


def test_random_wiring_is_a_perfect_matching_at_every_layer() -> None:
    torch.manual_seed(127)
    model = CircuitPredictor(wiring="random")

    for layer in model.gate_layers:
        wires = torch.cat((layer.left_indices, layer.right_indices))
        assert torch.equal(torch.sort(wires).values, torch.arange(layer.in_features))
        assert bool((layer.left_indices != layer.right_indices).all())
        assert layer.left_indices.shape == layer.right_indices.shape == (layer.out_features,)
    model.validate_wiring()


def test_duplicate_wires_loaded_from_checkpoint_are_rejected() -> None:
    source = CircuitPredictor()
    state = source.state_dict()
    state["first_gate.left_indices"][1] = state["first_gate.left_indices"][0]
    restored = CircuitPredictor()
    restored.load_state_dict(state)

    with pytest.raises(ValueError, match="cover every input exactly once"):
        restored.validate_wiring()
    with pytest.raises(ValueError, match="cover every input exactly once"):
        restored(torch.zeros((1, 64), dtype=torch.float32))


def _handwire_xor_of_four_pairwise_ands(model: CircuitPredictor) -> None:
    """Install a synthetic Boolean witness, not a fitted target model."""
    layers = model.gate_layers
    with torch.no_grad():
        layers[0].left_indices.copy_(torch.tensor([0, 2, 4, 6]))
        layers[0].right_indices.copy_(torch.tensor([1, 3, 5, 7]))
        layers[1].left_indices.copy_(torch.tensor([0, 2]))
        layers[1].right_indices.copy_(torch.tensor([1, 3]))
        layers[2].left_indices.copy_(torch.tensor([0]))
        layers[2].right_indices.copy_(torch.tensor([1]))
        # AND has mask 8: in little-endian corner order its only set bit
        # is the 11 corner. XOR has mask 6.
        for layer in layers:
            layer.gate_logits.fill_(-15.0)
        layers[0].gate_logits[:, 8] = 15.0
        layers[1].gate_logits[:, 6] = 15.0
        layers[2].gate_logits[:, 6] = 15.0
        model.head.weight.fill_(20.0)
        model.head.bias.fill_(-10.0)
    model.validate_wiring()


@pytest.mark.parametrize("wiring", ["aligned", "random"])
def test_handwired_circuit_classifies_xor_of_four_ands_in_train_and_eval(
    wiring: str,
) -> None:
    model = CircuitPredictor(wiring=wiring)
    _handwire_xor_of_four_pairwise_ands(model)
    values = torch.arange(256, dtype=torch.int64)
    first_eight = ((values[:, None] >> torch.arange(7, -1, -1)) & 1).float()
    inputs = torch.zeros((256, 64), dtype=torch.float32)
    inputs[:, :8] = first_eight
    targets = (
        (
            first_eight[:, 0] * first_eight[:, 1]
            + first_eight[:, 2] * first_eight[:, 3]
            + first_eight[:, 4] * first_eight[:, 5]
            + first_eight[:, 6] * first_eight[:, 7]
        )
        .remainder(2)
        .bool()
    )

    model.train()
    train_logits = model(inputs)
    model.eval()
    eval_logits = model(inputs)
    assert train_logits.shape == (256, 1)
    assert torch.equal(train_logits, eval_logits)
    assert float(((train_logits[:, 0] > 0) == targets).float().mean()) > 0.99
    assert bool(torch.isfinite(train_logits).all())


@pytest.mark.parametrize("wiring", ["aligned", "random"])
def test_handwired_witness_uses_first_eight_in_order_and_ignores_rest(wiring: str) -> None:
    model = CircuitPredictor(wiring=wiring)
    _handwire_xor_of_four_pairwise_ands(model)
    inputs = torch.zeros((2, 64), dtype=torch.float32)
    inputs[:, 1] = 1.0
    inputs[1, 0] = 1.0
    changed_unused_bits = inputs.clone()
    changed_unused_bits[:, 8:] = 1.0

    logits = model(inputs)
    assert bool(logits[0, 0] < 0 < logits[1, 0])
    assert torch.equal(logits, model(changed_unused_bits))


def test_input_validation_and_constructor_wiring_validation() -> None:
    with pytest.raises(ValueError, match="wiring"):
        CircuitPredictor(wiring="unknown")
    with pytest.raises(TypeError, match="string"):
        CircuitPredictor(wiring=cast(str, None))
    with pytest.raises(TypeError, match="torch.Tensor"):
        CircuitPredictor()(cast(torch.Tensor, [[0.0] * 64]))
    with pytest.raises(ValueError, match="shape"):
        CircuitPredictor()(torch.zeros((1, 7), dtype=torch.float32))
    with pytest.raises(TypeError, match="float32"):
        CircuitPredictor()(torch.zeros((1, 64), dtype=torch.float64))
    with pytest.raises(ValueError, match="binary"):
        CircuitPredictor()(torch.full((1, 64), 0.5, dtype=torch.float32))


def test_circuit_module_imports_only_torch_logic_and_prediction_helpers() -> None:
    path = Path(__file__).resolve().parents[1] / "src/cipher_experiments/models/circuit.py"
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
    assert imported_roots <= {"torch", "logic", "prediction", "typing", "__future__"}

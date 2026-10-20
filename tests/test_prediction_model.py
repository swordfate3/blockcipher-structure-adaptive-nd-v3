"""Synthetic CPU contracts for key-free plaintext bit prediction models."""

from __future__ import annotations

import ast
import inspect
import random
from pathlib import Path
from typing import cast

import pytest
import torch
from torch.nn import functional as F

from cipher_experiments.ciphers import encrypt
from cipher_experiments.models.mlp import BitMLP
from cipher_experiments.models.prediction import (
    PlaintextPredictor,
    public_reference,
    support_indices,
)


def _plaintext(value: int) -> list[float]:
    return [float((value >> (63 - bit)) & 1) for bit in range(64)]


def _scalar_reference(plaintext: int, rounds: int, target_bit: int) -> float:
    return float((encrypt(plaintext, 0, rounds) >> (63 - target_bit)) & 1)


@pytest.mark.parametrize("rounds", [1, 2, 3, 4])
@pytest.mark.parametrize("target_bit", [0, 7, 31, 32, 47, 63])
def test_public_reference_matches_zero_key_scalar_encryption(
    rounds: int, target_bit: int
) -> None:
    plaintexts = [
        0,
        (1 << 64) - 1,
        1 << 63,
        0x0123456789ABCDEF,
        0xFEDCBA9876543210,
        0xA5A55A5AC3C33C3C,
    ]
    tensor = torch.tensor([_plaintext(value) for value in plaintexts], dtype=torch.float32)
    expected = torch.tensor(
        [_scalar_reference(value, rounds, target_bit) for value in plaintexts],
        dtype=torch.float32,
    )[:, None]
    result = public_reference(tensor, rounds, target_bit)
    assert result.shape == (len(plaintexts), 1)
    assert result.dtype == torch.float32
    assert torch.equal(result, expected)


def test_public_reference_supports_bool_without_changing_bits() -> None:
    inputs = torch.randint(0, 2, (7, 64), dtype=torch.int64).bool()
    result = public_reference(inputs, 4, 63)
    assert result.shape == (7, 1)
    assert result.dtype is torch.bool
    assert bool((result | ~result).all())


@pytest.mark.parametrize("rounds", [1, 2, 3, 4, 11, 44])
@pytest.mark.parametrize("target_bit", [0, 31, 32, 63])
def test_support_indices_are_sorted_valid_and_keyed_cipher_invariant(
    rounds: int, target_bit: int
) -> None:
    support = support_indices(rounds, target_bit)
    assert isinstance(support, tuple)
    assert support == tuple(sorted(set(support)))
    assert support and all(0 <= index < 64 for index in support)

    # XOR with a round key is a constant w.r.t. plaintext, so the same set
    # contains every dependency for arbitrary key schedules, not just zero key.
    rng = random.Random(rounds * 100 + target_bit)
    outside = sorted(set(range(64)) - set(support))
    if outside:
        for _ in range(8):
            plaintext = rng.getrandbits(64)
            changed = plaintext ^ (1 << (63 - rng.choice(outside)))
            key = rng.getrandbits(128)
            before = (encrypt(plaintext, key, rounds) >> (63 - target_bit)) & 1
            after = (encrypt(changed, key, rounds) >> (63 - target_bit)) & 1
            assert before == after


def test_support_predictor_is_invariant_to_plaintext_outside_dependencies() -> None:
    rng = random.Random(8173)
    for rounds, target_bit in ((1, 0), (2, 31), (3, 47), (4, 63), (8, 4)):
        model = PlaintextPredictor(
            hidden_width=5,
            rounds=rounds,
            target_bit=target_bit,
            feature_mode="support",
        ).eval()
        dependencies = support_indices(rounds, target_bit)
        untouched = sorted(set(range(64)) - set(dependencies))
        original = rng.getrandbits(64)
        changed = original
        for bit in untouched:
            if rng.randrange(2):
                changed ^= 1 << (63 - bit)
        inputs = torch.tensor([_plaintext(original), _plaintext(changed)], dtype=torch.float32)
        with torch.no_grad():
            logits = model(inputs)
        assert torch.equal(logits[0], logits[1])


def test_residual_absolute_logits_preserve_bce_and_gradients_and_zero_ties() -> None:
    inputs = torch.tensor(
        [_plaintext(value) for value in (0, 1, 2, 0x123456789ABCDEF0)], dtype=torch.float32
    )
    reference = public_reference(inputs, 3, 0)
    residual_logits = torch.tensor(
        [[-2.0], [0.0], [0.75], [0.0]], dtype=torch.float32, requires_grad=True
    )
    residual_labels = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    absolute_labels = torch.remainder(residual_labels + reference, 2.0)

    absolute_logits = (1.0 - 2.0 * reference) * residual_logits
    absolute_loss = F.binary_cross_entropy_with_logits(absolute_logits, absolute_labels)
    residual_loss = F.binary_cross_entropy_with_logits(residual_logits, residual_labels)
    absolute_gradient, = torch.autograd.grad(absolute_loss, residual_logits, retain_graph=True)
    residual_gradient, = torch.autograd.grad(residual_loss, residual_logits)

    assert torch.allclose(absolute_loss, residual_loss, atol=1e-7, rtol=1e-7)
    assert torch.allclose(absolute_gradient, residual_gradient, atol=1e-7, rtol=1e-7)
    zero_residual = residual_logits.detach() == 0
    assert torch.equal(absolute_logits[zero_residual], torch.zeros(2))
    assert not bool((absolute_logits[zero_residual] > 0).any())  # strict-positive ties map to 0
    assert torch.equal(absolute_logits, (1.0 - 2.0 * reference) * residual_logits)


def test_predictor_checkpoint_keys_and_initialization_are_mode_compatible() -> None:
    torch.manual_seed(701)
    direct = PlaintextPredictor(7, rounds=3, target_bit=0, prediction_mode="direct")
    torch.manual_seed(701)
    residual = PlaintextPredictor(
        7, rounds=3, target_bit=0, prediction_mode="public-residual"
    )
    assert isinstance(direct, BitMLP)
    assert direct.state_dict().keys() == residual.state_dict().keys()
    assert all(torch.equal(direct.state_dict()[key], residual.state_dict()[key])
               for key in direct.state_dict())
    assert all(key.startswith("layers.") for key in direct.state_dict())
    assert "indices" not in direct.state_dict()
    assert direct.layers[0].in_features == 64
    assert residual(torch.zeros((2, 64), dtype=torch.float32)).shape == (2, 1)

    support = PlaintextPredictor(7, rounds=3, target_bit=0, feature_mode="support")
    assert support.layers[0].in_features == len(support_indices(3, 0))


def test_plaintext_predictor_and_reference_reject_malformed_inputs() -> None:
    model = PlaintextPredictor(4)
    with pytest.raises(TypeError, match="torch.Tensor"):
        model(cast(torch.Tensor, [[0.0] * 64]))
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros((0, 64), dtype=torch.float32))
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros((2, 63), dtype=torch.float32))
    with pytest.raises(TypeError, match="float32"):
        model(torch.zeros((1, 64), dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        model(torch.full((1, 64), float("nan"), dtype=torch.float32))
    with pytest.raises(ValueError, match="binary"):
        model(torch.full((1, 64), 0.5, dtype=torch.float32))
    with pytest.raises(ValueError, match="shape"):
        public_reference(torch.zeros((1, 63), dtype=torch.bool), 1, 0)
    with pytest.raises(TypeError, match="dtype"):
        public_reference(torch.zeros((1, 64), dtype=torch.int64), 1, 0)
    with pytest.raises(ValueError, match="finite"):
        public_reference(torch.full((1, 64), float("inf")), 1, 0)


@pytest.mark.parametrize("rounds,target_bit", [(0, 0), (45, 0), (True, 0), (1, -1), (1, 64)])
def test_support_indices_rejects_invalid_builtin_integer_contract(
    rounds: object, target_bit: object
) -> None:
    error = TypeError if type(rounds) is not int or type(target_bit) is not int else ValueError
    with pytest.raises(error):
        support_indices(cast(int, rounds), cast(int, target_bit))


def test_residual_mode_rejects_unsupported_reference_rounds_and_bad_modes() -> None:
    with pytest.raises(ValueError, match="rounds 1..4"):
        PlaintextPredictor(rounds=5, prediction_mode="public-residual")
    with pytest.raises(ValueError, match="feature_mode"):
        PlaintextPredictor(feature_mode="unknown")
    with pytest.raises(ValueError, match="prediction_mode"):
        PlaintextPredictor(prediction_mode="unknown")
    with pytest.raises(TypeError, match="built-in int"):
        PlaintextPredictor(rounds=True)
    with pytest.raises(ValueError, match="rounds"):
        public_reference(torch.zeros((1, 64)), 5, 0)


def test_prediction_module_has_only_torch_and_local_model_dependencies() -> None:
    module_path = (
        Path(__file__).resolve().parents[1] / "src/cipher_experiments/models/prediction.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
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
    assert imported_roots <= {
        "torch", "mlp", "logic", "circuit", "structured_simon", "typing", "__future__"}
    assert "inputs" in inspect.signature(PlaintextPredictor.forward).parameters
    assert "key" not in inspect.signature(PlaintextPredictor).parameters
    assert "key" not in inspect.signature(PlaintextPredictor.forward).parameters

"""Synthetic contracts for the fixed-cone SIMON predictor."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from cipher_experiments.ciphers.simon import _round_keys, encrypt
from cipher_experiments.models.structured_simon import StructuredSimonPredictor

_SYNTHETIC_KEY = 0xFFEEDDCCBBAA99887766554433221100


def _plaintexts(count: int = 24) -> tuple[Tensor, list[int]]:
    generator = torch.Generator().manual_seed(2718)
    bits = torch.randint(0, 2, (count, 64), generator=generator, dtype=torch.int64)
    values = [
        sum(int(bit) << (63 - index) for index, bit in enumerate(row))
        for row in bits.tolist()
    ]
    return bits.to(dtype=torch.float32), values


def _install_synthetic_round_key_witness(
    model: StructuredSimonPredictor, rounds: int, key: int = _SYNTHETIC_KEY
) -> None:
    """Install test-only synthetic key bits; never part of production fitting/init."""
    round_keys = _round_keys(key, rounds)
    with torch.no_grad():
        for index, (round_index, bit_index) in enumerate(model.key_coordinates):
            bit = (round_keys[round_index] >> (31 - bit_index)) & 1
            model.key_logits[index] = 12.0 if bit else -12.0


def _expected_bits(plaintexts: list[int], rounds: int, target_bit: int) -> Tensor:
    return torch.tensor(
        [
            (encrypt(plaintext, _SYNTHETIC_KEY, rounds) >> (63 - target_bit)) & 1
            for plaintext in plaintexts
        ],
        dtype=torch.float32,
    ).unsqueeze(1)


@pytest.mark.parametrize("rounds", range(1, 7))
@pytest.mark.parametrize("target_bit", [0, 31, 32, 63])
def test_hard_dag_matches_synthetic_scalar_cipher_witness(
    rounds: int, target_bit: int
) -> None:
    if rounds == 1 and target_bit >= 32:
        with pytest.raises(ValueError, match="no learnable round-key"):
            StructuredSimonPredictor(rounds=rounds, target_bit=target_bit)
        return
    model = StructuredSimonPredictor(rounds=rounds, target_bit=target_bit)
    _install_synthetic_round_key_witness(model, rounds)
    inputs, plaintexts = _plaintexts()

    hard_logits = model.hard_logits(inputs)
    expected = _expected_bits(plaintexts, rounds, target_bit)
    assert hard_logits.shape == (len(plaintexts), 1)
    assert hard_logits.dtype == torch.float32
    assert torch.equal(hard_logits > 0, expected.bool())
    assert bool(torch.isfinite(hard_logits).all())


def test_soft_dag_converges_to_synthetic_key_witness_across_rounds() -> None:
    inputs, plaintexts = _plaintexts(96)
    for rounds, target_bit in ((1, 0), (2, 31), (3, 0), (4, 32), (5, 31), (6, 0)):
        model = StructuredSimonPredictor(rounds=rounds, target_bit=target_bit)
        _install_synthetic_round_key_witness(model, rounds)
        gate_count = sum(model.cone.gate_counts.values())
        assert gate_count <= 406

        probabilities = torch.sigmoid(model(inputs))
        expected = _expected_bits(plaintexts, rounds, target_bit)
        assert torch.max(torch.abs(probabilities - expected)) < 0.003
        assert torch.equal(probabilities > 0.5, expected.bool())


def test_only_sorted_symbolic_key_effects_are_trainable_and_persisted() -> None:
    model = StructuredSimonPredictor(rounds=4, target_bit=0)

    assert model.key_coordinates == tuple(sorted(model.cone.round_key_coordinates))
    assert model.key_logits.shape == (24,)
    assert list(model.named_parameters()) == [("key_logits", model.key_logits)]
    assert list(model.state_dict()) == ["key_logits"]
    assert model.indices.dtype == torch.long
    assert model.indices.tolist() == list(model.cone.plaintext_indices)
    assert "indices" not in model.state_dict()
    assert model.checkpoint_config_sha256 == ""
    assert model.key_logits.dtype == torch.float32
    assert model.key_logits.requires_grad
    assert bool((model.key_logits.abs() > 1.0).all())


def test_parameter_counts_follow_key_coordinates_without_heads_or_extra_state() -> None:
    r3 = StructuredSimonPredictor(rounds=3, target_bit=0)
    r4 = StructuredSimonPredictor(rounds=4, target_bit=0)
    assert r3.key_logits.numel() == 11
    assert r4.key_logits.numel() == 24
    assert sum(parameter.numel() for parameter in r3.parameters()) == 11
    assert sum(parameter.numel() for parameter in r4.parameters()) == 24


@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_initial_synthetic_input_gradients_are_finite_and_nonzero(rounds: int) -> None:
    torch.manual_seed(863 + rounds)
    model = StructuredSimonPredictor(rounds=rounds, target_bit=0)
    inputs, _ = _plaintexts(16)
    logits = model(inputs)
    # An arbitrary all-zero synthetic target checks autograd, not cipher fitting.
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits)
    )
    loss.backward()  # type: ignore[no-untyped-call]
    gradient = model.key_logits.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0.0


def test_half_key_probabilities_produce_half_output_for_one_round() -> None:
    model = StructuredSimonPredictor(rounds=1, target_bit=0)
    with torch.no_grad():
        model.key_logits.zero_()
    inputs, _ = _plaintexts(8)
    soft_probabilities = torch.sigmoid(model(inputs))
    assert torch.equal(soft_probabilities, torch.full((8, 1), 0.5))
    # Zero logits intentionally expose the relaxed q=0.5 XOR collapse; this is
    # a diagnostic of the soft circuit, not a reason to special-case the output.
    assert bool(torch.isfinite(model.hard_logits(inputs)).all())


def test_public_api_accepts_no_actual_round_key() -> None:
    constructor_parameters = inspect.signature(StructuredSimonPredictor).parameters
    assert tuple(constructor_parameters) == ("rounds", "target_bit")
    assert tuple(inspect.signature(StructuredSimonPredictor.forward).parameters) == (
        "self",
        "inputs",
    )


def test_initialization_is_caller_seeded_random_not_fixed_key_lookup() -> None:
    torch.manual_seed(1927)
    first = StructuredSimonPredictor(rounds=3, target_bit=0)
    torch.manual_seed(1927)
    repeated = StructuredSimonPredictor(rounds=3, target_bit=0)
    torch.manual_seed(1928)
    different = StructuredSimonPredictor(rounds=3, target_bit=0)

    assert torch.equal(first.key_logits, repeated.key_logits)
    assert not torch.equal(first.key_logits, different.key_logits)
    assert not bool((first.key_logits == 0).all())
    assert bool(((first.key_logits > 0) | (first.key_logits < 0)).all())


def test_train_eval_and_explicit_hard_path_have_no_mode_or_state_side_effects() -> None:
    model = StructuredSimonPredictor(rounds=3, target_bit=0)
    _install_synthetic_round_key_witness(model, 3)
    inputs, _ = _plaintexts(20)
    saved_logits = model.key_logits.detach().clone()

    model.train()
    train_soft = model(inputs)
    train_hard = model.hard_logits(inputs)
    model.eval()
    eval_soft = model(inputs)
    eval_hard = model.hard_logits(inputs)

    assert torch.equal(train_soft, eval_soft)
    assert torch.equal(train_hard, eval_hard)
    assert torch.equal(saved_logits, model.key_logits)
    assert torch.equal(train_hard > 0, eval_soft > 0)


def test_zero_logit_threshold_is_explicitly_bit_zero() -> None:
    model = StructuredSimonPredictor(rounds=1, target_bit=0)
    with torch.no_grad():
        model.key_logits.fill_(0.0)
    inputs, plaintexts = _plaintexts(4)

    hard = model.hard_logits(inputs)
    expected = torch.tensor(
        [(encrypt(plaintext, 0, 1) >> 63) & 1 for plaintext in plaintexts],
        dtype=torch.bool,
    ).unsqueeze(1)
    assert torch.equal(hard > 0, expected)


@pytest.mark.parametrize("rounds", [0, 17, True, 1.0])
def test_constructor_rejects_non_builtin_or_out_of_range_rounds(rounds: object) -> None:
    error = TypeError if type(rounds) is not int else ValueError
    with pytest.raises(error):
        StructuredSimonPredictor(rounds=cast(int, rounds))


@pytest.mark.parametrize("target_bit", [-1, 64, True, 0.0])
def test_constructor_rejects_non_builtin_or_out_of_range_target(target_bit: object) -> None:
    error = TypeError if type(target_bit) is not int else ValueError
    with pytest.raises(error):
        StructuredSimonPredictor(target_bit=cast(int, target_bit))


@pytest.mark.parametrize("target_bit", range(32, 64))
def test_one_round_key_free_cones_are_rejected(target_bit: int) -> None:
    with pytest.raises(ValueError, match="no learnable round-key"):
        StructuredSimonPredictor(rounds=1, target_bit=target_bit)


def test_forward_rejects_invalid_inputs_and_wrong_device() -> None:
    model = StructuredSimonPredictor()
    with pytest.raises(TypeError, match="torch.Tensor"):
        model(cast(Tensor, [[0.0] * 64]))
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros((1, 63), dtype=torch.float32))
    with pytest.raises(ValueError, match="nonempty"):
        model(torch.empty((0, 64), dtype=torch.float32))
    with pytest.raises(TypeError, match="float32"):
        model(torch.zeros((1, 64), dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        model(torch.full((1, 64), float("nan"), dtype=torch.float32))
    with pytest.raises(ValueError, match="binary"):
        model(torch.full((1, 64), 0.5, dtype=torch.float32))
    with pytest.raises(ValueError, match="same device"):
        model.to(device="meta")(torch.zeros((1, 64), dtype=torch.float32))


def test_model_imports_no_cipher_dataset_or_training_modules() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src/cipher_experiments/models/structured_simon.py"
    )
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
    assert imported_roots <= {"torch", "representations", "prediction", "__future__"}

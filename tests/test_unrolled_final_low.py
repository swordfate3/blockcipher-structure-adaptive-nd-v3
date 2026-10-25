"""Final-round low-channel ablation for plaintext-xor unrolled prediction."""
from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from cipher_experiments.config import ExperimentConfig
from cipher_experiments.models.prediction import build_predictor
from cipher_experiments.models.unrolled_simon import UnrolledSimonPredictor


def _inputs(count: int = 9) -> Tensor:
    generator = torch.Generator().manual_seed(14821)
    return torch.randint(0, 2, (count, 64), generator=generator).float()


def _rotate_left(values: Tensor, amount: int) -> Tensor:
    return torch.cat((values[:, amount:], values[:, :amount]), dim=1)


def _manual_final_low_raw(model: UnrolledSimonPredictor, inputs: Tensor) -> Tensor:
    """Recreate unrolled recursion, masking only final-round low features."""
    high, low = inputs[:, :32], inputs[:, 32:]
    output: Tensor | None = None
    for round_index in range(model.rounds):
        local = torch.stack((low, _rotate_left(high, 1), _rotate_left(high, 8),
                             _rotate_left(high, 2)), dim=-1)
        local = local.mul(2.0).sub(1.0)
        if round_index == model.rounds - 1:
            # Out-of-place construction preserves autograd and the untouched channels.
            local = torch.cat((torch.zeros_like(local[..., :1]), local[..., 1:]), dim=-1)
        condition = model.conditions[round_index].unsqueeze(0).expand(inputs.shape[0], -1, -1)
        output = model.cell(torch.cat((local, condition), dim=-1)).squeeze(-1)
        low, high = high, torch.sigmoid(output)
    assert output is not None
    return output[:, model.target_bit : model.target_bit + 1]


def _manual_final_low_xor(model: UnrolledSimonPredictor, inputs: Tensor) -> Tensor:
    raw = _manual_final_low_raw(model, inputs)
    sign = 1.0 - 2.0 * inputs[:, model.target_bit : model.target_bit + 1]
    return sign * raw


@pytest.mark.parametrize("patch", [
    {"dataset": "synthetic"},
    {"model_kind": "mlp"},
    {"rounds": 1},
    {"rounds": 3},
    {"target_bit": 32},
    {"target_bit": 64},
    {"model_features": "support"},
])
def test_config_rejects_out_of_contract_mode(patch: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "name": "invalid", "dataset": "simon", "model_kind": "simon-unrolled",
        "prediction_mode": "plaintext-xor-no-final-low", "rounds": 2, "target_bit": 0,
    }
    values.update(patch)
    with pytest.raises(ValueError):
        ExperimentConfig(**values)


def test_factory_and_constructor_enforce_mode_contract() -> None:
    with pytest.raises(ValueError, match="full/direct"):
        build_predictor(model_kind="simon-unrolled", feature_mode="support",
                        rounds=2, target_bit=0,
                        prediction_mode="plaintext-xor-no-final-low")
    with pytest.raises(ValueError, match="model_kind=simon-unrolled"):
        build_predictor(model_kind="mlp", rounds=2, target_bit=0,
                        prediction_mode="plaintext-xor-no-final-low")
    for kwargs in ({"rounds": 1}, {"rounds": 3}, {"target_bit": 32}):
        with pytest.raises(ValueError, match="rounds=2"):
            cast(Any, UnrolledSimonPredictor)(
                prediction_mode="plaintext-xor-no-final-low", **kwargs)


def test_new_mode_has_old_xor_initialization_and_strict_state_contract() -> None:
    torch.manual_seed(593)
    old = UnrolledSimonPredictor(32, rounds=2, target_bit=11,
                                 prediction_mode="plaintext-xor")
    torch.manual_seed(593)
    ablated = UnrolledSimonPredictor(32, rounds=2, target_bit=11,
                                     prediction_mode="plaintext-xor-no-final-low")
    assert sum(parameter.numel() for parameter in old.parameters()) == 577
    assert sum(parameter.numel() for parameter in ablated.parameters()) == 577
    assert set(old.state_dict()) == set(ablated.state_dict())
    assert all(torch.equal(value, ablated.state_dict()[name])
               for name, value in old.state_dict().items())
    assert dict(old.named_buffers()).keys() == dict(ablated.named_buffers()).keys()
    ablated.load_state_dict(old.state_dict(), strict=True)
    inputs = _inputs()
    assert torch.equal(ablated(inputs), _manual_final_low_xor(ablated, inputs))


def test_cell_inputs_only_mask_final_centered_low_channel() -> None:
    torch.manual_seed(1803)
    old = UnrolledSimonPredictor(9, rounds=2, target_bit=3,
                                 prediction_mode="plaintext-xor")
    ablated = UnrolledSimonPredictor(9, rounds=2, target_bit=3,
                                     prediction_mode="plaintext-xor-no-final-low")
    ablated.load_state_dict(old.state_dict(), strict=True)
    inputs = _inputs(6)
    old_cells: list[Tensor] = []
    new_cells: list[Tensor] = []
    old_hook = old.cell.register_forward_pre_hook(
        lambda _module, args: old_cells.append(cast(Tensor, args[0]).detach().clone()))
    new_hook = ablated.cell.register_forward_pre_hook(
        lambda _module, args: new_cells.append(cast(Tensor, args[0]).detach().clone()))
    try:
        old(inputs)
        ablated(inputs)
    finally:
        old_hook.remove()
        new_hook.remove()

    assert len(old_cells) == len(new_cells) == 2
    # Distinct examples carry distinct actual first-round latent states into round two.
    assert not torch.equal(inputs[0], inputs[1])
    assert bool((old_cells[1][0, :, 1:4] != old_cells[1][1, :, 1:4]).any())
    # First recurrence is untouched; final positions retain all rotations and conditions.
    assert torch.equal(new_cells[0], old_cells[0])
    assert torch.equal(new_cells[1][..., 1:], old_cells[1][..., 1:])
    assert torch.equal(new_cells[1][..., :1], torch.zeros_like(new_cells[1][..., :1]))
    assert bool((old_cells[1][..., :1] != 0).any())
    assert not torch.equal(old(inputs), ablated(inputs))


def test_signed_output_and_absolute_bce_match_manual_final_low_prediction() -> None:
    torch.manual_seed(442)
    target_bit = 27
    model = UnrolledSimonPredictor(12, rounds=2, target_bit=target_bit,
                                   prediction_mode="plaintext-xor-no-final-low")
    inputs = _inputs(20)
    labels = torch.randint(0, 2, (20, 1), generator=torch.Generator().manual_seed(704)).float()
    raw = _manual_final_low_raw(model, inputs)
    sign = 1.0 - 2.0 * inputs[:, target_bit : target_bit + 1]
    expected = sign * raw
    output = model(inputs)
    assert torch.equal(output, expected)
    # The sign is exactly the existing plaintext-xor convention.
    assert torch.equal(output, sign * raw)
    expected_bce = F.binary_cross_entropy_with_logits(expected, labels)
    assert torch.equal(F.binary_cross_entropy_with_logits(output, labels), expected_bce)
    equivalent_labels = torch.logical_xor(
        labels.bool(), inputs[:, target_bit : target_bit + 1].bool()
    ).float()
    torch.testing.assert_close(
        F.binary_cross_entropy_with_logits(output, labels),
        F.binary_cross_entropy_with_logits(output * sign, equivalent_labels),
        rtol=1e-6, atol=1e-7,
    )


def test_backward_is_finite_without_claiming_plaintext_gradient_independence() -> None:
    torch.manual_seed(771)
    model = UnrolledSimonPredictor(8, rounds=2, target_bit=5,
                                   prediction_mode="plaintext-xor-no-final-low")
    inputs = _inputs(12).requires_grad_()
    output = model(inputs)
    output.sum().backward()
    assert inputs.grad is not None and bool(torch.isfinite(inputs.grad).all())
    assert model.conditions.grad is not None and bool(torch.isfinite(model.conditions.grad).all())
    # No independence claim: input may affect the first-round latent state and later output.
    assert float(inputs.grad.abs().sum()) > 0.0

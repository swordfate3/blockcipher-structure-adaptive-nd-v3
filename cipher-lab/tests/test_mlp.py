"""Artificial CPU bits only: no labels, fitting, cipher calls or optimizer."""
from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import Tensor, nn

from cipher_lab.models import BitMLP
from cipher_lab.training_config import MLPSpec

torch.set_num_threads(1)


def test_two_equal_relu_layers_and_backward_without_update() -> None:
    model = BitMLP(MLPSpec(5))
    linear = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linear] == [(64, 5), (5, 5), (5, 1)]
    assert sum(isinstance(m, nn.ReLU) for m in model.modules()) == 2
    inputs = (torch.arange(128).reshape(2, 64) % 2).float().requires_grad_()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    logits = model(inputs)
    assert logits.shape == (2, 1) and torch.isfinite(logits).all()
    logits.sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.equal(before[name], parameter)


@pytest.mark.parametrize("inputs", [torch.zeros(64), torch.zeros(0, 64), torch.zeros(2, 63),
                                  torch.full((1, 64), float('nan')),
                                  torch.full((1, 64), float('inf')),
                                  torch.full((1, 64), .5), torch.zeros(1, 64).long(),
                                  torch.zeros(1, 64).to_sparse()])
def test_reject_invalid_input(inputs: Tensor) -> None:
    with pytest.raises((ValueError, TypeError)):
        BitMLP(MLPSpec(2))(inputs)


def test_requires_task1_typed_spec() -> None:
    with pytest.raises(ValueError):
        BitMLP(cast(MLPSpec, 8))


def test_nonfinite_model_output_is_not_success() -> None:
    model = BitMLP(MLPSpec(2))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float('nan'))
    with pytest.raises(ValueError, match="finite"):
        model(torch.zeros(1, 64))

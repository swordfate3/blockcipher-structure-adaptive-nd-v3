"""Pure paired inputs, exact tails, and one metered forward pass only."""
from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch
from test_training_steps import make_meter

from cipher_lab.models import BitMLP
from cipher_lab.training_config import MLPSpec
from cipher_lab_plugins.prediction_common.metrics import (
    EvaluationPanels,
    PublicModelSpec,
    TrainBaseline,
    evaluate_batches,
)
from cipher_lab_plugins.residual.tests.test_residual_config import poison
from cipher_lab_plugins.residual.tests.test_residual_formats import bundle, task

__all__ = ['poison']


def test_absolute_flip_zero_tie_and_saved_h_no_second_forward() -> None:
    from cipher_lab_plugins.prediction_common.paired import absolute_logits
    h = torch.tensor([[1.], [0.]])
    assert torch.equal(absolute_logits(torch.tensor([[0.], [-2.]]), h),
                       torch.tensor([[0.], [-2.]]))
    spec = PublicModelSpec(MLPSpec(2), (task(),), 'residual')
    model = BitMLP(MLPSpec(2))
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        cast(torch.Tensor, model.layers[4].bias).fill_(2.)
    outputs: dict[str, torch.Tensor] = {}
    meter = make_meter(rows=2)
    result = evaluate_batches(model, EvaluationPanels.from_bundle(bundle(),
        {'SIMON': TrainBaseline(2, 1)}), spec, meter, 1, outputs=outputs)
    assert torch.equal(outputs['SIMON'], torch.tensor([[-2.], [2.]]))
    assert cast(dict[str, dict[str, float]], result['tasks'])['SIMON']['acc'] == 1.
    assert meter.snapshot()['completed_forward_rows'] == 2
    with pytest.raises(ValueError):
        evaluate_batches(model, EvaluationPanels(spec.method, (task(),), bundle().panels,
            {'SIMON': TrainBaseline(2, 1)}), spec, make_meter(), 1)


@pytest.mark.parametrize('b,c,p', [(0, 0, 1.), (4, 0, .0625), (6, 0, .015625),
                                  (0, 6, 1.), (3, 3, .65625)])
def test_exact_sparse_and_zero(b: int, c: int, p: float) -> None:
    from cipher_lab_plugins.prediction_common.paired import paired_accuracy
    direct = torch.tensor([-1.] * b + [1.] * c + [1.]).reshape(-1, 1)
    residual = torch.tensor([1.] * b + [-1.] * c + [1.]).reshape(-1, 1)
    stats = paired_accuracy(direct, residual, torch.ones_like(direct))
    assert stats['b'] == b and stats['c'] == c
    assert stats['sum_d'] == b - c and stats['sum_d2'] == b + c
    assert stats['delta'] == pytest.approx((b - c) / (b+c+1))
    assert stats['p_one_sided'] == pytest.approx(p)
    assert stats['normal_95_interval'] is None
    assert math.isfinite(cast(dict[str, float], stats['loss_difference'])['sum'])


def test_large_exact_tail_and_normal_interval() -> None:
    from cipher_lab_plugins.prediction_common.paired import binomial_tail, paired_accuracy
    assert binomial_tail(1000000, 1000000) == pytest.approx(.5002820947565, abs=2e-9)
    assert binomial_tail(1010000, 990000) < 1e-40
    assert binomial_tail(990000, 1010000) > .999999999
    assert binomial_tail(2000000, 0) == 0.
    logits = torch.cat((torch.ones(25, 1), -torch.ones(25, 1)))
    stats = paired_accuracy(-logits, logits, torch.ones(50, 1))
    assert stats['normal_95_interval'] is not None


def test_train_only_frozen_diagnostics() -> None:
    from cipher_lab_plugins.prediction_common.paired import reference_diagnostics
    y = torch.tensor([[0.], [1.]])
    h = 1 - y
    frozen = TrainBaseline(2, 2)  # residual always one on TRAIN
    result: Any = reference_diagnostics(y, h, frozen)
    assert result['fixed_flip']['acc'] == 1.
    assert result['residual_probability']['acc'] == 1.
    assert result['train_residual_constant'] == 1

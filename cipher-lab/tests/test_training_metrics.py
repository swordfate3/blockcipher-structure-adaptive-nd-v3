"""Artificial CPU observations only; no fitting or label generation."""
from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import cast

import pytest
import torch
from torch import Tensor

from cipher_lab.config import Policy
from cipher_lab.data.bundles import DatasetBundle, PredictionTask
from cipher_lab.evaluation import (
    EvaluationPanels,
    PublicModelSpec,
    TrainBaseline,
    binary_metrics,
    evaluate_batches,
)
from cipher_lab.execution import Clocks, Meter, Stopped
from cipher_lab.models import BitMLP, SharedM
from cipher_lab.training_config import TASK_FAMILIES, MLPSpec, SharedMSpec, TaskSpec

torch.set_num_threads(1)
TASK = PredictionTask('SIMON', 3, 7, 'cipher')
SPEC = MLPSpec(3)


def meter(limit: int = 100) -> Meter:
    return Meter(Policy(2, 100., 100., '2099-01-01T00:00:00+00:00',
                        0, 0, 1000000, 100000, 0, limit),
                 Clocks(lambda: 0., lambda: 0., lambda: datetime(2026, 1, 1, tzinfo=UTC)),
                 lambda: 0)


def panels(rows: int = 3) -> EvaluationPanels:
    bundle = DatasetBundle('bit-prediction-v1', (TASK,),
                           {'SIMON': (torch.zeros(rows, 64), torch.zeros(rows, 1))})
    return EvaluationPanels.from_bundle(bundle, {'SIMON': TrainBaseline(4, 3)})


def test_threshold_stable_bce_and_train_baseline() -> None:
    result = binary_metrics(torch.tensor([[0.], [1000.], [-1000.]]),
                            torch.tensor([[0.], [0.], [1.]]), majority=1, probability=.7)
    assert result['rows'] == 3 and result['acc'] == pytest.approx(1 / 3)
    assert result['bce'] == pytest.approx((2000 + math.log(2)) / 3)
    assert result['baseline_acc'] == pytest.approx(1 / 3)
    assert result['baseline_bce'] == pytest.approx(-(2*math.log(.3)+math.log(.7))/3)
    baseline = TrainBaseline(4, 3)
    assert baseline.majority == 1 and baseline.probability == .7
    assert TrainBaseline(2, 1).majority == 0
    assert TrainBaseline(1, 0).probability == .25


@pytest.mark.parametrize('logits,labels', [
    (torch.zeros(0, 1), torch.zeros(0, 1)),
    (torch.zeros(2), torch.zeros(2)),
    (torch.zeros(2, 1), torch.zeros(1, 1)),
    (torch.ones(1, 1, dtype=torch.int64), torch.zeros(1, 1)),
    (torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.int64)),
    (torch.full((1, 1), float('nan')), torch.zeros(1, 1)),
    (torch.zeros(1, 1), torch.full((1, 1), .5)),
    (torch.zeros(1, 1).to_sparse(), torch.zeros(1, 1)),
])
def test_invalid_metrics(logits: Tensor, labels: Tensor) -> None:
    with pytest.raises(ValueError):
        binary_metrics(logits, labels, majority=0, probability=.5)


@pytest.mark.parametrize('majority,probability', [(True, .5), (2, .5), (0, 0.),
                                               (0, 1.), (0, float('nan')), (0, True)])
def test_invalid_baseline_arguments(majority: int, probability: float) -> None:
    with pytest.raises(ValueError):
        binary_metrics(torch.zeros(1, 1), torch.zeros(1, 1),
                       majority=majority, probability=probability)


@pytest.mark.parametrize('rows,ones', [(0, 0), (2, 3), (True, 0), (2, -1)])
def test_invalid_train_counts(rows: int, ones: int) -> None:
    with pytest.raises(ValueError):
        TrainBaseline(rows, ones)


def test_batched_evaluation_charges_rows_preserves_modes_and_no_grad() -> None:
    model = BitMLP(SPEC)
    model.train()
    model.layers[0].eval()
    modes = [module.training for module in model.modules()]
    observed: list[tuple[int, bool, bool]] = []
    ledger = meter()

    def hook(module: torch.nn.Module, args: tuple[Tensor, ...], output: Tensor) -> None:
        observed.append((args[0].shape[0], module.training, output.requires_grad))
        assert cast(int, ledger.snapshot()['attempted_forward_rows']) >= sum(
            x[0] for x in observed)

    handle = model.register_forward_hook(hook)
    try:
        result = evaluate_batches(model, panels(), PublicModelSpec(SPEC, (TASK,)), ledger, 2)
    finally:
        handle.remove()
    assert observed == [(2, False, False), (1, False, False)]
    assert [module.training for module in model.modules()] == modes
    assert ledger.snapshot()['completed_forward_rows'] == 3
    assert result['rows'] == 3
    assert result['tasks']['SIMON']['baseline_acc'] == 0.  # type: ignore[index]


def test_failed_forward_not_refunded_or_completed_and_restores_mode() -> None:
    model = BitMLP(SPEC)
    ledger = meter()
    def fail(module: torch.nn.Module, args: tuple[Tensor, ...], output: Tensor) -> Tensor:
        return torch.full_like(output, float('nan'))
    handle = model.register_forward_hook(fail)
    try:
        with pytest.raises(ValueError):
            evaluate_batches(model, panels(), PublicModelSpec(SPEC, (TASK,)), ledger, 2)
    finally:
        handle.remove()
    assert model.training
    assert ledger.snapshot()['attempted_forward_rows'] == 2
    assert ledger.snapshot()['completed_forward_rows'] == 0


def test_budget_stops_before_forward() -> None:
    ledger = meter(1)
    with pytest.raises(Stopped, match='FORWARD_ROWS_LIMIT'):
        evaluate_batches(BitMLP(SPEC), panels(), PublicModelSpec(SPEC, (TASK,)), ledger, 2)
    assert ledger.snapshot()['attempted_forward_rows'] == 0


@pytest.mark.parametrize('task', [PredictionTask('SIMON', 4, 7, 'cipher'),
                                 PredictionTask('SIMON', 3, 8, 'cipher'),
                                 PredictionTask('SIMON', 3, 7, 'null')])
def test_wrong_target_rejected_before_forward(task: PredictionTask) -> None:
    ledger = meter()
    with pytest.raises(ValueError):
        evaluate_batches(BitMLP(SPEC), panels(), PublicModelSpec(SPEC, (task,)), ledger, 2)
    assert ledger.snapshot()['attempted_forward_rows'] == 0


def test_shared_m_public13_rounds_and_row_unit() -> None:
    tasks = tuple(TaskSpec(f.cipher, 1, 1) for f in TASK_FAMILIES)
    spec = SharedMSpec(2, 2, 1, (1,), 2, 2)
    bundle = DatasetBundle('shared-m-four-task-v1', tasks,
                           {t.cipher: (torch.zeros(2, 4, 2, 64), torch.tensor([[0.], [1.]]))
                            for t in tasks})
    context = EvaluationPanels.from_bundle(bundle, {t.cipher: TrainBaseline(2, 1) for t in tasks})
    model = SharedM(spec)
    seen: list[int] = []
    def hook(module: torch.nn.Module, args: tuple[Tensor, ...],
             kwargs: dict[str, object]) -> None:
        assert args[1].shape == (13,)
        seen.append(int(str(kwargs['rounds'])))
    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    ledger = meter()
    try:
        result = evaluate_batches(model, context, PublicModelSpec(spec, tasks), ledger, 1)
    finally:
        handle.remove()
    assert seen == [1]*8 and result['rows'] == 8
    assert ledger.snapshot()['completed_forward_rows'] == 8


@pytest.mark.parametrize('kind', ['input_nan', 'input_half', 'input_shape', 'label_half',
                                 'label_rows', 'batch_bool', 'model_dtype', 'missing_baseline'])
def test_invalid_context_before_reservation(kind: str) -> None:
    context = panels()
    x, y = context.panels['SIMON']
    model = BitMLP(SPEC)
    baseline = dict(context.baseline)
    batch_size = 2
    if kind == 'input_nan':
        x[0, 0] = float('nan')
    elif kind == 'input_half':
        x[0, 0] = .5
    elif kind == 'input_shape':
        x = torch.zeros(3, 63)
    elif kind == 'label_half':
        y[0, 0] = .5
    elif kind == 'label_rows':
        y = y[:2]
    elif kind == 'batch_bool':
        batch_size = True
    elif kind == 'model_dtype':
        model.double()
    else:
        baseline = {}
    bad = EvaluationPanels(context.method, context.task_declarations, {'SIMON': (x, y)}, baseline)
    ledger = meter()
    with pytest.raises(ValueError):
        evaluate_batches(model, bad, PublicModelSpec(SPEC, (TASK,)), ledger, batch_size)
    assert ledger.snapshot()['attempted_forward_rows'] == 0


def test_partial_failure_keeps_completed_previous_batch() -> None:
    model = BitMLP(SPEC)
    ledger = meter()
    count = 0
    def hook(module: torch.nn.Module, args: tuple[Tensor, ...], output: Tensor) -> Tensor:
        nonlocal count
        count += 1
        return output if count == 1 else output[:0]
    handle = model.register_forward_hook(hook)
    try:
        with pytest.raises(ValueError):
            evaluate_batches(model, panels(), PublicModelSpec(SPEC, (TASK,)), ledger, 2)
    finally:
        handle.remove()
    assert ledger.snapshot()['attempted_forward_rows'] == 3
    assert ledger.snapshot()['completed_forward_rows'] == 2

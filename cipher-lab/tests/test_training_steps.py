"""Single artificial batches; optimizer.step is mocked, never repeated fitting."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from unittest.mock import Mock

import pytest
import torch

from cipher_lab.config import Policy
from cipher_lab.data.bundles import PredictionTask
from cipher_lab.execution import Clocks, Meter, Stopped
from cipher_lab.models import BitMLP
from cipher_lab.training.steps import BestSelection, make_optimizer, train_step
from cipher_lab.training_config import MLPSpec

torch.set_num_threads(1)
TASK = PredictionTask('SIMON', 3, 0, 'null')


def make_meter(updates: int = 100, rows: int = 10000) -> Meter:
    return Meter(Policy(2, 60., 60., '2031-01-01T00:00:00Z', 0, 0,
                        1000000, 10000000, updates, rows),
                 Clocks(lambda: 0., lambda: 0.,
                        lambda: datetime(2030, 1, 1, tzinfo=UTC)), lambda: 0)


def test_strict_selection_earliest_tie() -> None:
    best = BestSelection()
    assert best.consider(1, .7)
    assert not best.consider(2, .7)
    assert not best.consider(3, .8)
    assert best.epoch == 1 and best.dev_bce == .7
    assert best.consider(4, .6)
    assert best.epoch == 4


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.1])
def test_selection_rejects_nonfinite(value: float) -> None:
    with pytest.raises(ValueError):
        BestSelection().consider(1, value)


@pytest.mark.parametrize('failure', ['none', 'optimizer', 'nan', 'budget'])
def test_step_attempt_completion_clip_and_fixed_errors(failure: str) -> None:
    model = BitMLP(MLPSpec(2))
    optimizer = make_optimizer(model, .01)
    assert isinstance(optimizer, torch.optim.Adam)
    step = Mock(side_effect=RuntimeError('PRIVATE') if failure == 'optimizer' else None)
    optimizer.step = step  # type: ignore[method-assign]
    meter = make_meter(updates=0 if failure == 'budget' else 1)
    if failure == 'nan':
        with torch.no_grad():
            next(model.parameters()).fill_(float('nan'))
    if failure in ('optimizer', 'nan'):
        with pytest.raises(ValueError, match='^TRAIN_STEP_FAILED$'):
            train_step(model, torch.zeros(2, 64), torch.ones(2, 1), TASK, optimizer, meter)
    elif failure == 'budget':
        with pytest.raises(Stopped, match='OPTIMIZER_UPDATES_LIMIT'):
            train_step(model, torch.zeros(2, 64), torch.ones(2, 1), TASK, optimizer, meter)
    else:
        loss = train_step(model, torch.zeros(2, 64), torch.ones(2, 1), TASK, optimizer, meter)
        assert loss >= 0 and step.call_count == 1
        norm = torch.linalg.vector_norm(torch.cat([
            cast(torch.Tensor, p.grad).flatten() for p in model.parameters()]))
        assert norm <= 1.00001
    stats = meter.snapshot()
    assert stats['attempted_optimizer_updates'] == (0 if failure == 'budget' else 1)
    assert stats['completed_optimizer_updates'] == (1 if failure == 'none' else 0)
    if failure == 'optimizer':
        assert stats['completed_forward_rows'] == 2


def test_step_partial_forward_failure_not_refunded(monkeypatch: pytest.MonkeyPatch) -> None:
    model = BitMLP(MLPSpec(2))
    def fail(*args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError('PRIVATE')
    monkeypatch.setattr(model, 'forward', fail)
    meter = make_meter()
    with pytest.raises(ValueError, match='^TRAIN_STEP_FAILED$'):
        train_step(model, torch.zeros(2, 64), torch.ones(2, 1), TASK,
                   make_optimizer(model, .01), meter)
    assert meter.snapshot()['attempted_forward_rows'] == 2
    assert meter.snapshot()['completed_forward_rows'] == 0


def test_forward_limit_keeps_reserved_update_without_optimizer_step() -> None:
    model = BitMLP(MLPSpec(2))
    optimizer = make_optimizer(model, .01)
    step = Mock()
    optimizer.step = step  # type: ignore[method-assign]
    meter = make_meter(rows=0)
    with pytest.raises(Stopped, match='FORWARD_ROWS_LIMIT'):
        train_step(model, torch.zeros(2, 64), torch.ones(2, 1), TASK, optimizer, meter)
    assert step.call_count == 0
    assert meter.snapshot()['attempted_optimizer_updates'] == 1
    assert meter.snapshot()['completed_optimizer_updates'] == 0
    assert meter.snapshot()['attempted_forward_rows'] == 0

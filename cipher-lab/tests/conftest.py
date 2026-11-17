"""Core migration regressions are fail-closed; science still needs explicit permission.

Unsafe original cases are retained unchanged, marked science, and excluded by default.
The artificial method state-machine cases bind manual bias outputs and dummy
optimizer objects; no actual network forward/backward/Adam can execute.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest
from cipher_lab_plugins.prediction_common.tests.guards import science_guard as science_guard


@pytest.fixture(autouse=True)
def task1_core_isolation(request: pytest.FixtureRequest,
                         monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if request.node.get_closest_marker('science') is not None:
        yield
        return
    guarded = {'test_bundles.py', 'test_training_data.py',
               'test_prediction_training.py', 'test_shared_m_training.py',
               'test_training_cli.py', 'test_training_report.py'}
    if request.node.path.name not in guarded:
        yield
        return
    # Apply the actual test-only guard fixture using pytest dependency ordering.
    request.getfixturevalue('science_guard')
    import torch
    from torch import Tensor

    from cipher_lab.models import BitMLP, SharedM

    def manual(model: Any, inputs: Tensor, *args: object, **kwargs: object) -> Tensor:
        bias = model.state_dict().get('layers.4.bias', torch.zeros(1))
        return cast(Tensor, bias.detach().reshape(1, 1).expand(len(inputs), 1).clone())

    monkeypatch.setattr(BitMLP, 'forward', manual)
    monkeypatch.setattr(SharedM, 'forward', manual)
    if request.node.path.name in {'test_prediction_training.py', 'test_shared_m_training.py',
                                  'test_training_report.py'}:
        from cipher_lab.training import prediction, shared_m
        monkeypatch.setattr(prediction, 'make_optimizer', lambda *args: object())
        monkeypatch.setattr(shared_m, 'make_optimizer', lambda *args: object())
    yield


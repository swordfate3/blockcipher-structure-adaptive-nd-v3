"""Vector reporting must stay read-only and cannot invoke a hidden model pass."""
from __future__ import annotations

import ast
import inspect
from typing import Any

import pytest
import torch

from cipher_experiments.models.simon_vector import SimonVectorPredictor
from cipher_experiments.prediction_reports import experiment_kind, model_metadata


def test_vector_metadata_does_not_read_train_or_run_model(monkeypatch: pytest.MonkeyPatch) -> None:
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = SimonVectorPredictor()
    model.train()
    original = {n: t.detach().clone() for n, t in model.state_dict().items()}
    rng = torch.get_rng_state().clone()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("metadata must not invoke a forward")

    monkeypatch.setattr(model, "forward", forbidden)
    metadata = model_metadata(model, model_kind="simon-vector", prediction_mode="direct",
                              rounds=6, target_bit=0, x_train=torch.tensor(float('nan')),
                              y_train=torch.tensor(float('nan')), device=torch.device("cpu"))
    assert metadata is not None
    assert (metadata['nodes'], metadata['depth'], metadata['parameters']) == (406, 24, 10529)
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(original[n], t) for n, t in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert experiment_kind('simon', 'simon-vector') == (
        'simon-vector-target-dag-final-bit-supervision')


def test_vector_dependency_boundary() -> None:
    import cipher_experiments.models.simon_vector as model_module
    import cipher_experiments.representations.simon_vector as public_module

    def imports(module: Any) -> set[str]:
        tree = ast.parse(inspect.getsource(module))
        names = {n.module for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom) and n.module is not None}
        names.update(a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names)
        return names

    assert imports(model_module) <= {'__future__', 'math', 'typing', 'torch', 'torch.nn',
                                     'torch.nn.functional', 'representations.simon_vector',
                                     'prediction'}
    assert imports(public_module) <= {'__future__', 'dataclasses', 'simon_cone'}

"""Metadata-only production construction, never numerical forward or optimizer."""
from __future__ import annotations

from typing import Any

import pytest
import torch
from cipher_lab_plugins.innovation1.config import Recipe


def test_cpu_construct_shared_default_and_isolated_seed() -> None:
    from cipher_lab_plugins.innovation1.models import construct
    before = torch.random.get_rng_state().clone()
    m = construct(Recipe(), 'M-best')
    assert sum(p.numel() for p in m.parameters()) == 1087509
    assert torch.equal(before, torch.random.get_rng_state())
    a = construct(Recipe(), 'D-SIMECK-best')
    b = construct(Recipe(), 'D-SIMON-best')
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    for slot, size in [('D-PRESENT-best', 651073), ('D-SKINNY-best', 289313)]:
        model = construct(Recipe(), slot)
        assert sum(p.numel() for p in model.parameters()) == size
    with pytest.raises(ValueError):
        construct(Recipe(), 'bad')


def test_runtime_update_wiring_with_explicit_artificial_objects(monkeypatch: Any) -> None:
    """Call real adapter body with fake operations, no genuine NN/Adam/autograd."""
    from cipher_lab_plugins.innovation1.models import TorchRuntime
    events: list[str] = []
    class FakeLoss:
        def backward(self) -> None:
            events.append('backward')
        def detach(self) -> Any:
            return self
        def cpu(self) -> float:
            return 0.7
    class FakeModel:
        def train(self) -> None:
            events.append('train')
        def parameters(self) -> list[Any]:
            return []
    class FakeOptimizer:
        param_groups = [{'lr': 0.}]
        def zero_grad(self, **kwargs: Any) -> None:
            events.append('zero')
        def step(self) -> None:
            events.append('step')
    class FakeResources:
        def reserve(self, kind: str, amount: int) -> None:
            events.append(f'reserve-{kind}-{amount}')
        def complete(self, kind: str, amount: int) -> None:
            events.append(f'complete-{kind}-{amount}')
        def check(self) -> None:
            events.append('check')
    runtime = object.__new__(TorchRuntime)
    runtime.recipe = Recipe()
    runtime.device = torch.device('cpu')
    runtime.models = {'M-best': FakeModel()}  # type: ignore[dict-item]
    runtime.optimizers = {'M-best': FakeOptimizer()}
    # Fake loss is opaque and non-numerical; finite checks are artificial evidence.
    monkeypatch.setattr(torch.nn.functional, 'binary_cross_entropy_with_logits',
                        lambda *a, **k: FakeLoss())
    original_finite = torch.isfinite
    monkeypatch.setattr(torch, 'isfinite', lambda x: torch.tensor(True) if isinstance(
        x, FakeLoss) else original_finite(x))
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_',
                        lambda *a, **k: events.append('clip'))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: events.append('sync'))
    monkeypatch.setattr(runtime, '_forward', lambda *a: torch.zeros(2, 1))
    value = runtime.update('M-best', Recipe().tasks[0], torch.zeros(2, 4, 2, 64),
                           torch.zeros(2, 1), 0.001, FakeResources())
    assert value == 0.7
    assert events.index('complete-forward_rows-2') < events.index('backward')
    assert events.index('backward') < events.index('clip') < events.index('step')
    assert events[-2:] == ['sync', 'zero']
    assert runtime.optimizers['M-best'].param_groups[0]['lr'] == 0.001


def test_public_card_device_dtype_rounds_and_d_no_extra_inputs() -> None:
    from cipher_lab_plugins.innovation1.models import TorchRuntime

    from cipher_lab.representations.tasks import public_features
    calls: list[Any] = []
    class ManualCallable:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            return 'manual-logit'
    runtime = object.__new__(TorchRuntime)
    runtime.models = {'M-best': ManualCallable(),  # type: ignore[dict-item]
                      'D-SKINNY-best': ManualCallable()}  # type: ignore[dict-item]
    x = torch.zeros(2, 4, 2, 64)
    task = Recipe().tasks[-1]
    assert runtime._forward('M-best', task, x) == 'manual-logit'
    args, kwargs = calls[-1]
    assert args[0] is x and len(args) == 2
    assert args[1].device == x.device and args[1].dtype == x.dtype
    assert args[1].tolist() == list(public_features(task.cipher))
    assert kwargs == {'rounds': 8}
    assert runtime._forward('D-SKINNY-best', task, x) == 'manual-logit'
    assert calls[-1] == ((x,), {})


def test_empty_inference_runtime_does_not_construct_optimizer(monkeypatch: Any) -> None:
    from cipher_lab_plugins.innovation1.models import TorchRuntime
    runtime = TorchRuntime(Recipe(), 'cuda:0', inference_only=True)
    assert runtime.models == {} and runtime.optimizers == {}
    with pytest.raises(ValueError):
        TorchRuntime(Recipe(), 'cpu', inference_only=True)

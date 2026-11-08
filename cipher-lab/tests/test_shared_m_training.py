"""Four-task artificial groups; only the named closed loop does Adam steps."""
from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import pytest
import torch
from test_prediction_training import fake_step, recipe, summary
from test_training_steps import make_meter
from torch import Tensor

from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.data.bundles import DatasetBundle
from cipher_lab.training import shared_m
from cipher_lab.training.shared_m import run_shared_m
from cipher_lab.training_config import TASK_FAMILIES, SharedMRecipe, SharedMSpec, TaskSpec


def shared_recipe() -> SharedMRecipe:
    return SharedMRecipe(1, 'shared-m-four-task-v1', SharedMSpec(2, 2, 1, (1,), 2, 2),
                         recipe().training,
                         tuple(TaskSpec(t.cipher, 1, 1) for t in TASK_FAMILIES))


class ArtificialShared:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def panel(self, split: str, task: TaskSpec) -> tuple[Tensor, Tensor]:
        self.calls.append(split + '/' + task.cipher)
        return torch.zeros(2, 4, 2, 64), torch.tensor([[0.], [1.]])

    def test_bundle(self) -> DatasetBundle:
        self.calls.append('bundle')
        tasks = shared_recipe().tasks
        return DatasetBundle('shared-m-four-task-v1', tasks,
                             {t.cipher: self.panel('test', t) for t in tasks})


def test_shared_real_four_updates_closed_loop(tmp_path: Path) -> None:
    data, meter = ArtificialShared(), make_meter()
    store = RunStore.claim(tmp_path, 'shared', output_bytes=10000000, needs_models=True)
    rng = torch.get_rng_state().clone()
    start = time.monotonic()
    result = run_shared_m(shared_recipe(), data, meter, store, backend_identity='demo')
    elapsed = time.monotonic() - start
    print(f'ENGINEERING shared actual_optimizer_steps=4 elapsed_seconds={elapsed:.3f}')
    assert elapsed < 60 and result['status'] == 'SUCCEEDED'
    assert torch.equal(rng, torch.get_rng_state())
    assert meter.snapshot()['completed_optimizer_updates'] == 4
    assert meter.snapshot()['completed_forward_rows'] == 40
    saved = read_saved(tmp_path, 'shared')
    metrics = cast(dict[str, object], summary(saved)['metrics'])
    dev = cast(dict[str, object], metrics['dev'])
    task_metrics = cast(dict[str, dict[str, float]], dev['tasks'])
    assert dev['mean_bce'] == sum(t['bce'] for t in task_metrics.values()) / 4
    assert data.calls[:8] == [s + '/' + t.cipher
                              for s in ('train', 'dev') for t in shared_recipe().tasks]
    assert data.calls.count('bundle') == 1


def test_shared_tasks_schedule_and_no_test_before_loaded(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab.training import steps
    data, meter = ArtificialShared(), make_meter()
    order: list[str] = []
    def step(model: object, x: Tensor, y: Tensor, task: TaskSpec,
             optimizer: object, resources: object) -> float:
        order.append(task.cipher)
        return fake_step(model, x, y, task, optimizer, meter)
    monkeypatch.setattr(shared_m, 'train_step', step)
    def reject(*args: object, **kwargs: object) -> object:
        assert not any(c.startswith('test') for c in data.calls)
        raise ValueError('PRIVATE')
    monkeypatch.setattr(steps, 'load_checkpoint', reject)
    store = RunStore.claim(tmp_path, 'reject', output_bytes=10000000, needs_models=True)
    result = run_shared_m(shared_recipe(), data, meter, store, backend_identity='demo')
    assert result['status'] == 'FAILED'
    assert order == ['SIMECK', 'PRESENT', 'SIMON', 'SKINNY']
    assert not any(c.startswith('test') for c in data.calls)
    assert meter.snapshot()['completed_optimizer_updates'] == 4


def test_shared_partial_stop_retains_one_completed_task(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab.execution import Stopped
    meter = make_meter()
    calls = 0
    def partial(model: object, x: Tensor, y: Tensor, task: object,
                optimizer: object, resources: object) -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise Stopped('CPU_LIMIT')
        return fake_step(model, x, y, task, optimizer, meter)
    monkeypatch.setattr(shared_m, 'train_step', partial)
    store = RunStore.claim(tmp_path, 'partial', output_bytes=10000000, needs_models=True)
    data = ArtificialShared()
    result = run_shared_m(shared_recipe(), data, meter, store, backend_identity='ciphers')
    assert result['status'] == 'STOPPED' and summary(result)['reason'] == 'CPU_LIMIT'
    assert summary(result)['synthetic'] is False
    assert summary(result)['scientific_validation'] is False
    assert meter.snapshot()['completed_optimizer_updates'] == 1
    assert not any(c.startswith('test') for c in data.calls)

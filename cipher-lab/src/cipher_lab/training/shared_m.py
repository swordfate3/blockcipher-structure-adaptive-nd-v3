"""Joint model, four tasks per macro-batch; TEST is prebuilt, never selects."""
from __future__ import annotations

from typing import Protocol, cast

import torch
from torch import Tensor

from cipher_lab.artifacts import ArtifactError
from cipher_lab.data.bundles import DatasetBundle, encode_bundle
from cipher_lab.evaluation.metrics import EvaluationPanels, PublicModelSpec, evaluate_batches
from cipher_lab.execution import Meter, Stopped
from cipher_lab.training.steps import (
    IdentityChanged,
    Store,
    TrainingRecord,
    initialized_model,
    make_optimizer,
    train_step,
)
from cipher_lab.training_config import SharedMRecipe, TaskSpec


class SharedPanels(Protocol):
    def panel(self, split: str, task: TaskSpec) -> tuple[Tensor, Tensor]: ...
    def test_bundle(self) -> DatasetBundle: ...


def run_shared_m(recipe: SharedMRecipe, data: SharedPanels, meter: Meter,
                 store: Store, *, backend_identity: str) -> dict[str, object]:
    spec = PublicModelSpec(recipe.model, recipe.tasks)
    record = TrainingRecord(recipe, spec, meter, store, backend_identity)
    status, reason = 'SUCCEEDED', ''
    try:
        record.begin()
        panels: dict[str, dict[str, tuple[Tensor, Tensor]]] = {}
        for split in ('train', 'dev'):
            panels[split] = {}
            for task in recipe.tasks:
                record.guard()
                x, y = data.panel(split, task)
                record.panel(split, task.cipher, x, y)
                panels[split][task.cipher] = x, y
        model = initialized_model(spec, recipe.training.init_seed)
        optimizer = make_optimizer(model, recipe.training.learning_rate)
        shuffle = torch.Generator(device='cpu').manual_seed(recipe.training.shuffle_seed)
        for epoch in range(1, recipe.training.epochs + 1):
            orders = {t.cipher: torch.randperm(recipe.training.train, generator=shuffle)
                      for t in recipe.tasks}
            total = 0.
            for start in range(0, recipe.training.train, recipe.training.batch_size):
                for task in recipe.tasks:  # exactly one step/task per macro-batch
                    record.guard()
                    x, y = panels['train'][task.cipher]
                    indices = orders[task.cipher][start:start + recipe.training.batch_size]
                    loss = train_step(model, x[indices], y[indices], task, optimizer, meter)
                    record.counts['updates'] = cast(int, record.counts['updates']) + 1
                    total += loss * len(indices)
                    record.snapshot()
            record.guard()
            dev = evaluate_batches(model, EvaluationPanels(recipe.method, recipe.tasks,
                panels['dev'], record.baseline), spec, meter, recipe.training.batch_size)
            record.choose(model, epoch, dev, total / (recipe.training.train * len(recipe.tasks)))
        model = record.selected_model()
        panels['test'] = {}
        for task in recipe.tasks:
            record.guard()
            x, y = data.panel('test', task)
            record.panel('test', task.cipher, x, y)
            panels['test'][task.cipher] = x, y
        for split in ('train', 'dev', 'test'):
            record.guard()
            record.summary['metrics'][split] = evaluate_batches(model,
                EvaluationPanels(recipe.method, recipe.tasks, panels[split], record.baseline),
                spec, meter, recipe.training.batch_size)
            record.snapshot()
        record.guard()
        bundle = data.test_bundle()
        if (bundle.method != spec.method or bundle.task_declarations != spec.task_declarations
                or set(bundle.panels) != set(spec.names)
                or not all(torch.equal(a, b) for name in spec.names for a, b in zip(
                    bundle.panels[name], panels['test'][name], strict=True))):
            raise ValueError('TEST_BUNDLE_MISMATCH')
        record.summary['refs']['test'] = store.write_dataset('test', encode_bundle(bundle))
        record.snapshot()
    except ArtifactError:
        raise
    except IdentityChanged:
        status, reason = 'STOPPED', 'IDENTITY_CHANGED'
    except Stopped as error:
        status, reason = 'STOPPED', error.reason
    except Exception:
        status, reason = 'FAILED', 'TRAINING_FAILED'
    return record.finish(status, reason)

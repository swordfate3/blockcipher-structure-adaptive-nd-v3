"""Absolute-bit prediction timing; TEST labels remain gated until strict load."""
from __future__ import annotations

from typing import Protocol, cast

import torch
from torch import Tensor

from cipher_lab.artifacts import ArtifactError
from cipher_lab.data.bundles import DatasetBundle, PredictionTask, encode_bundle
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
from cipher_lab.training_config import PredictionRecipe


class PredictionPanels(Protocol):
    def inputs(self, split: str) -> Tensor: ...
    def labels(self, split: str) -> Tensor: ...
    def release_test(self) -> None: ...
    def test_bundle(self) -> DatasetBundle: ...


def run_prediction(recipe: PredictionRecipe, data: PredictionPanels, meter: Meter,
                   store: Store, *, backend_identity: str) -> dict[str, object]:
    task = PredictionTask('SIMON', recipe.rounds, recipe.target_bit, recipe.label_mode)
    spec = PublicModelSpec(recipe.model, (task,))
    record = TrainingRecord(recipe, spec, meter, store, backend_identity)
    status, reason = 'SUCCEEDED', ''
    try:
        record.begin()  # claim/current guard and initial identity before data access
        panels: dict[str, tuple[Tensor, Tensor]] = {}
        for split in ('train', 'dev'):
            record.guard()
            x, y = data.inputs(split), data.labels(split)
            record.panel(split, task.cipher, x, y)
            panels[split] = x, y
        model = initialized_model(spec, recipe.training.init_seed)
        optimizer = make_optimizer(model, recipe.training.learning_rate)
        shuffle = torch.Generator(device='cpu').manual_seed(recipe.training.shuffle_seed)
        x, y = panels['train']
        for epoch in range(1, recipe.training.epochs + 1):
            order = torch.randperm(len(x), generator=shuffle)
            total = 0.
            for start in range(0, len(x), recipe.training.batch_size):
                record.guard()
                indices = order[start:start + recipe.training.batch_size]
                loss = train_step(model, x[indices], y[indices], task, optimizer, meter)
                record.counts['updates'] = cast(int, record.counts['updates']) + 1
                total += loss * len(indices)
                record.snapshot()
            record.guard()
            dev = evaluate_batches(model, EvaluationPanels(recipe.method, (task,),
                {task.cipher: panels['dev']}, record.baseline), spec, meter,
                recipe.training.batch_size)
            record.choose(model, epoch, dev, total / len(x))
        model = record.selected_model()  # strict saved-SHA load before granting TEST
        record.guard()
        data.release_test()
        record.guard()
        panels['test'] = data.inputs('test'), data.labels('test')
        record.panel('test', task.cipher, *panels['test'])
        for split in ('train', 'dev', 'test'):
            record.guard()
            record.summary['metrics'][split] = evaluate_batches(model,
                EvaluationPanels(recipe.method, (task,), {task.cipher: panels[split]},
                                 record.baseline), spec, meter, recipe.training.batch_size)
            record.snapshot()
        record.guard()
        bundle = data.test_bundle()
        if (bundle.method != spec.method or bundle.task_declarations != spec.task_declarations
                or set(bundle.panels) != {task.cipher}
                or not all(torch.equal(a, b) for a, b in zip(
                    bundle.panels[task.cipher], panels['test'], strict=True))):
            raise ValueError('TEST_BUNDLE_MISMATCH')
        record.summary['refs']['test'] = store.write_dataset('test', encode_bundle(bundle))
        record.snapshot()
    except ArtifactError:
        raise  # sole storage owner may already be sealed; never retry finalization
    except IdentityChanged:
        status, reason = 'STOPPED', 'IDENTITY_CHANGED'
    except Stopped as error:
        status, reason = 'STOPPED', error.reason
    except Exception:
        status, reason = 'FAILED', 'TRAINING_FAILED'
    return record.finish(status, reason)

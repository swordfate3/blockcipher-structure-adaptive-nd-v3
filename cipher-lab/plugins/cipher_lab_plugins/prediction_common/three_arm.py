"""Narrow fixed ABC scheduling shared by legacy probe and frontier.

No recipe dispatch, cipher, storage implementation, representation construction,
selection policy or experiment-cycle approval. Arms own durable selection policy;
this loop owns exactly the pre-existing train/DEV/three-load/TEST/final order.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Protocol, cast

import torch
from torch import Tensor

from cipher_lab.artifacts import ArtifactError
from cipher_lab.execution import Meter, Stopped
from cipher_lab_plugins.feature_probe.config import FeatureProbeRecipe
from cipher_lab_plugins.feature_probe.tasks import FeatureProbeTask
from cipher_lab_plugins.frontier.config import FrontierRecipe
from cipher_lab_plugins.frontier.tasks import FrontierTask
from cipher_lab_plugins.prediction_common.bundles import DatasetBundle, encode_bundle
from cipher_lab_plugins.prediction_common.metrics import (
    EvaluationPanels,
    Model,
    PublicModelSpec,
    _tensor,
)
from cipher_lab_plugins.prediction_common.paired import paired_accuracy
from cipher_lab_plugins.prediction_common.steps import IdentityChanged, Store, TrainingRecord


class Panels(Protocol):
    def inputs(self, split: str) -> Tensor: ...
    def labels(self, split: str) -> Tensor: ...
    def reference(self, split: str) -> Tensor: ...
    def release_test(self) -> None: ...
    def test_bundle(self) -> DatasetBundle: ...


class Arm(Protocol):
    spec: PublicModelSpec
    updates: int
    evidence: dict[str, object]

    def choose(self, record: TrainingRecord, model: Model, epoch: int,
               dev: dict[str, object], loss: float) -> None: ...
    def selected_model(self, record: TrainingRecord) -> Model: ...

def _state_sha(model: Model, *, shared: bool = False) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if shared and name.startswith('layers.0.'):
            continue
        digest.update(name.encode('ascii'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _panel_identity(pairing: dict[str, object], split: str,
                    x: Tensor, y: Tensor, h: Tensor) -> None:
    _tensor(h, (1,), binary=True)
    if h.shape != y.shape:
        raise ValueError('INVALID_REFERENCE_PANEL')
    hashes: dict[str, object] = {'rows': len(x)}
    for name, value in (('x', x), ('y', y), ('h', h)):
        digest = hashlib.sha256()
        for start in range(0, len(value), 8192):
            digest.update(value[start:start+8192].to(torch.uint8).contiguous().numpy().tobytes())
        hashes[f'{name}_sha256'] = digest.hexdigest()
    cast(dict[str, object], pairing['panels'])[split] = hashes


def run_abc(recipe: FeatureProbeRecipe | FrontierRecipe, data: Panels,
            meter: Meter, store: Store, *, backend_identity: str,
            task: FeatureProbeTask | FrontierTask, arms: Mapping[str, Arm],
            initialized_arm: Callable[[PublicModelSpec, int], Model],
            make_optimizer: Callable[..., object], train_step: Callable[..., float],
            evaluate_batches: Callable[..., dict[str, object]],
            initialization_rule: str,
            record_type: type[TrainingRecord] = TrainingRecord) -> dict[str, object]:
    record = record_type(recipe, arms['A'].spec, meter, store, backend_identity)
    record.counts.update(train_presentations=0, dev_evaluations=0)
    summary = cast(dict[str, object], record.summary)
    pairing: dict[str, object] = {'initial_state_sha256': {}, 'shared_layers_sha256': {},
        'panels': {}, 'shuffle_seed': recipe.training.shuffle_seed,
        'init_seed': recipe.training.init_seed, 'initialization_rule': initialization_rule}
    budgets = {name: meter.cumulative_budget(wall_seconds=600, cpu_seconds=600) for name in arms}
    resources: dict[str, object] = {name: budget.snapshot() for name, budget in budgets.items()}
    comparisons: dict[str, object] = {'C-A': {}, 'C-B': {}}
    summary.update(arms={name: arm.evidence for name, arm in arms.items()},
                   comparisons=comparisons, pairing=pairing, arm_resources=resources)
    status, reason = 'SUCCEEDED', ''
    try:
        record.begin()
        panels: dict[str, tuple[Tensor, Tensor]] = {}
        references: dict[str, Tensor] = {}
        for split in ('train', 'dev'):
            record.guard()
            x, y, h = data.inputs(split), data.labels(split), data.reference(split)
            record.panel(split, task.cipher, x, y)
            _panel_identity(pairing, split, x, y, h)
            panels[split], references[split] = (x, y), h
        x, y = panels['train']
        residual_y = (y != references['train']).float()
        for name, arm in arms.items():
            with budgets[name]:
                record.guard()
                model = initialized_arm(arm.spec, recipe.training.init_seed)
                identities = cast(dict[str, str], pairing['initial_state_sha256'])
                shared = cast(dict[str, str], pairing['shared_layers_sha256'])
                identities[name], shared[name] = _state_sha(model), _state_sha(model, shared=True)
                if ((name == 'B' and identities['A'] != identities['B'])
                        or len(set(shared.values())) != 1):
                    raise ValueError('INITIAL_STATE_MISMATCH')
                arm.evidence.update(input_representation=arm.spec.input_representation,
                    output_mode=arm.spec.output_mode, model_spec=arm.spec.model.to_dict(),
                    parameter_count=sum(p.numel() for p in model.parameters()), arm_id=name)
                optimizer = make_optimizer(model, recipe.training.learning_rate)
                shuffle = torch.Generator(device='cpu').manual_seed(recipe.training.shuffle_seed)
                targets = y if name == 'A' else residual_y
                for epoch in range(1, recipe.training.epochs + 1):
                    order = torch.randperm(len(x), generator=shuffle)
                    total = 0.
                    for start in range(0, len(x), recipe.training.batch_size):
                        record.guard()
                        indices = order[start:start + recipe.training.batch_size]
                        before = cast(int, meter.snapshot()['completed_forward_rows'])
                        try:
                            loss = train_step(model, x[indices], targets[indices], task,
                                              optimizer, meter)
                        finally:
                            after = cast(int, meter.snapshot()['completed_forward_rows'])
                            record.counts['train_presentations'] = (
                                cast(int, record.counts['train_presentations']) + after - before)
                        # Only a normal return is an owner-completed update.
                        arm.updates += 1
                        record.counts['updates'] = cast(int, record.counts['updates']) + 1
                        total += loss * len(indices)
                        record.snapshot()
                    record.guard()
                    dev = evaluate_batches(model, EvaluationPanels(recipe.method, (task,),
                        {'SIMON': panels['dev']}, record.baseline, {'SIMON': references['dev']}),
                        arm.spec, meter, recipe.training.batch_size)
                    record.counts['dev_evaluations'] = (
                        cast(int, record.counts['dev_evaluations']) + 1)
                    arm.choose(record, model, epoch, dev, total / len(x))
                del optimizer, model
            resources[name] = budgets[name].snapshot()
        selected: dict[str, Model] = {}
        for name, arm in arms.items():
            with budgets[name]:
                selected[name] = arm.selected_model(record)
            resources[name] = budgets[name].snapshot()
        record.guard()
        data.release_test()  # ALL THREE durable selected identities passed strict load.
        record.guard()
        x_test, y_test, h_test = data.inputs('test'), data.labels('test'), data.reference('test')
        record.panel('test', 'SIMON', x_test, y_test)
        _panel_identity(pairing, 'test', x_test, y_test, h_test)
        panels['test'], references['test'] = (x_test, y_test), h_test
        for split in ('train', 'dev', 'test'):
            logits: dict[str, Tensor] = {}
            for name, arm in arms.items():
                with budgets[name]:
                    record.guard()
                    collected: dict[str, Tensor] = {}
                    metrics = evaluate_batches(selected[name], EvaluationPanels(recipe.method,
                        (task,), {'SIMON': panels[split]}, record.baseline,
                        {'SIMON': references[split]}), arm.spec, meter, recipe.training.batch_size,
                        outputs=collected)
                    cast(dict[str, object], arm.evidence['metrics'])[split] = metrics
                    logits[name] = collected['SIMON']
                    record.snapshot()
                resources[name] = budgets[name].snapshot()
            for baseline in ('A', 'B'):
                comparison = paired_accuracy(logits[baseline], logits['C'], panels[split][1])
                loss_difference = cast(dict[str, object], comparison['loss_difference'])
                loss_difference['direction'] = f'C-{baseline}'
                cast(dict[str, object], comparisons[f'C-{baseline}'])[split] = comparison
        record.guard()
        bundle = data.test_bundle()
        if (bundle.method != recipe.method or bundle.task_declarations != (task,)
                or set(bundle.panels) != {'SIMON'} or set(bundle.references) != {'SIMON'}
                or not all(torch.equal(a, b) for a, b in zip(bundle.panels['SIMON'],
                                                           panels['test'], strict=True))
                or not torch.equal(bundle.references['SIMON'], h_test)):
            raise ValueError('TEST_BUNDLE_MISMATCH')
        record.summary['refs']['test'] = store.write_dataset('test', encode_bundle(bundle))
        for name, arm in arms.items():
            record.summary['refs'][name] = store.write_payload(name, arm.evidence)
        record.summary['refs']['comparisons'] = store.write_payload('comparisons', comparisons)
        record.summary['refs']['pairing'] = store.write_payload('pairing', pairing)
        record.snapshot()
    except ArtifactError:
        raise  # The storage owner is sealed; never retry or finalize it here.
    except IdentityChanged:
        status, reason = 'STOPPED', 'IDENTITY_CHANGED'
    except Stopped as error:
        status, reason = 'STOPPED', error.reason
    except Exception:
        status, reason = 'FAILED', 'TRAINING_FAILED'
    finally:
        # Include a failing active segment; __exit__ accumulates before propagation.
        resources.update({name: budget.snapshot() for name, budget in budgets.items()})
    return record.finish(status, reason)

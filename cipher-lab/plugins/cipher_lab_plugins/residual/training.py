"""Two fixed prediction arms, one private panel, and a dual strict-load TEST gate.

Data owns references/key/labels; metrics owns absolute-logit transformation and
statistics. This owner schedules fits and commits unique durable selections.
No production cipher, private randomness or remote execution is imported here.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol, cast

import torch
from torch import Tensor

from cipher_lab.artifacts import ArtifactError
from cipher_lab.execution import Meter, Stopped
from cipher_lab_plugins.prediction_common.bundles import DatasetBundle, encode_bundle
from cipher_lab_plugins.prediction_common.checkpoints import (
    Selection,
    encode_checkpoint,
    load_checkpoint,
)
from cipher_lab_plugins.prediction_common.metrics import (
    EvaluationPanels,
    Model,
    PublicModelSpec,
    TrainBaseline,
    _tensor,
    evaluate_batches,
)
from cipher_lab_plugins.prediction_common.paired import paired_accuracy, reference_diagnostics
from cipher_lab_plugins.prediction_common.steps import (
    BestSelection,
    IdentityChanged,
    Store,
    TrainingRecord,
    initialized_model,
    make_optimizer,
    train_step,
)
from cipher_lab_plugins.residual.config import ResidualPredictionRecipe
from cipher_lab_plugins.residual.tasks import ResidualPredictionTask


class PairedPanels(Protocol):
    def inputs(self, split: str) -> Tensor: ...
    def labels(self, split: str) -> Tensor: ...
    def reference(self, split: str) -> Tensor: ...
    def release_test(self) -> None: ...
    def test_bundle(self) -> DatasetBundle: ...


@dataclass
class _Arm:
    spec: PublicModelSpec
    best: BestSelection = field(default_factory=BestSelection)
    updates: int = 0
    content: bytes | None = None
    sha: str | None = None
    selection: Selection | None = None
    evidence: dict[str, object] = field(default_factory=lambda: {
        'selected': {}, 'learning_curve': [], 'metrics': {}})

    def choose(self, record: TrainingRecord, model: Model, epoch: int,
               dev: dict[str, object], loss: float) -> None:
        curve = cast(list[dict[str, object]], self.evidence['learning_curve'])
        curve.append({'epoch': epoch, 'train_bce': loss, 'dev': dev})
        bce = cast(float, dev['mean_bce'])
        record.guard()
        # Validate with a temporary selection. Do not mutate best until the
        # immutable epoch checkpoint has durably committed in RunStore.
        proposed = BestSelection(self.best.epoch, self.best.dev_bce)
        if proposed.consider(epoch, bce):
            selection = Selection(epoch, self.updates, bce)
            content = encode_checkpoint(model, self.spec, self.spec.task_declarations,
                selection, record.baseline, record.summary['recipe_sha256'])
            name = f'{self.spec.output_mode}-e{epoch}'
            ref = record.store.write_model(name, content)
            self.best = proposed
            self.content, self.selection = content, selection
            self.sha = hashlib.sha256(content).hexdigest()
            self.evidence['selected'] = {'epoch': epoch, 'updates': self.updates,
                'selected_dev_bce': bce, 'sha256': self.sha, 'ref': ref}
        record.snapshot()
        record.guard()

    def selected_model(self, record: TrainingRecord) -> Model:
        record.guard()
        if self.content is None or self.sha is None:
            raise ValueError('NO_SELECTED_MODEL')
        loaded = load_checkpoint(self.content, self.sha)
        if (loaded.public_model_spec != self.spec or loaded.selection != self.selection
                or loaded.recipe_sha != record.summary['recipe_sha256']
                or loaded.task_declarations != self.spec.task_declarations
                or loaded.baseline != record.baseline):
            raise ValueError('CHECKPOINT_IDENTITY_MISMATCH')
        record.guard()
        return loaded.model


def _reference(h: Tensor, y: Tensor) -> None:
    _tensor(h, (1,), binary=True)
    if h.shape != y.shape:
        raise ValueError('INVALID_REFERENCE_PANEL')


def _panel_identity(pairing: dict[str, object], split: str,
                    x: Tensor, y: Tensor, h: Tensor) -> None:
    # Public panel/label evidence only, never a key or key-derived fingerprint.
    # Hash binary uint8 in bounded chunks (no full-panel byte expansion).
    hashes: dict[str, object] = {'rows': len(x)}
    for name, value in (('x', x), ('y', y), ('h', h)):
        digest = hashlib.sha256()
        for start in range(0, len(value), 8192):
            digest.update(value[start:start+8192].to(torch.uint8).contiguous().numpy().tobytes())
        hashes[f'{name}_sha256'] = digest.hexdigest()
    cast(dict[str, object], pairing['panels'])[split] = hashes


def run_paired_prediction(recipe: ResidualPredictionRecipe, data: PairedPanels,
                          meter: Meter, store: Store, *,
                          backend_identity: str) -> dict[str, object]:
    task = ResidualPredictionTask('SIMON', recipe.rounds, recipe.target_bit,
                                  recipe.label_mode, recipe.reference)
    arms = {mode: _Arm(PublicModelSpec(recipe.model, (task,), mode))
            for mode in ('direct', 'residual')}
    record = TrainingRecord(recipe, arms['direct'].spec, meter, store, backend_identity)
    # TrainingRecord owns only common evidence/guard/finalization here; the new
    # fields deliberately do not alter its old single-arm TypedDict contract.
    record.counts.update(train_presentations=0, dev_evaluations=0)
    summary = cast(dict[str, object], record.summary)
    pairing: dict[str, object] = {'initial_state_sha256': {}, 'panels': {},
                                  'shuffle_seed': recipe.training.shuffle_seed}
    summary.update(arms={name: arm.evidence for name, arm in arms.items()},
                   paired={}, diagnostics={}, pairing=pairing)
    status, reason = 'SUCCEEDED', ''
    try:
        record.begin()
        panels: dict[str, tuple[Tensor, Tensor]] = {}
        references: dict[str, Tensor] = {}
        for split in ('train', 'dev'):
            record.guard()
            x, y, h = data.inputs(split), data.labels(split), data.reference(split)
            record.panel(split, task.cipher, x, y)
            _reference(h, y)
            panels[split], references[split] = (x, y), h
            _panel_identity(pairing, split, x, y, h)
        x, y = panels['train']
        residual_y = (y != references['train']).float()
        residual_baseline = TrainBaseline(len(y), int(residual_y.sum()))
        for mode, arm in arms.items():
            record.guard()
            model = initialized_model(arm.spec, recipe.training.init_seed)
            digest = hashlib.sha256()
            for value in model.state_dict().values():
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            identities = cast(dict[str, str], pairing['initial_state_sha256'])
            identities[mode] = digest.hexdigest()
            if len(set(identities.values())) != 1:
                raise ValueError('INITIAL_STATE_MISMATCH')
            optimizer = make_optimizer(model, recipe.training.learning_rate)
            shuffle = torch.Generator(device='cpu').manual_seed(recipe.training.shuffle_seed)
            targets = y if mode == 'direct' else residual_y
            for epoch in range(1, recipe.training.epochs + 1):
                order = torch.randperm(len(x), generator=shuffle)
                total = 0.
                for start in range(0, len(x), recipe.training.batch_size):
                    record.guard()
                    indices = order[start:start + recipe.training.batch_size]
                    completed_before = cast(int, meter.snapshot()['completed_forward_rows'])
                    try:
                        loss = train_step(model, x[indices], targets[indices], task,
                                          optimizer, meter)
                    finally:
                        # This serial window contains only this TRAIN call, not
                        # DEV. Forward completion survives later backward/step
                        # failures; a refused reservation contributes zero rows.
                        completed_after = cast(int, meter.snapshot()['completed_forward_rows'])
                        record.counts['train_presentations'] = (
                            cast(int, record.counts['train_presentations'])
                            + completed_after - completed_before)
                    arm.updates += 1
                    record.counts['updates'] = cast(int, record.counts['updates']) + 1
                    total += loss * len(indices)
                    record.snapshot()
                record.guard()
                dev = evaluate_batches(model, EvaluationPanels(recipe.method, (task,),
                    {'SIMON': panels['dev']}, record.baseline, {'SIMON': references['dev']}),
                    arm.spec, meter, recipe.training.batch_size)
                record.counts['dev_evaluations'] = cast(int, record.counts['dev_evaluations']) + 1
                arm.choose(record, model, epoch, dev, total / len(x))
            # Never retain optimizer state across arms or share trainable params.
            del optimizer, model
        selected = {name: arm.selected_model(record) for name, arm in arms.items()}
        record.guard()
        data.release_test()  # BOTH durable selections strictly loaded above
        record.guard()
        x_test, y_test, h_test = data.inputs('test'), data.labels('test'), data.reference('test')
        record.panel('test', 'SIMON', x_test, y_test)
        _reference(h_test, y_test)
        panels['test'], references['test'] = (x_test, y_test), h_test
        _panel_identity(pairing, 'test', x_test, y_test, h_test)
        paired = cast(dict[str, object], summary['paired'])
        diagnostics: dict[str, object] = {}
        for split in ('train', 'dev', 'test'):
            logits: dict[str, Tensor] = {}
            for mode, arm in arms.items():
                record.guard()
                collected: dict[str, Tensor] = {}
                metrics = evaluate_batches(selected[mode], EvaluationPanels(recipe.method,
                    (task,), {'SIMON': panels[split]}, record.baseline,
                    {'SIMON': references[split]}), arm.spec, meter, recipe.training.batch_size,
                    outputs=collected)
                cast(dict[str, object], arm.evidence['metrics'])[split] = metrics
                logits[mode] = collected['SIMON']
                record.snapshot()
            paired[split] = paired_accuracy(logits['direct'], logits['residual'], panels[split][1])
            diagnostics[split] = reference_diagnostics(panels[split][1], references[split],
                                                      residual_baseline)
        summary['diagnostics'] = {'reference_constant': {
            'train_residual_rows': residual_baseline.rows,
            'train_residual_ones': residual_baseline.ones,
            'train_residual_constant': residual_baseline.majority,
            'train_residual_probability': residual_baseline.probability,
            'splits': diagnostics}}
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
        record.summary['refs']['paired'] = store.write_payload('paired', paired)
        record.summary['refs']['diagnostics'] = store.write_payload('diagnostics',
                                                                  summary['diagnostics'])
        record.snapshot()
    except ArtifactError:
        raise  # sealed storage owner must never be retried/finalized by this method
    except IdentityChanged:
        status, reason = 'STOPPED', 'IDENTITY_CHANGED'
    except Stopped as error:
        status, reason = 'STOPPED', error.reason
    except Exception:
        status, reason = 'FAILED', 'TRAINING_FAILED'
    return record.finish(status, reason)

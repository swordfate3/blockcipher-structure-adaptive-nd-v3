"""Shared optimization, strict DEV selection and public training evidence.

No data collection or generic stage executor. Method modules own split timing
and task scheduling. RunStore alone owns persistence and storage failure seals.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

import torch
from torch import Tensor

from cipher_lab.artifacts import ArtifactError
from cipher_lab.execution import Meter, Stopped
from cipher_lab.models import BitMLP, SharedM
from cipher_lab.training_config import MLPSpec, PredictionRecipe, SharedMRecipe, SharedMSpec
from cipher_lab_plugins.feature_probe.config import FeatureMLPSpec, FeatureProbeRecipe
from cipher_lab_plugins.feature_probe.model import FeatureBitMLP
from cipher_lab_plugins.frontier.config import FrontierMLPSpec, FrontierRecipe
from cipher_lab_plugins.frontier.model import FrontierBitMLP
from cipher_lab_plugins.prediction_common.bundles import TaskDeclaration
from cipher_lab_plugins.prediction_common.checkpoints import (
    Selection,
    encode_checkpoint,
    load_checkpoint,
)
from cipher_lab_plugins.prediction_common.metrics import Model, PublicModelSpec, TrainBaseline
from cipher_lab_plugins.residual.config import ResidualPredictionRecipe


class Store(Protocol):
    def current_source_identity(self) -> dict[str, str]: ...
    def source_snapshot(self) -> dict[str, str]: ...
    def write_snapshot(self, value: Mapping[str, object]) -> None: ...
    def write_payload(self, name: str, value: object) -> str: ...
    def write_model(self, name: str, content: bytes, *, replace: bool = False) -> str: ...
    def read_model(self, ref: str, expected_sha256: str, max_bytes: int) -> bytes: ...
    def write_dataset(self, name: str, content: bytes) -> str: ...
    def finalize(self, value: Mapping[str, object]) -> None: ...


@dataclass
class BestSelection:
    epoch: int | None = None
    dev_bce: float = math.inf

    def consider(self, epoch: int, dev_bce: float) -> bool:
        if (type(epoch) is not int or not 1 <= epoch <= 100000
                or type(dev_bce) not in (int, float)
                or not math.isfinite(dev_bce) or dev_bce < 0):
            raise ValueError('INVALID_DEV_SELECTION')
        if dev_bce < self.dev_bce:
            self.epoch, self.dev_bce = epoch, dev_bce
            return True
        return False


def make_optimizer(model: Model, learning_rate: float) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def initialized_model(spec: PublicModelSpec, seed: int) -> Model:
    from cipher_lab.training.steps import initialized_network
    def factory() -> Model:
        if isinstance(spec.model, MLPSpec):
            return BitMLP(spec.model)
        if isinstance(spec.model, FeatureMLPSpec):
            return FeatureBitMLP(spec.model)
        if isinstance(spec.model, FrontierMLPSpec):
            return FrontierBitMLP(spec.model)
        return SharedM(spec.model)
    return initialized_network(factory, seed)


def train_step(model: Model, inputs: Tensor, labels: Tensor,
               public_context: TaskDeclaration, optimizer: torch.optim.Optimizer,
               meter: Meter) -> float:
    from cipher_lab.training.steps import train_step as bound_step
    # This owner has only raw/feature bit models; the public wrapper owns representation.
    return bound_step(model, inputs, labels, public_context, optimizer, meter,
                      forward=lambda batch: cast(Tensor, model(batch)))


class TrainingSummary(TypedDict):
    method: str
    backend_identity: str
    synthetic: bool
    scientific_validation: bool
    recipe_sha256: str
    source: dict[str, str]
    model_spec: dict[str, object]
    task_declarations: list[dict[str, object]]
    baseline: dict[str, object]
    learning_curve: list[dict[str, object]]
    selected: dict[str, object]
    metrics: dict[str, object]
    refs: dict[str, object]
    meter: dict[str, object]
    reason: str


class IdentityChanged(Exception):
    pass


class TrainingRecord:
    """Evidence/identity/checkpoint operations, not an experiment workflow.

    Methods explicitly invoke guard/snapshot/choose/finish at their split gates.
    A successful write_model synchronously commits its binary reference. Reload
    checks SHA on these exact committed bytes; no optimizer/private RNG is saved.
    """

    def __init__(self, recipe: (PredictionRecipe | ResidualPredictionRecipe
                               | FeatureProbeRecipe | FrontierRecipe | SharedMRecipe),
                 spec: PublicModelSpec,
                 meter: Meter, store: Store, backend_identity: str) -> None:
        allowed = (('demo', 'simon')
                   if isinstance(recipe, (PredictionRecipe, ResidualPredictionRecipe,
                                          FeatureProbeRecipe, FrontierRecipe))
                   else ('demo', 'ciphers'))
        if type(backend_identity) is not str or backend_identity not in allowed:
            raise ValueError('INVALID_BACKEND_IDENTITY')
        self.recipe, self.spec, self.meter, self.store = recipe, spec, meter, store
        self.frozen = recipe.to_dict()
        self.source: dict[str, str] = {}
        public_json = json.dumps(self.frozen, ensure_ascii=False, sort_keys=True,
                                 separators=(',', ':'), allow_nan=False).encode()
        digest = hashlib.sha256(public_json).hexdigest()
        self.summary: TrainingSummary = {
            'method': recipe.method, 'backend_identity': backend_identity,
            'synthetic': backend_identity == 'demo', 'scientific_validation': False,
            'recipe_sha256': digest, 'source': self.source,
            'model_spec': spec.model.to_dict(),
            'task_declarations': [t.to_dict() for t in spec.task_declarations],
            'baseline': {}, 'learning_curve': [], 'selected': {}, 'metrics': {},
            'refs': {}, 'meter': meter.snapshot(), 'reason': '',
        }
        self.counts: dict[str, object] = {'updates': 0, 'panels': {}}
        self.baseline: dict[str, TrainBaseline] = {}
        self.best = BestSelection()
        self.content: bytes | None = None
        self.selection: Selection | None = None
        self.model_sha: str | None = None

    def begin(self) -> None:
        self.source = self.store.source_snapshot()
        self.summary['source'] = self.source
        self.guard()
        self.snapshot()

    def guard(self) -> None:
        self.meter.check()
        try:
            current = self.store.current_source_identity()
        except ArtifactError:
            raise IdentityChanged from None
        if current != self.source or self.recipe.to_dict() != self.frozen:
            raise IdentityChanged
        self.meter.check()

    def panel(self, split: str, name: str, x: Tensor, y: Tensor) -> None:
        expected = getattr(self.recipe.training, split)
        suffix = (4, 2, 64) if type(self.spec.model) is SharedMSpec else (64,)
        if (x.shape != (expected, *suffix) or y.shape != (expected, 1)
                or x.dtype != torch.float32 or y.dtype != torch.float32
                or x.device.type != 'cpu' or y.device.type != 'cpu'
                or not bool(((x == 0) | (x == 1)).all())
                or not bool(((y == 0) | (y == 1)).all())):
            raise ValueError('INVALID_TRAINING_PANEL')
        panels = cast(dict[str, object], self.counts['panels'])
        panels[f'{split}/{name}'] = expected
        if split == 'train':
            self.baseline[name] = TrainBaseline(expected, int(y.sum()))
            self.summary['baseline'][name] = {
                'rows': expected, 'ones': int(y.sum()),
                'majority': self.baseline[name].majority,
                'probability': self.baseline[name].probability}
        self.snapshot()

    def snapshot(self) -> None:
        self.summary['meter'] = self.meter.snapshot()
        self.store.write_snapshot({'recipe': self.frozen, 'counts': self.counts,
                                   'summary': dict(self.summary)})

    def choose(self, model: Model, epoch: int, dev: dict[str, object], loss: float) -> None:
        bce = cast(float, dev['mean_bce'])
        self.summary['learning_curve'].append({'epoch': epoch, 'train_bce': loss, 'dev': dev})
        self.guard()
        if self.best.consider(epoch, bce):
            selection = Selection(epoch, cast(int, self.counts['updates']), bce)
            content = encode_checkpoint(model, self.spec, self.spec.task_declarations,
                                        selection, self.baseline, self.summary['recipe_sha256'])
            ref = self.store.write_model('best', content, replace=self.content is not None)
            self.content, self.selection = content, selection
            self.model_sha = hashlib.sha256(content).hexdigest()
            self.summary['selected'] = {'epoch': epoch, 'updates': selection.updates,
                                        'selected_dev_bce': bce, 'sha256': self.model_sha}
            self.summary['refs']['best'] = ref
        self.snapshot()
        self.guard()

    def selected_model(self) -> Model:
        self.guard()
        if self.content is None or self.model_sha is None:
            raise ValueError('NO_SELECTED_MODEL')
        loaded = load_checkpoint(self.content, self.model_sha)
        if (loaded.public_model_spec != self.spec or loaded.selection != self.selection
                or loaded.recipe_sha != self.summary['recipe_sha256']
                or loaded.baseline != self.baseline):
            raise ValueError('CHECKPOINT_IDENTITY_MISMATCH')
        self.guard()
        return loaded.model

    def finish(self, status: str, reason: str = '') -> dict[str, object]:
        self.summary['reason'] = reason
        if status == 'SUCCEEDED':
            try:
                self.summary['refs']['curve'] = self.store.write_payload(
                    'learning-curve', self.summary['learning_curve'])
                self.summary['refs']['selected'] = self.store.write_payload(
                    'selected', self.summary['selected'])
                self.summary['refs']['metrics'] = self.store.write_payload(
                    'metrics', self.summary['metrics'])
                self.snapshot()
                self.guard()  # fresh clock/source decision AFTER all evidence writes
            except ArtifactError:
                raise  # sealed storage is never corrected or finalized again
            except Stopped as error:
                status, reason = 'STOPPED', error.reason
            except IdentityChanged:
                status, reason = 'STOPPED', 'IDENTITY_CHANGED'
            except Exception:
                status, reason = 'FAILED', 'TRAINING_FAILED'
        self.summary['reason'] = reason
        self.summary['meter'] = self.meter.snapshot()
        result: dict[str, object] = {'status': status, 'recipe': self.frozen,
                                    'counts': self.counts, 'summary': dict(self.summary)}
        self.store.finalize(result)
        return result

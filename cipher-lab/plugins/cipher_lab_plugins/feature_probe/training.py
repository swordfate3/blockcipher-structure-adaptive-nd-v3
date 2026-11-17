"""Fixed A/B/C public-feature probe on one panel owner and one resource meter.

Only scheduling and public evidence live here. Data owns labels/references;
metrics owns absolute-logit conversion and paired statistics; RunStore owns all
persistence. No cipher, private randomness, or CLI/application dependency.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from cipher_lab.execution import Meter
from cipher_lab_plugins.feature_probe.config import FeatureMLPSpec, FeatureProbeRecipe
from cipher_lab_plugins.feature_probe.tasks import FeatureProbeTask
from cipher_lab_plugins.prediction_common.bundles import DatasetBundle
from cipher_lab_plugins.prediction_common.checkpoints import (
    MAX_BYTES,
    Selection,
    encode_checkpoint,
    load_checkpoint,
)
from cipher_lab_plugins.prediction_common.metrics import Model, PublicModelSpec, evaluate_batches
from cipher_lab_plugins.prediction_common.steps import (
    BestSelection,
    Store,
    TrainingRecord,
    initialized_model,
    make_optimizer,
    train_step,
)
from cipher_lab_plugins.prediction_common.three_arm import run_abc


class ProbePanels(Protocol):
    def inputs(self, split: str) -> Tensor: ...
    def labels(self, split: str) -> Tensor: ...
    def reference(self, split: str) -> Tensor: ...
    def release_test(self) -> None: ...
    def test_bundle(self) -> DatasetBundle: ...


_INITIALIZATION = (
    'layer-streams-v1: SHA256(feature-probe-v1/<init_seed>/<layer_name>/<fan_in>), '
    'first 8 bytes big-endian seed; CPU Linear.reset_parameters (fan-in uniform); '
    'A/B identical streams, C distinct input stream, identical shared layer streams'
)


def initialized_arm(spec: PublicModelSpec, seed: int) -> Model:
    """Independent public streams per layer; input dimensions cannot shift later RNG."""
    spec.__post_init__()
    if spec.method != 'bit-public-feature-probe-v1':
        raise ValueError('INVALID_PROBE_SPEC')
    return initialized_layers(spec, seed)


def initialized_layers(spec: PublicModelSpec, seed: int) -> Model:
    """Public layer-stream initialization shared by the two fixed ABC methods."""
    model = initialized_model(spec, seed)
    with torch.random.fork_rng(devices=[]):
        for name, layer in model.named_modules():
            if type(layer) is nn.Linear:
                domain = f'feature-probe-v1/{seed}/{name}/{layer.in_features}'.encode('ascii')
                layer_seed = int.from_bytes(hashlib.sha256(domain).digest()[:8], 'big')
                torch.set_rng_state(torch.Generator(device='cpu').manual_seed(layer_seed).get_state())
                layer.reset_parameters()
    return model


@dataclass
class _Arm:
    spec: PublicModelSpec
    best: BestSelection = field(default_factory=BestSelection)
    updates: int = 0
    content: bytes | None = None
    ref: str | None = None
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
        proposed = BestSelection(self.best.epoch, self.best.dev_bce)
        if proposed.consider(epoch, bce):
            selection = Selection(epoch, self.updates, bce)
            content = encode_checkpoint(model, self.spec, self.spec.task_declarations,
                selection, record.baseline, record.summary['recipe_sha256'])
            # Output mode is NOT unique: both B and C predict residuals.
            ref = record.store.write_model(f'{self.spec.arm_id}-e{epoch}', content)
            self.best = proposed
            self.content, self.selection, self.ref = content, selection, ref
            self.sha = hashlib.sha256(content).hexdigest()
            self.evidence['selected'] = {'epoch': epoch, 'updates': self.updates,
                'selected_dev_bce': bce, 'sha256': self.sha, 'ref': ref}
        record.snapshot()
        record.guard()

    def selected_model(self, record: TrainingRecord) -> Model:
        record.guard()
        if self.ref is None or self.sha is None:
            raise ValueError('NO_SELECTED_MODEL')
        content = record.store.read_model(self.ref, self.sha, MAX_BYTES)
        loaded = load_checkpoint(content, self.sha)
        if (loaded.public_model_spec != self.spec or loaded.selection != self.selection
                or loaded.recipe_sha != record.summary['recipe_sha256']
                or loaded.task_declarations != self.spec.task_declarations
                or loaded.baseline != record.baseline):
            raise ValueError('CHECKPOINT_IDENTITY_MISMATCH')
        record.guard()
        return loaded.model


def run_feature_probe(recipe: FeatureProbeRecipe, data: ProbePanels,
                      meter: Meter, store: Store, *, backend_identity: str) -> dict[str, object]:
    task = FeatureProbeTask('SIMON', recipe.rounds, recipe.target_bit,
                            recipe.label_mode, recipe.reference, recipe.representation)
    arms = {name: _Arm(PublicModelSpec(
        FeatureMLPSpec(recipe.model.hidden_width, recipe.representation)
        if name == 'C' else recipe.model, (task,),
        'direct' if name == 'A' else 'residual', name,
        recipe.representation if name == 'C' else 'raw64-msb0-v1')) for name in 'ABC'}
    return run_abc(recipe, data, meter, store, backend_identity=backend_identity,
        task=task, arms=arms, initialized_arm=initialized_arm, make_optimizer=make_optimizer,
        train_step=train_step, evaluate_batches=evaluate_batches,
        initialization_rule=_INITIALIZATION)

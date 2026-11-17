"""Frontier ABC binding and three durable best slots; no cipher or cycle policy."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import cast

from cipher_lab.execution import Meter
from cipher_lab_plugins.feature_probe.training import _INITIALIZATION
from cipher_lab_plugins.frontier.checkpoint import encode_frontier_checkpoint
from cipher_lab_plugins.frontier.config import FrontierMLPSpec, FrontierRecipe
from cipher_lab_plugins.frontier.simon_frontier import PublicFeaturePlan, decode_plan, encode_plan
from cipher_lab_plugins.frontier.tasks import FrontierTask
from cipher_lab_plugins.prediction_common.checkpoints import MAX_BYTES, Selection, load_checkpoint
from cipher_lab_plugins.prediction_common.metrics import Model, PublicModelSpec, evaluate_batches
from cipher_lab_plugins.prediction_common.steps import (
    BestSelection,
    IdentityChanged,
    Store,
    TrainingRecord,
    make_optimizer,
    train_step,
)
from cipher_lab_plugins.prediction_common.three_arm import Panels, run_abc


def initialized_arm(spec: PublicModelSpec, seed: int) -> Model:
    """Same public per-layer stream domain as the legacy three-arm experiment."""
    from cipher_lab_plugins.feature_probe.training import initialized_layers
    spec.__post_init__()
    if spec.method != 'bit-public-feature-frontier-v1':
        raise ValueError('INVALID_FRONTIER_SPEC')
    return initialized_layers(spec, seed)


def output_bound(recipe: FrontierRecipe, plan: PublicFeaturePlan) -> dict[str, int]:
    """Conservative binary+metadata+temporary peak; evaluated before panels/updates.

    Four model footprints = 3 best + largest temporary replacement. Every copy
    includes the full plan even for raw A/B. 1MiB allowance per archive covers
    bounded pickle/ZIP. 16MiB covers saved metadata, journal, curves and reports;
    model source remains independently subject to RunStore's real peak budget.
    """
    recipe.__post_init__()
    validated = decode_plan(encode_plan(plan))
    if validated.identity != recipe.identity:
        raise ValueError('FRONTIER_PLAN_MISMATCH')
    # The metadata reserve is for this bounded experiment, never an arbitrary
    # 100000-epoch recipe. Tiny artificial fixtures remain inside these ceilings.
    if (recipe.training.epochs > 32 or recipe.training.train > 16384
            or recipe.training.dev > 4096 or recipe.training.test > 16384):
        raise ValueError('ENGINEERING_LIMIT_OUTPUT')
    width = recipe.model.hidden_width
    def parameters(d: int) -> int:
        return width*d + width*width + 3*width + 1
    weights = [4*parameters(d) for d in (64, 64, plan.dimension)]
    plan_bytes = len(encode_plan(plan))
    checkpoints = sum(weights) + 3*(plan_bytes + 1024*1024)
    temporary = max(weights) + plan_bytes + 1024*1024
    bundle = 66*recipe.training.test + 4*4096 + 8192
    metadata = 16*1024*1024
    total = checkpoints + temporary + bundle + metadata
    if total > 128*1024*1024:
        raise ValueError('ENGINEERING_LIMIT_OUTPUT')
    return {'three_best_bytes': checkpoints, 'maximum_temporary_bytes': temporary,
            'dataset_bytes': bundle, 'metadata_bytes': metadata, 'peak_bytes': total}


class _FrontierRecord(TrainingRecord):
    def guard(self) -> None:
        super().guard()
        try:
            self.recipe.__post_init__()
            self.spec.__post_init__()
            # A/B compact task identity still binds the same plan as C.
            for spec in self.frontier_specs:
                spec.__post_init__()
        except (ValueError, TypeError, AttributeError):
            raise IdentityChanged from None

    frontier_specs: tuple[PublicModelSpec, ...] = ()


@dataclass
class _Arm:
    spec: PublicModelSpec
    plan: PublicFeaturePlan
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
        cast(list[dict[str, object]], self.evidence['learning_curve']).append(
            {'epoch': epoch, 'train_bce': loss, 'dev': dev})
        bce = cast(float, dev['mean_bce'])
        record.guard()
        proposed = BestSelection(self.best.epoch, self.best.dev_bce)
        if proposed.consider(epoch, bce):
            selected = Selection(epoch, self.updates, bce)
            content = encode_frontier_checkpoint(model, self.spec, self.spec.task_declarations,
                selected, record.baseline, record.summary['recipe_sha256'], plan=self.plan)
            ref = record.store.write_model(f'{self.spec.arm_id}-best', content,
                                           replace=self.ref is not None)
            # A failed atomic write must not commit best or selected metadata.
            self.best, self.selection, self.ref = proposed, selected, ref
            self.content, self.sha = content, hashlib.sha256(content).hexdigest()
            self.evidence['selected'] = {'epoch': epoch, 'updates': self.updates,
                'selected_dev_bce': bce, 'sha256': self.sha, 'ref': ref}
        record.snapshot()
        record.guard()

    def selected_model(self, record: TrainingRecord) -> Model:
        record.guard()
        if self.ref is None or self.sha is None:
            raise ValueError('NO_SELECTED_MODEL')
        loaded = load_checkpoint(record.store.read_model(self.ref, self.sha, MAX_BYTES), self.sha)
        if (loaded.public_model_spec != self.spec or loaded.selection != self.selection
                or loaded.recipe_sha != record.summary['recipe_sha256']
                or loaded.task_declarations != self.spec.task_declarations
                or loaded.baseline != record.baseline):
            raise ValueError('CHECKPOINT_IDENTITY_MISMATCH')
        record.guard()
        return loaded.model


def run_frontier(recipe: FrontierRecipe, data: Panels, meter: Meter, store: Store, *,
                 backend_identity: str, plan: PublicFeaturePlan) -> dict[str, object]:
    if type(recipe) is not FrontierRecipe or type(plan) is not PublicFeaturePlan:
        raise ValueError('INVALID_FRONTIER_INPUT')
    recipe.__post_init__()
    validated = decode_plan(encode_plan(plan))
    if validated.identity != recipe.identity:
        raise ValueError('FRONTIER_PLAN_MISMATCH')
    bound = output_bound(recipe, validated)
    task = FrontierTask('SIMON', recipe.rounds, recipe.target_bit,
                        recipe.label_mode, recipe.reference, recipe.identity)
    arms = {name: _Arm(PublicModelSpec(
        FrontierMLPSpec(recipe.model.hidden_width, recipe.identity, plan)
        if name == 'C' else recipe.model, (task,),
        'direct' if name == 'A' else 'residual', name,
        recipe.representation if name == 'C' else 'raw64-msb0-v1'), plan) for name in 'ABC'}

    class Record(_FrontierRecord):
        frontier_specs = tuple(arm.spec for arm in arms.values())

        def begin(self) -> None:
            cast(dict[str, object], self.summary).update(
                plan_identity=recipe.identity.to_dict(), output_bound=bound)
            super().begin()

    return run_abc(recipe, data, meter, store, backend_identity=backend_identity,
        task=task, arms=arms, initialized_arm=initialized_arm, make_optimizer=make_optimizer,
        train_step=train_step, evaluate_batches=evaluate_batches,
        initialization_rule=_INITIALIZATION, record_type=Record)

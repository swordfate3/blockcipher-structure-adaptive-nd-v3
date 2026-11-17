"""CPU binary metrics and declaration-bound, metered evaluation (no collection)."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F

from cipher_lab.data.bundles import DatasetBundle, PredictionTask, TaskDeclaration
from cipher_lab.execution import Meter
from cipher_lab.models import BitMLP, SharedM
from cipher_lab.representations.tasks import public_features
from cipher_lab.training_config import TASK_FAMILIES, MLPSpec, SharedMSpec, TaskSpec

Model = torch.nn.Module


@dataclass(frozen=True, slots=True)
class TrainBaseline:
    """TRAIN counts only. Ties predict zero; probability uses Jeffreys smoothing."""

    rows: int
    ones: int

    def __post_init__(self) -> None:
        if (type(self.rows) is not int or not 1 <= self.rows <= 2000000
                or type(self.ones) is not int or not 0 <= self.ones <= self.rows):
            raise ValueError('invalid TRAIN baseline')

    @property
    def majority(self) -> int:
        return int(2 * self.ones > self.rows)

    @property
    def probability(self) -> float:
        return (self.ones + .5) / (self.rows + 1)


@dataclass(frozen=True, slots=True)
class PublicModelSpec:
    """Full forward identity; task declarations are the bundle's public types."""

    model: MLPSpec | SharedMSpec
    task_declarations: tuple[TaskDeclaration, ...]
    output_mode: str = "direct"
    arm_id: str | None = None
    input_representation: str | None = None

    def __post_init__(self) -> None:
        tasks = self.task_declarations
        if type(tasks) is not tuple:
            raise ValueError('invalid public model identity')
        if self.arm_id is not None or self.input_representation is not None:
            raise ValueError('unexpected arm identity')
        if type(self.model) is MLPSpec:
            if len(tasks) != 1 or type(tasks[0]) is not PredictionTask:
                raise ValueError('invalid prediction identity')
        elif type(self.model) is SharedMSpec:
            if (len(tasks) != 4 or any(type(t) is not TaskSpec for t in tasks)
                    or tuple(t.cipher for t in tasks) != tuple(f.cipher for f in TASK_FAMILIES)):
                raise ValueError('invalid shared-M identity')
        else:
            raise ValueError('unsupported public model specification')
        if self.output_mode != 'direct':
            raise ValueError('invalid output mode')
        # Revalidate dataclasses, including instances altered with object.__setattr__.
        self.model.__post_init__()
        for task in tasks:
            task.__post_init__()

    @property
    def method(self) -> str:
        return 'bit-prediction-v1' if type(self.model) is MLPSpec else 'shared-m-four-task-v1'

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(t.cipher for t in self.task_declarations)


def _tensor(value: Tensor, suffix: tuple[int, ...], *, binary: bool = False) -> None:
    if (type(value) is not Tensor or value.device.type != 'cpu'
            or value.dtype != torch.float32 or value.layout != torch.strided
            or value.ndim != len(suffix) + 1 or not 1 <= value.shape[0] <= 2000000
            or tuple(value.shape[1:]) != suffix or not bool(torch.isfinite(value).all())
            or (binary and not bool(((value == 0) | (value == 1)).all()))):
        raise ValueError('invalid evaluation tensor')


def binary_metrics(logits: Tensor, labels: Tensor, *, majority: int,
                   probability: float) -> dict[str, float | int]:
    """ACC uses logit > 0, not >= 0; BCE is stable float64, averaged per row."""
    _tensor(logits, (1,))
    _tensor(labels, (1,), binary=True)
    if (logits.shape != labels.shape or type(majority) is not int or majority not in (0, 1)
            or type(probability) not in (float, int) or not 0 < probability < 1
            or not math.isfinite(probability)):
        raise ValueError('invalid binary metrics')
    with torch.no_grad():
        y = labels.double()
        return {'rows': labels.shape[0],
                'acc': float(((logits > 0) == labels).double().mean()),
                'bce': float(F.binary_cross_entropy_with_logits(logits.double(), y)),
                'baseline_acc': float((labels == majority).double().mean()),
                'baseline_bce': float((-y * math.log(probability)
                                       - (1-y) * math.log1p(-probability)).mean())}


@dataclass(frozen=True, slots=True)
class EvaluationPanels:
    """Actual panel identity and TRAIN-only baselines; never infer either from y."""

    method: str
    task_declarations: tuple[TaskDeclaration, ...]
    panels: Mapping[str, tuple[Tensor, Tensor]]
    baseline: Mapping[str, TrainBaseline]
    references: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'panels', MappingProxyType(dict(self.panels)))
        object.__setattr__(self, 'baseline', MappingProxyType(dict(self.baseline)))
        object.__setattr__(self, 'references', MappingProxyType(dict(self.references)))

    @classmethod
    def from_bundle(cls, bundle: DatasetBundle,
                    baseline: Mapping[str, TrainBaseline]) -> EvaluationPanels:
        return cls(bundle.method, bundle.task_declarations, bundle.panels, baseline,
                   bundle.references)


def validate_baseline(spec: PublicModelSpec, baseline: Mapping[str, TrainBaseline]) -> None:
    if set(baseline) != set(spec.names):
        raise ValueError('invalid TRAIN baseline identity')
    for value in baseline.values():
        if type(value) is not TrainBaseline:
            raise ValueError('invalid TRAIN baseline')
        value.__post_init__()
        if type(spec.model) is SharedMSpec and (value.rows % 2 or value.ones * 2 != value.rows
                                               or value.rows > 65536):
            raise ValueError('shared-M TRAIN baseline must be balanced')


def validate_model(model: Model, spec: PublicModelSpec) -> None:
    expected = BitMLP if type(spec.model) is MLPSpec else SharedM
    if type(model) is not expected or getattr(model, "spec", None) != spec.model:
        raise ValueError('model does not match public specification')
    for value in model.state_dict().values():
        if (value.device.type != 'cpu' or value.layout != torch.strided
                or value.dtype not in (torch.float32, torch.int64)
                or not bool(torch.isfinite(value).all())):
            raise ValueError('invalid CPU model')


def evaluate_batches(model: Model, panels: EvaluationPanels, model_spec: PublicModelSpec,
                     meter: Meter, batch_size: int, *,
                     outputs: dict[str, Tensor] | None = None) -> dict[str, object]:
    """Validate context before reserving; each valid batch completes exactly N rows.

    Return {'rows': total, 'tasks': {cipher: metrics}, 'mean_bce': equal-task mean}.
    The mean is deliberately not row-weighted (shared-M DEV selection contract).
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 2000000:
        raise ValueError('invalid evaluation batch size')
    model_spec.__post_init__()
    if (panels.method != model_spec.method
            or panels.task_declarations != model_spec.task_declarations
            or set(panels.panels) != set(model_spec.names)):
        raise ValueError('evaluation identity mismatch')
    if panels.references:
        raise ValueError('unexpected references')
    validate_baseline(model_spec, panels.baseline)
    validate_model(model, model_spec)
    suffix = (4, 2, 64) if type(model_spec.model) is SharedMSpec else (64,)
    for x, y in panels.panels.values():
        _tensor(x, suffix, binary=True)
        _tensor(y, (1,), binary=True)
        if x.shape[0] != y.shape[0]:
            raise ValueError('evaluation row mismatch')
        if type(model_spec.model) is SharedMSpec and (x.shape[0] % 2 or x.shape[0] > 65536):
            raise ValueError('invalid shared-M rows')
    tasks = {task.cipher: task for task in model_spec.task_declarations}
    def forward(name: str, batch: Tensor) -> Tensor:
        if type(model_spec.model) is SharedMSpec:
            public = torch.tensor(public_features(name), dtype=torch.float32)
            return cast(Tensor, model(batch, public, rounds=tasks[name].rounds))
        return cast(Tensor, model(batch))
    return evaluate_bound_panels(model, model_spec.names, panels.panels, panels.baseline,
        meter, batch_size, forward=forward, outputs=outputs)


def evaluate_bound_panels(model: Model, names: tuple[str, ...],
        panels: Mapping[str, tuple[Tensor, Tensor]], baseline: Mapping[str, TrainBaseline],
        meter: Meter, batch_size: int, *, forward: Callable[[str, Tensor], Tensor],
        absolute: Callable[[str, Tensor], Tensor] | None = None,
        outputs: dict[str, Tensor] | None = None) -> dict[str, object]:
    """Metered forward/metrics only; strict typed context is validated by caller.

    No experiment/spec dispatch. Preserve modes, batch reservation order, concatenation,
    caller absolute conversion, and equal-task mean exactly as the original evaluator.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 2000000:
        raise ValueError('invalid evaluation batch size')
    modes = [(module, module.training) for module in model.modules()]
    results: dict[str, dict[str, float | int]] = {}
    try:
        model.eval()
        with torch.no_grad():
            for name in names:
                x, y = panels[name]
                batches: list[Tensor] = []
                for start in range(0, len(x), batch_size):
                    batch = x[start:start + batch_size]
                    meter.reserve('forward_rows', len(batch))
                    logits = forward(name, batch)
                    _tensor(logits, (1,))
                    if len(logits) != len(batch):
                        raise ValueError('invalid forward row count')
                    meter.complete('forward_rows', len(batch))
                    batches.append(logits)
                train_baseline = baseline[name]
                absolute_logits = torch.cat(batches)
                if absolute is not None:
                    absolute_logits = absolute(name, absolute_logits)
                if outputs is not None:
                    outputs[name] = absolute_logits.detach().clone()
                results[name] = binary_metrics(absolute_logits, y,
                    majority=train_baseline.majority, probability=train_baseline.probability)
    finally:
        for module, training in modes:
            module.training = training
    return {'rows': sum(int(r['rows']) for r in results.values()), 'tasks': results,
            'mean_bce': sum(float(r['bce']) for r in results.values()) / len(results)}

"""CPU binary metrics and declaration-bound, metered evaluation (no collection)."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import torch
from torch import Tensor

from cipher_lab.data.bundles import PredictionTask
from cipher_lab.evaluation.metrics import TrainBaseline as TrainBaseline
from cipher_lab.evaluation.metrics import _tensor as _tensor
from cipher_lab.evaluation.metrics import binary_metrics as binary_metrics
from cipher_lab.execution import Meter
from cipher_lab.models import BitMLP, SharedM
from cipher_lab.training_config import TASK_FAMILIES, MLPSpec, SharedMSpec, TaskSpec
from cipher_lab_plugins.feature_probe.config import FeatureMLPSpec
from cipher_lab_plugins.feature_probe.model import FeatureBitMLP
from cipher_lab_plugins.feature_probe.tasks import FeatureProbeTask
from cipher_lab_plugins.frontier.config import FrontierMLPSpec
from cipher_lab_plugins.frontier.model import FrontierBitMLP
from cipher_lab_plugins.frontier.tasks import FrontierTask
from cipher_lab_plugins.prediction_common.bundles import DatasetBundle, TaskDeclaration
from cipher_lab_plugins.residual.tasks import ResidualPredictionTask

Model = torch.nn.Module


@dataclass(frozen=True, slots=True)
class PublicModelSpec:
    """Full forward identity; task declarations are the bundle's public types."""

    model: MLPSpec | FeatureMLPSpec | FrontierMLPSpec | SharedMSpec
    task_declarations: tuple[TaskDeclaration, ...]
    output_mode: str = "direct"
    arm_id: str | None = None
    input_representation: str | None = None

    def __post_init__(self) -> None:
        tasks = self.task_declarations
        if type(tasks) is not tuple:
            raise ValueError('invalid public model identity')
        probe = len(tasks) == 1 and type(tasks[0]) is FeatureProbeTask
        frontier = len(tasks) == 1 and type(tasks[0]) is FrontierTask
        if frontier:
            frontier_task = cast(FrontierTask, tasks[0])
            frontier_model = FrontierMLPSpec if self.arm_id == 'C' else MLPSpec
            representation = (frontier_task.identity.representation if self.arm_id == 'C'
                              else 'raw64-msb0-v1')
            if (self.arm_id not in ('A', 'B', 'C') or type(self.model) is not frontier_model
                    or self.input_representation != representation
                    or self.output_mode != ('direct' if self.arm_id == 'A' else 'residual')
                    or (type(self.model) is FrontierMLPSpec
                        and self.model.identity != frontier_task.identity)):
                raise ValueError('invalid frontier arm binding')
        elif probe:
            expected_model = FeatureMLPSpec if self.arm_id == 'C' else MLPSpec
            representation = ('simon-r3-residual13-v1' if self.arm_id == 'C'
                              else 'raw64-msb0-v1')
            if (self.arm_id not in ('A', 'B', 'C') or type(self.model) is not expected_model
                    or self.input_representation != representation
                    or self.output_mode != ('direct' if self.arm_id == 'A' else 'residual')):
                raise ValueError('invalid feature probe arm binding')
        elif self.arm_id is not None or self.input_representation is not None:
            raise ValueError('unexpected arm identity')
        elif type(self.model) is MLPSpec:
            if len(tasks) != 1 or type(tasks[0]) not in (PredictionTask, ResidualPredictionTask):
                raise ValueError('invalid prediction identity')
        elif type(self.model) is SharedMSpec:
            if (len(tasks) != 4 or any(type(t) is not TaskSpec for t in tasks)
                    or tuple(t.cipher for t in tasks) != tuple(f.cipher for f in TASK_FAMILIES)):
                raise ValueError('invalid shared-M identity')
        else:
            raise ValueError('unsupported public model specification')
        residual = (frontier or probe
                    or (len(tasks) == 1 and type(tasks[0]) is ResidualPredictionTask))
        if (self.output_mode not in ("direct", "residual")
                or (not residual and self.output_mode != "direct")):
            raise ValueError("invalid output mode")
        # Revalidate dataclasses, including instances altered with object.__setattr__.
        self.model.__post_init__()
        for task in tasks:
            task.__post_init__()

    @property
    def method(self) -> str:
        if type(self.task_declarations[0]) is FrontierTask:
            return 'bit-public-feature-frontier-v1'
        if type(self.task_declarations[0]) is FeatureProbeTask:
            return 'bit-public-feature-probe-v1'
        if type(self.task_declarations[0]) is ResidualPredictionTask:
            return 'bit-residual-paired-v1'
        return 'bit-prediction-v1' if type(self.model) is MLPSpec else 'shared-m-four-task-v1'

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(t.cipher for t in self.task_declarations)


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
    expected = (BitMLP if type(spec.model) is MLPSpec else
                FeatureBitMLP if type(spec.model) is FeatureMLPSpec else
                FrontierBitMLP if type(spec.model) is FrontierMLPSpec else SharedM)
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
    residual_method = model_spec.method in ('bit-residual-paired-v1', 'bit-public-feature-probe-v1',
                                            'bit-public-feature-frontier-v1')
    if set(panels.references) != (set(model_spec.names) if residual_method else set()):
        raise ValueError('reference identity mismatch')
    for name, h in panels.references.items():
        _tensor(h, (1,), binary=True)
        if h.shape != panels.panels[name][1].shape:
            raise ValueError('reference row mismatch')
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
    from cipher_lab.evaluation.metrics import evaluate_bound_panels
    from cipher_lab_plugins.prediction_common.paired import absolute_logits

    def forward(name: str, batch: Tensor) -> Tensor:
        return cast(Tensor, model(batch))

    def absolute(name: str, logits: Tensor) -> Tensor:
        return absolute_logits(logits, panels.references[name])

    return evaluate_bound_panels(model, model_spec.names, panels.panels, panels.baseline,
        meter, batch_size, forward=forward,
        absolute=absolute if model_spec.output_mode == 'residual' else None, outputs=outputs)

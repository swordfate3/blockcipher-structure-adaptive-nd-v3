"""Stdlib-only immutable recipe, strict parser and static accounting.

Parsing is not execution authorization. Device binding belongs to the application.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .four_single.config import FourSingleRecipe
    from .layout_probe.config import LayoutRecipe


@dataclass(frozen=True)
class Task:
    cipher: str
    rounds: int
    difference: int

    @property
    def key_bytes(self) -> int:
        return {'SIMECK': 16, 'PRESENT': 10, 'SIMON': 16, 'SKINNY': 8}[self.cipher]


TASKS = (Task('SIMECK', 16, 0x40), Task('PRESENT', 7, 0xD00000),
         Task('SIMON', 13, 0x40), Task('SKINNY', 8, 0x2000))


@dataclass(frozen=True)
class Training:
    epochs: int = 24
    batches_per_epoch: int = 1024
    batch_groups: int = 512
    dev_groups: int = 65536
    test_groups: int = 262144
    eval_batch_groups: int = 512
    pairs_per_group: int = 4
    gradient_accumulation: int = 1
    retries: int = 0


@dataclass(frozen=True)
class Optimizer:
    name: str = 'Adam'
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False
    lr_max: float = 0.001
    lr_min: float = 0.0001
    schedule: str = 'cycle-cosine'
    clip_norm: float = 1.0
    loss: str = 'mean-BCEWithLogits'
    label_smoothing: float = 0.0
    kernel_l2: float = 0.0
    routing_penalty: float = 0.0


@dataclass(frozen=True)
class Seeds:
    init: int = 26111741
    sample: int = 26111742
    shuffle: int = 26111743
    bootstrap: int = 26111744


@dataclass(frozen=True)
class Statistics:
    replicates: int = 20000
    confidence: float = 0.9875
    family_alpha: float = 0.05
    comparisons: int = 4
    threshold: float = 0.0
    quantile: str = 'linear'
    bootstrap: str = 'fixed-class-quota-paired-group-multinomial'


@dataclass(frozen=True)
class Model:
    stem_width: int = 96
    channels: int = 96
    norm_groups: int = 8
    suffix_kernels: tuple[int, ...] = (3, 5, 7, 9, 11)
    head_hidden: int = 64
    router_hidden: int = 16
    top_k: int = 2
    dedicated: tuple[str, ...] = ('feistel-se-simeck64', 'present-xor-undilated',
                                 'feistel-se-simon64', 'skinny-backstep-xor-resnet')


@dataclass(frozen=True)
class Selection:
    shared: str = 'macro-dev-bce'
    dedicated: str = 'task-dev-bce'
    interval: str = 'epoch-end'
    ties: str = 'earliest'
    early_stopping: bool = False
    test: str = 'after-five-persisted-best-reloads'


@dataclass(frozen=True)
class Resources:
    precision: str = 'float32'
    device_kind: str = 'cuda'
    amp: bool = False
    tf32: bool = False
    cudnn_benchmark: bool = False
    gpu_bytes: int = 20 * 1024**3
    rss_bytes: int = 16 * 1024**3
    artifact_bytes: int = 1024**3
    wall_seconds: int = 24 * 3600
    cpu_seconds: int = 96 * 3600
    intra_threads: int = 4
    inter_threads: int = 1


@dataclass(frozen=True)
class Recipe:
    method: str = 'innovation1-shared-vs-dedicated-v1'
    namespace: str = 'innovation1-fresh-v1-20261117'
    tasks: tuple[Task, ...] = TASKS
    training: Training = Training()
    optimizer: Optimizer = Optimizer()
    seeds: Seeds = Seeds()
    statistics: Statistics = Statistics()
    model: Model = Model()
    selection: Selection = Selection()
    resources: Resources = Resources()

    def to_dict(self) -> dict[str, Any]:
        # JSON canonical representation: tuples become arrays, no opaque objects.
        return dict(json.loads(json.dumps(asdict(self), allow_nan=False)))


def _decode(raw: Any, template: Any, path: str) -> Any:
    if is_dataclass(template):
        if type(raw) is not dict or set(raw) != {f.name for f in fields(template)}:
            raise ValueError(f'{path}: exact fields required')
        constructor: Any = type(template)
        return constructor(**{f.name: _decode(raw[f.name], getattr(template, f.name),
                                               f'{path}.{f.name}') for f in fields(template)})
    if isinstance(template, tuple):
        if type(raw) is not list or len(raw) != len(template):
            raise ValueError(f'{path}: exact array length required')
        return tuple(_decode(x, t, path) for x, t in zip(raw, template, strict=True))
    if isinstance(template, float):
        if type(raw) not in (int, float) or not math.isfinite(raw):
            raise ValueError(f'{path}: finite number required')
        return float(raw)
    if type(raw) is not type(template):
        raise ValueError(f'{path}: wrong type')
    return raw


def validate(recipe: Recipe) -> None:
    """Validate combinations also for manually constructed small artificial recipes."""
    # Round-trip typing validation protects dataclass constructors/replace as well.
    _decode(recipe.to_dict(), Recipe(), 'recipe')
    if (recipe.method != Recipe().method or recipe.namespace != Recipe().namespace
            or recipe.tasks != TASKS or recipe.model != Model()
            or recipe.selection != Selection() or recipe.resources != Resources()):
        raise ValueError('fixed experiment identity/model/selection/resources required')
    tr, opt, stats = recipe.training, recipe.optimizer, recipe.statistics
    if any(type(x) is not int or x <= 0 for x in (
        tr.epochs, tr.batches_per_epoch, tr.batch_groups, tr.dev_groups,
        tr.test_groups, tr.eval_batch_groups,
    )) or tr.epochs * tr.batches_per_epoch < 2:
        raise ValueError('positive training sizes and at least two cycles required')
    if (any(x % 2 for x in (tr.batch_groups, tr.dev_groups, tr.test_groups,
                           tr.eval_batch_groups))
            or tr.dev_groups % tr.eval_batch_groups or tr.test_groups % tr.eval_batch_groups
            or tr.pairs_per_group != 4 or tr.gradient_accumulation != 1 or tr.retries != 0):
        raise ValueError('even fixed-quota groups, whole eval batches, four pairs required')
    if (opt.lr_min <= 0 or opt.lr_max < opt.lr_min
            or opt != Optimizer(lr_min=opt.lr_min, lr_max=opt.lr_max)):
        raise ValueError('fixed Adam/loss/regularization and ordered positive LR required')
    if (type(stats.replicates) is not int or stats.replicates <= 0
            or stats != Statistics(replicates=stats.replicates)):
        raise ValueError('fixed four-comparison 98.75% bootstrap protocol required')
    if any(type(x) is not int or not 0 <= x < 2**63 for x in asdict(recipe.seeds).values()):
        raise ValueError('public seeds must be nonnegative 63-bit integers')


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate config field')
        result[key] = value
    return result


MAX_CONFIG_BYTES = 16384
RUN_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', re.ASCII)
GPU_UUID = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
_RETIRED_IDS = frozenset({
    'innovation1-skinny-single-train-20261119-01',
    'innovation1-skinny-layout-train-20261119-01',
    'innovation1-skinny-layout-early-stop-20261120-01',
})
_WINDOWS_RESERVED = re.compile(r'CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9]', re.IGNORECASE)


def validate_run_id(run_id: str) -> None:
    if (not RUN_ID.fullmatch(run_id) or run_id.casefold() in _RETIRED_IDS
            or _WINDOWS_RESERVED.fullmatch(run_id)):
        raise ValueError('INVALID_RUN_ID')


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def require_official(recipe: Recipe) -> None:
    expected = Recipe()
    if (type(recipe) is not Recipe or recipe != expected
            or canonical(recipe.to_dict()) != canonical(expected.to_dict())):
        raise ValueError('CONFIG_EXACT')


def load_config(path: str | Path) -> Recipe | LayoutRecipe | FourSingleRecipe:
    from .four_single.config import FOUR_SINGLE, FOUR_SINGLE_50
    from .layout_probe.config import EARLY_STOP, FIXED_BCE, FIXED_MSE
    from .serial_bc.config import INDEPENDENT_RECIPES, SERIAL_B, SERIAL_C

    # Reject pipes/devices; the bounded read is for regular config files only.
    path = Path(path)
    if not path.is_file():
        raise ValueError('CONFIG_FILE')
    with path.open('rb') as stream:
        content = stream.read(MAX_CONFIG_BYTES + 1)
    if len(content) > MAX_CONFIG_BYTES:
        raise ValueError('CONFIG_SIZE')
    try:
        raw = json.loads(content.decode('utf-8'), object_pairs_hook=_unique)
        encoded = canonical(raw)  # strict JSON types as well as values
        exact: tuple[Recipe | LayoutRecipe | FourSingleRecipe, ...] = (
            Recipe(), EARLY_STOP, FIXED_MSE, FIXED_BCE, FOUR_SINGLE, FOUR_SINGLE_50,
            SERIAL_B, SERIAL_C, *INDEPENDENT_RECIPES)
        for recipe in exact:
            if encoded == canonical(recipe.to_dict()):
                return recipe
    except (ValueError, RecursionError):
        pass
    raise ValueError('CONFIG_EXACT')


def learning_rate(recipe: Recipe, cycle: int) -> float:
    cycles = recipe.training.epochs * recipe.training.batches_per_epoch
    if type(cycle) is not int or not 0 <= cycle < cycles or cycles < 2:
        raise ValueError('cycle outside schedule')
    opt = recipe.optimizer
    return opt.lr_min + 0.5 * (opt.lr_max - opt.lr_min) * (
        1 + math.cos(math.pi * cycle / (cycles - 1)))


def estimate(recipe: Recipe) -> dict[str, int]:
    validate(recipe)
    tr = recipe.training
    u, b, v, t, e = (tr.epochs * tr.batches_per_epoch, tr.batch_groups,
                     tr.dev_groups, tr.test_groups, tr.epochs)
    return dict(cycles=u, fits=5, retries=0, optimizer_updates=8*u,
                shared_updates=4*u, dedicated_updates_each=u,
                train_groups_each=u*b, train_groups=4*u*b, dev_groups=4*v, test_groups=4*t,
                generated_groups=4*(u*b+v+t), real_blocks=32*(u*b+v+t),
                train_model_groups=8*u*b, dev_model_groups=8*v*e, test_model_groups=8*t,
                model_groups=8*(u*b+v*e+t), saved_panel_bytes=4*(v+t)*65)

"""Stdlib-only, exact SKINNY single-pair layout training contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from cipher_lab.training.early_stopping import Settings

from ..config import Resources, Seeds, canonical, validate_run_id
from ..single_train.config import ARCHITECTURE, TASK, SingleRecipe

METHOD = 'innovation1-skinny-layout-early-stop-v1'
NAMESPACE = 'innovation1-skinny-layout-early-stop-v1-20261120'
FIXED_METHOD = 'innovation1-skinny-layout-fixed-mse-v1'
FIXED_NAMESPACE = 'innovation1-skinny-layout-fixed-mse-v1-20261120'
BCE_METHOD = 'innovation1-skinny-layout-fixed-bce-v1'
BCE_NAMESPACE = 'innovation1-skinny-layout-fixed-bce-v1-20261121'
LAYOUT_NAMESPACES = {METHOD: NAMESPACE, FIXED_METHOD: FIXED_NAMESPACE, BCE_METHOD: BCE_NAMESPACE}
SLOTS = ('A-row-best', 'A-prime-column-best')
SCANS = ('row', 'column')
SEEDS = Seeds(init=26111951, sample=26111952, shuffle=26111953)
PUBLIC_STREAMS = replace(ARCHITECTURE, namespace=NAMESPACE, tasks=(TASK,), seeds=SEEDS)


@dataclass(frozen=True)
class FilePin:
    bytes: int
    sha256: str


@dataclass(frozen=True)
class PoolSource:
    root: str
    run_id: str
    record: FilePin
    train: FilePin
    dev: FilePin


@dataclass(frozen=True)
class LayoutRecipe(SingleRecipe):
    method: str = METHOD
    run_id: str | None = None
    namespace: str = NAMESPACE
    resources: Resources = SingleRecipe.resources
    early_stopping: Settings | None = None
    cosine_epochs: int | None = None
    pool_source: PoolSource | None = None

    @property
    def fixed_pool(self) -> bool:
        return self.method in (FIXED_METHOD, BCE_METHOD)

    @property
    def reuse_pool(self) -> bool:
        return self.method == BCE_METHOD

    @property
    def loss_key(self) -> str:
        return 'mse' if self.method == FIXED_METHOD else 'bce'

    @property
    def shuffle_namespace(self) -> str:
        return FIXED_NAMESPACE if self.reuse_pool else self.namespace

    @property
    def artifact_schema(self) -> int:
        return 4 if self.reuse_pool else 3 if self.fixed_pool else 2

    def to_dict(self) -> dict[str, Any]:
        recipe = super().to_dict()
        model = recipe['model']
        for key in ('dedicated', 'router_hidden', 'top_k'):
            del model[key]
        model.update(route={'nibble': 0.5, 'word': 0.5},
                     scans=dict(zip(SLOTS, SCANS, strict=True)), parameters_each=1087234)
        recipe.update(method=self.method, run_id=self.run_id, namespace=self.namespace,
                      seeds=dict(init=SEEDS.init, sample=SEEDS.sample, shuffle=SEEDS.shuffle))
        if self.early_stopping is not None:
            recipe['selection']['early_stopping'] = dict(asdict(self.early_stopping),
                                                        stop_when='all')
        if self.cosine_epochs is not None:
            recipe['optimizer'].update(schedule='cycle-cosine-then-constant',
                                       cosine_epochs=self.cosine_epochs)
        if self.fixed_pool:
            tr = self.training
            train = tr.batches_per_epoch * tr.batch_groups
            recipe['data'] = dict(protocol='fixed-stratified-90-10-v1',
                                  pool_groups=train + tr.dev_groups, train_groups=train)
            if self.reuse_pool:
                if self.pool_source is None:
                    raise ValueError('POOL_RECIPE')
                recipe['data'].update(protocol='reuse-fixed-stratified-90-10-v1',
                                      shuffle_namespace=self.shuffle_namespace,
                                      source=asdict(self.pool_source))
            else:
                recipe['optimizer']['loss'] = 'mean-MSE-on-sigmoid'
                recipe['selection']['metric'] = 'per-model-dev-mse'
        return recipe


EARLY_STOP = LayoutRecipe(
    training=replace(SingleRecipe.training, epochs=100),
    resources=replace(SingleRecipe.resources, wall_seconds=43200),
    early_stopping=Settings(warmup_epochs=24, patience=12, min_delta=1e-5),
    cosine_epochs=24,
)

FIXED_MSE = replace(
    EARLY_STOP, method=FIXED_METHOD, namespace=FIXED_NAMESPACE,
    training=replace(EARLY_STOP.training, epochs=50, batches_per_epoch=1800,
                     batch_groups=5000, dev_groups=1_000_000),
)


FIXED_BCE = replace(
    FIXED_MSE, method=BCE_METHOD, namespace=BCE_NAMESPACE,
    pool_source=PoolSource(
        root=r'G:\lxy\shared-m-cipher-nd-runs\20261121-skinny-fixed-mse-01',
        run_id='innovation1-skinny-fixed-mse-20261121-01',
        record=FilePin(48314, '5c6c96d95ef39d941b841d051c3d3df22584bda1cc675ef74791c20e5e66c1d7'),
        train=FilePin(153012425,
                      '78e6d1a219d3a20f5c3746ae5c45bfb14f33028f09a6cd13fa4f81287739637d'),
        dev=FilePin(17012423, 'ed8792908f1ddca0842e97a55153f02a1efe2467d2a588ae69feae199145441a'),
    ),
)


def official(recipe: LayoutRecipe, run_id: str) -> LayoutRecipe:
    """Compare against the frozen base with ONLY the caller's ID substituted."""
    validate_run_id(run_id)
    base = {METHOD: EARLY_STOP, FIXED_METHOD: FIXED_MSE, BCE_METHOD: FIXED_BCE}.get(recipe.method)
    if base is None or (base.pool_source is not None
                        and run_id.casefold() == base.pool_source.run_id.casefold()):
        raise ValueError('CONFIG_EXACT')
    expected = replace(base, run_id=run_id)
    if (type(recipe) is not LayoutRecipe or recipe != expected
            or canonical(recipe.to_dict()) != canonical(expected.to_dict())):
        raise ValueError('CONFIG_EXACT')
    return expected

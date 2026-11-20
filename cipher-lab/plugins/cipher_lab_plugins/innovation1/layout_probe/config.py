"""Stdlib-only, exact SKINNY single-pair layout training contract."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from cipher_lab.training.early_stopping import Settings

from ..config import Resources, Seeds, _unique
from ..single_train.config import ARCHITECTURE, TASK, SingleRecipe, canonical

METHOD = 'innovation1-skinny-layout-train-v1'
RUN_ID = 'innovation1-skinny-layout-train-20261119-01'
NAMESPACE = 'innovation1-skinny-layout-train-v1-20261119'
SLOTS = ('A-row-best', 'A-prime-column-best')
SCANS = ('row', 'column')
SEEDS = Seeds(init=26111951, sample=26111952, shuffle=26111953)
PUBLIC_STREAMS = replace(ARCHITECTURE, namespace=NAMESPACE, tasks=(TASK,), seeds=SEEDS)


@dataclass(frozen=True)
class LayoutRecipe(SingleRecipe):
    method: str = METHOD
    run_id: str = RUN_ID
    namespace: str = NAMESPACE
    resources: Resources = SingleRecipe.resources
    early_stopping: Settings | None = None
    cosine_epochs: int | None = None

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
        return recipe


FIXED = LayoutRecipe()
EARLY_STOP = LayoutRecipe(
    training=replace(FIXED.training, epochs=100),
    resources=replace(FIXED.resources, wall_seconds=43200),
    method='innovation1-skinny-layout-early-stop-v1',
    run_id='innovation1-skinny-layout-early-stop-20261120-01',
    namespace='innovation1-skinny-layout-early-stop-v1-20261120',
    early_stopping=Settings(warmup_epochs=24, patience=12, min_delta=1e-5),
    cosine_epochs=24,
)
RECIPES = (FIXED, EARLY_STOP)


def load_config(path: Path) -> LayoutRecipe:
    with path.open('rb') as stream:
        content = stream.read(16385)
    if len(content) > 16384:
        raise ValueError('CONFIG_SIZE')
    try:
        raw = json.loads(content.decode('utf-8'), object_pairs_hook=_unique)
        encoded = canonical(raw)
        for recipe in RECIPES:
            if encoded == canonical(recipe.to_dict()):
                return recipe
    except (ValueError, RecursionError):
        pass
    raise ValueError('CONFIG_EXACT')

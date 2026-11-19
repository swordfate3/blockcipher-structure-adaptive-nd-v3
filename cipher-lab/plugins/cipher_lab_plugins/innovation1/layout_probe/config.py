"""Stdlib-only, exact SKINNY single-pair layout training contract."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..config import Seeds, _unique
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
    # All budget/optimizer fields remain those of the released single-pair recipe.
    def to_dict(self) -> dict[str, Any]:
        recipe = super().to_dict()
        model = recipe['model']
        for key in ('dedicated', 'router_hidden', 'top_k'):
            del model[key]
        model.update(route={'nibble': 0.5, 'word': 0.5},
                     scans=dict(zip(SLOTS, SCANS, strict=True)), parameters_each=1087234)
        recipe.update(method=METHOD, run_id=RUN_ID, namespace=NAMESPACE,
                      seeds=dict(init=SEEDS.init, sample=SEEDS.sample, shuffle=SEEDS.shuffle))
        return recipe


FIXED = LayoutRecipe()


def load_config(path: Path) -> LayoutRecipe:
    with path.open('rb') as stream:
        content = stream.read(16385)
    if len(content) > 16384:
        raise ValueError('CONFIG_SIZE')
    try:
        raw = json.loads(content.decode('utf-8'), object_pairs_hook=_unique)
        if canonical(raw) == canonical(FIXED.to_dict()):
            return FIXED
    except (ValueError, RecursionError):
        pass
    raise ValueError('CONFIG_EXACT')

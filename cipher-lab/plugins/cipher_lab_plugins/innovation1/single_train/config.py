"""Exact stdlib recipe and independent quantitative ledger limits."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks

from ..config import TASKS, Recipe, Training, _unique
from ..config import learning_rate as cosine

METHOD = 'innovation1-skinny-single-train-v1'
RUN_ID = 'innovation1-skinny-single-train-20261119-01'
NAMESPACE = 'innovation1-skinny-single-train-v1-20261119'
BACKEND = 'ciphers'
DEVICE = 'cuda:0'
GPU_UUID = 'GPU-171dfbbe-c1a7-d93b-549d-8824cb8a18c7'
GPU_PCI = '00000000:3B:00.0'
TASK = TASKS[-1]
SLOTS = ('M-best', 'D-SKINNY-best')
ARCHITECTURE = Recipe()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class SingleRecipe:
    # Only direct guarded engineering calls may replace the training sizes.
    training: Training = replace(ARCHITECTURE.training, pairs_per_group=1)
    resources = replace(ARCHITECTURE.resources, wall_seconds=28800, cpu_seconds=115200)

    def to_dict(self) -> dict[str, Any]:
        model = asdict(ARCHITECTURE.model)
        model['dedicated'] = ARCHITECTURE.model.dedicated[-1]
        return dict(method=METHOD, run_id=RUN_ID, namespace=NAMESPACE, backend=BACKEND,
                    task=asdict(TASK), device=DEVICE, gpu_uuid=GPU_UUID, gpu_pci=GPU_PCI,
                    training=asdict(self.training), optimizer=asdict(ARCHITECTURE.optimizer),
                    seeds=asdict(ARCHITECTURE.seeds), model=model, resources=asdict(self.resources),
                    selection=dict(metric='per-model-dev-bce', ties='earliest',
                                   interval='epoch-end', test='after-two-persisted-best-reloads'))


FIXED = SingleRecipe()


def load_config(path: Path) -> SingleRecipe:
    with path.open('rb') as stream:
        content = stream.read(16385)
    if len(content) > 16384:
        raise ValueError('CONFIG_SIZE')
    try:
        raw = json.loads(content.decode('utf-8'), object_pairs_hook=_unique)
        # Distinguish types (e.g. true, 1, 1.0) as well as values, with no overrides.
        if canonical(raw) == canonical(FIXED.to_dict()):
            return FIXED
    except (ValueError, RecursionError):
        pass
    raise ValueError('CONFIG_EXACT')


def estimate(recipe: SingleRecipe) -> dict[str, int]:
    tr = recipe.training
    sizes = (tr.batch_groups, tr.dev_groups, tr.test_groups, tr.eval_batch_groups)
    cycles = tr.epochs * tr.batches_per_epoch
    if (any(type(n) is not int or n <= 0 for n in (*sizes, tr.epochs, tr.batches_per_epoch))
            or tr.batch_groups % 4 or tr.eval_batch_groups % 4
            or tr.dev_groups % tr.eval_batch_groups or tr.test_groups % tr.eval_batch_groups
            or cycles < 2
            or (tr.pairs_per_group, tr.gradient_accumulation, tr.retries) != (1, 1, 0)):
        raise ValueError('RECIPE_SHAPE')
    train = cycles * tr.batch_groups
    return dict(fits=2, retries=0, cycles=cycles, updates_each=cycles,
                optimizer_updates=2 * cycles, train_groups=train, dev_groups=tr.dev_groups,
                test_groups=tr.test_groups,
                real_blocks=2 * (train + tr.dev_groups + tr.test_groups),
                forward_rows=2 * (train + tr.epochs * tr.dev_groups + tr.test_groups), null_words=0)


def make_policy(recipe: SingleRecipe, clocks: Clocks) -> Policy:
    n, limits = estimate(recipe), recipe.resources
    deadline = clocks.utc() + timedelta(seconds=limits.wall_seconds)
    return Policy(2, limits.wall_seconds, limits.cpu_seconds, deadline.isoformat(),
                  n['real_blocks'], 0, limits.rss_bytes, limits.artifact_bytes,
                  n['optimizer_updates'], n['forward_rows'])


def learning_rate(recipe: SingleRecipe, cycle: int) -> float:
    return cosine(replace(ARCHITECTURE, training=recipe.training), cycle)

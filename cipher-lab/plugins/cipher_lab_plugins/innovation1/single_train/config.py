"""Exact stdlib recipe and independent quantitative ledger limits."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from typing import Any

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks

from ..config import TASKS, Recipe, Training
from ..config import learning_rate as cosine

BACKEND = 'ciphers'
DEVICE = 'cuda:0'
TASK = TASKS[-1]
ARCHITECTURE = Recipe()


@dataclass(frozen=True)
class SingleRecipe:
    # Only direct guarded engineering calls may replace the training sizes.
    training: Training = replace(ARCHITECTURE.training, pairs_per_group=1)
    resources = replace(ARCHITECTURE.resources, wall_seconds=28800, cpu_seconds=115200)

    def to_dict(self) -> dict[str, Any]:
        model = asdict(ARCHITECTURE.model)
        return dict(backend=BACKEND, task=asdict(TASK), device=DEVICE,
                    training=asdict(self.training), optimizer=asdict(ARCHITECTURE.optimizer),
                    seeds=asdict(ARCHITECTURE.seeds), model=model, resources=asdict(self.resources),
                    selection=dict(metric='per-model-dev-bce', ties='earliest',
                                   interval='epoch-end', test='after-two-persisted-best-reloads'))


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


def learning_rate(recipe: SingleRecipe, cycle: int, *, cosine_epochs: int | None = None) -> float:
    tr = recipe.training
    if cosine_epochs is not None:
        if (type(cosine_epochs) is not int or not 1 <= cosine_epochs <= tr.epochs
                or cosine_epochs * tr.batches_per_epoch < 2
                or type(cycle) is not int or not 0 <= cycle < tr.epochs * tr.batches_per_epoch):
            raise ValueError('cycle outside schedule')
        if cycle >= cosine_epochs * tr.batches_per_epoch:
            return ARCHITECTURE.optimizer.lr_min
        tr = replace(tr, epochs=cosine_epochs)
    return cosine(replace(ARCHITECTURE, training=tr), cycle)

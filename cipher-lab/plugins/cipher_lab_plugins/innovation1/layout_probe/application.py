"""Lazy production assembly and layout-specific bindings for the shared loop."""
from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, Any

from ..single_train.application import Assemble, admit_device, execute_pair, summary_for
from ..single_train.config import BACKEND
from .config import FIXED, RECIPES, SLOTS, LayoutRecipe

if TYPE_CHECKING:
    from cipher_lab.artifacts import RunStore

    from ..resources import Resources
    from ..single_train.config import SingleRecipe
    from ..single_train.data import PairBatch


def _production(resources: Resources, evidence: dict[str, Any], *,
                recipe: LayoutRecipe) -> tuple[Any, Any]:
    admit_device(resources, evidence)
    from secrets import token_bytes

    from cipher_lab.ciphers import skinny

    from .data import Provider
    from .runtime import Runtime

    resources.check()
    runtime = Runtime(resources.check)
    return Provider(recipe, skinny.encrypt_many, token_bytes, resources), runtime


def _train(recipe: SingleRecipe, provider: Any, runtime: Any, store: RunStore,
           resources: Resources, summary: dict[str, Any], backend: str) -> None:
    from ..single_train.training import train_pair
    from .data import encode_panel
    from .metrics import paired_statistics
    from .weights import identity

    if type(recipe) is not LayoutRecipe:
        raise ValueError('LAYOUT_RECIPE')

    def test_result(panel: PairBatch, scores: dict[str, dict[str, float]],
                    logits: dict[str, Any]) -> dict[str, Any]:
        paired = paired_statistics(panel.labels, logits[SLOTS[0]], logits[SLOTS[1]])
        baseline = math.log(2.)
        return dict(paired_test=paired, constant_bce=baseline,
                    bce_minus_constant={slot: scores[slot]['bce'] - baseline for slot in SLOTS})

    train_pair(recipe, provider, runtime, store, resources, summary, backend, slots=SLOTS,
               panel_bytes=encode_panel, checkpoint_identity=identity, test_result=test_result,
               early_stopping=recipe.early_stopping, cosine_epochs=recipe.cosine_epochs)


def _execute(recipe: LayoutRecipe, store: RunStore, resources: Resources, backend: str,
             assemble: Assemble) -> None:
    execute_pair(recipe, store, resources, backend, assemble, _train,
                 summary_for(backend, method=recipe.method, slots=SLOTS))


def execute(store: RunStore, resources: Resources, *, recipe: LayoutRecipe = FIXED) -> None:
    if not any(recipe is exact for exact in RECIPES):
        raise ValueError('CONFIG_EXACT')
    _execute(recipe, store, resources, BACKEND, partial(_production, recipe=recipe))

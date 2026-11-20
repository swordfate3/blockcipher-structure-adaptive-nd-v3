"""Explicit layout sampling stream and saved-panel identity; no new sampler."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..data import BlockMeter, Oracle, PrivateSource
from ..single_train.config import SingleRecipe
from ..single_train.data import PairBatch, pack_panel
from ..single_train.data import Provider as SingleProvider
from .config import PUBLIC_STREAMS, SLOTS, LayoutRecipe


class Provider(SingleProvider):
    def __init__(self, recipe: LayoutRecipe, oracle: Oracle, private: PrivateSource,
                 meter: BlockMeter) -> None:
        super().__init__(recipe, oracle, private, meter,
                         streams=replace(PUBLIC_STREAMS, namespace=recipe.namespace))


def encode_panel(batch: PairBatch, recipe: SingleRecipe, source: dict[str, str],
                 backend: str, *, logits: dict[str, Any] | None = None,
                 weights: dict[str, Any] | None = None) -> bytes:
    if type(recipe) is not LayoutRecipe:
        raise ValueError('PANEL_IDENTITY')
    metadata = dict(schema=2, method=recipe.method, run_id=recipe.run_id, recipe=recipe.to_dict(),
                    source=source, backend=backend, split=batch.split, digest=batch.digest,
                    encoding='packed-msb0-single-pair-v1', slots=list(SLOTS))
    return pack_panel(batch, recipe, metadata, SLOTS, ('a_logits', 'a_prime_logits'),
                      recipe.namespace, logits=logits, weights=weights)

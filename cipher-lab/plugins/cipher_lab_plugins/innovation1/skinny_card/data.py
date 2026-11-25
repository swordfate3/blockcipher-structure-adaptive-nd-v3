"""Adapter for exactly five archived inputs and one aligned new TEST encoding."""
from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import canonical
from ..four_single.config import PANEL_OVERHEAD
from ..four_single.replay import ReplayPools
from ..single_train.config import TASK
from ..single_train.data import PairBatch
from .config import SLOTS, CardRecipe


class CardPools:
    """Fixed TRAIN + DEV, and TEST only via the shared loop's two-reload phase."""

    def __init__(self, recipe: CardRecipe, check: Callable[[], None]) -> None:
        self._reader = _SkinnyReader(recipe, check)
        self._dev: PairBatch | None = None
        self._provenance: dict[str, Any] = {}
        self.test_input: dict[str, Any] | None = None

    def prepare_pool(self) -> PairBatch:
        train, self._dev, self._provenance = self._reader.prepare_skinny()
        return train

    def activate(self) -> None:
        self._reader.activate()

    def train(self, cycle: int) -> PairBatch:
        return self._reader.train(TASK, cycle)

    def panel(self, split: str) -> PairBatch:
        if split != 'dev' or self._dev is None:
            raise ValueError('TEST_ORDER' if split == 'test' else 'POOL_NOT_READY')
        return self._dev

    def pool_provenance(self) -> dict[str, Any]:
        if not self._provenance:
            raise ValueError('POOL_NOT_READY')
        return self._provenance

    def load_test(self, reloads: dict[str, Any]) -> PairBatch:
        if (set(reloads) != set(SLOTS) or self._dev is None or self.test_input is not None
                or any(reloads[slot].get('ref') != f'model/{slot}.pt' for slot in SLOTS)):
            raise ValueError('TEST_ORDER')
        self._reader.release_train()
        self._dev = None
        batch, self.test_input = self._reader.load_tests(lambda _: None, only=TASK)[TASK.cipher]
        return batch


class _SkinnyReader(ReplayPools):
    def __init__(self, recipe: CardRecipe, check: Callable[[], None]) -> None:
        super().__init__(recipe.archive_reader_recipe(), check)
        self._test_identity = recipe.test_sha256, recipe.test_digest

    def prepare_skinny(self) -> tuple[PairBatch, PairBatch, dict[str, Any]]:
        (_, train, dev, evidence), = self._read_pools(
            lambda _: None, only=TASK, test_identity=self._test_identity)
        self._bind_train(TASK, train)
        return train, dev, evidence


def encode_test(batch: PairBatch, recipe: CardRecipe, source: dict[str, str], backend: str, *,
                logits: dict[str, NDArray[np.float64]], weights: dict[str, Any],
                reused_input: dict[str, Any]) -> bytes:
    if ((batch.split, batch.index, batch.groups, batch.digest, batch.namespace) !=
            ('test', 0, recipe.training.test_groups, recipe.test_digest,
             recipe.archive_reader_recipe().namespace)
            or reused_input.get('sha256') != recipe.test_sha256
            or reused_input.get('digest') != batch.digest
            or set(weights) != set(logits) or set(weights) != set(SLOTS)
            or any(z.shape != batch.labels.shape or z.dtype != np.float64
                   or not np.isfinite(z).all() for z in logits.values())):
        raise ValueError('TEST_PANEL_IDENTITY')
    meta = dict(schema=1, method=recipe.method, recipe=recipe.to_dict(), run_id=recipe.run_id,
                source=source, backend=backend, cipher=TASK.cipher, split='test', rows=batch.groups,
                encoding='packed-msb0-single-pair-v1', digest=batch.digest,
                namespace=batch.namespace, reused_input=reused_input, best=weights)
    stream = io.BytesIO()
    arrays: dict[str, Any] = dict(metadata=np.frombuffer(canonical(meta).encode('ascii'),
                                 dtype=np.uint8), ciphertext=batch.ciphertext, labels=batch.labels,
                                 **logits)
    # All members are validated numeric arrays. Older NumPy treats an
    # allow_pickle keyword to savez as an extra archive member.
    np.savez(stream, **arrays)
    raw = stream.getvalue()
    if len(raw) > batch.groups * 33 + PANEL_OVERHEAD:
        raise ValueError('PANEL_SIZE')
    return raw

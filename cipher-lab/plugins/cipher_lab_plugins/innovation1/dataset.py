"""Packed saved panels have experiment identity, not a model family."""
from __future__ import annotations

import io
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .batch import Batch
from .checkpoint import canonical, preflight
from .config import Recipe, Task

MAX_PANEL = 128 * 2**20


def _identity(recipe: Recipe, task: Task, split: str, index: int,
              source: Mapping[str, str], backend: str) -> dict[str, Any]:
    return dict(schema=1, method=recipe.method, recipe=recipe.to_dict(), task=vars(task),
                split=split, index=index, source=dict(source), backend=backend,
                synthetic=backend == 'demo', encoding='packed-msb0-four-pairs-v1')


def encode_panel(batch: Batch, recipe: Recipe, source: Mapping[str, str], backend: str) -> bytes:
    metadata = _identity(recipe, batch.task, batch.split, batch.index, source, backend)
    stream = io.BytesIO()
    np.savez(stream, metadata=np.frombuffer(canonical(metadata).encode(), dtype=np.uint8),
             ciphertext=batch.ciphertext, labels=batch.labels)
    return stream.getvalue()


def decode_panel(raw: bytes, recipe: Recipe, task: Task, split: str,
                 source: Mapping[str, str], backend: str) -> Batch:
    preflight(raw, MAX_PANEL)
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != {'metadata', 'ciphertext', 'labels'}:
                raise ValueError
            meta_bytes = archive['metadata']
            if meta_bytes.dtype != np.uint8 or meta_bytes.ndim != 1 or meta_bytes.size > 100000:
                raise ValueError
            meta = json.loads(meta_bytes.tobytes())
            if canonical(meta) != canonical(_identity(recipe, task, split, 0, source, backend)):
                raise ValueError
            ciphertext, labels = archive['ciphertext'], archive['labels']
            size = recipe.training.dev_groups if split == 'dev' else recipe.training.test_groups
            if (split not in ('dev', 'test') or ciphertext.dtype != np.uint8
                    or labels.dtype != np.uint8 or ciphertext.shape != (size, 4, 2, 8)
                    or labels.shape != (size, 1) or not np.isin(labels, [0, 1]).all()
                    or int(labels.sum()) != size // 2):
                raise ValueError
            return Batch(ciphertext, labels, task, split, 0)
    except Exception:
        raise ValueError('DATASET_INVALID') from None

"""Batched read-only model scoring and stable BCE, no sampling or training imports."""
from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np

from .batch import Batch
from .config import Task


def tensor_digest(x: Any, y: Any) -> str:
    digest = sha256()
    for tensor in (x, y):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def bce(labels: Any, logits: Any) -> float:
    y, z = np.asarray(labels, dtype=np.float64), np.asarray(logits, dtype=np.float64)
    if y.shape != z.shape or not np.isfinite(z).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('SCORE_INVALID')
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def mse(labels: Any, logits: Any) -> float:
    y, z = np.asarray(labels, dtype=np.float64), np.asarray(logits, dtype=np.float64)
    if y.shape != z.shape or not np.isfinite(z).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('SCORE_INVALID')
    # Probability error is sigmoid(-margin); square without overflow at large logits.
    margin = np.where(y == 1, z, -z)
    return float(np.mean(np.square(np.exp(-np.logaddexp(0., margin)))))


def score_panel(batch: Batch, runtime: Any, slots: tuple[str, ...],
                chunk: int, resources: Any) -> dict[str, Any]:
    values: dict[str, list[Any]] = {s: [] for s in slots}
    for start in range(0, batch.groups, chunk):
        resources.check()
        section = batch.slice(start, min(start + chunk, batch.groups))
        x, y = runtime.prepare(section)
        digest = tensor_digest(x, y)
        for slot in slots:
            values[slot].append(predict(runtime, slot, batch.task, x, y, resources))
            if tensor_digest(x, y) != digest:
                raise ValueError('BATCH_MUTATED')
    return {s: np.concatenate(v) for s, v in values.items()}


def predict(runtime: Any, slot: str, task: Task, x: Any, y: Any, resources: Any) -> Any:
    groups = int(y.shape[0])
    resources.reserve('forward_rows', groups)
    value = np.asarray(runtime.predict(slot, task, x), dtype=np.float64)
    if value.shape != (groups, 1) or not np.isfinite(value).all():
        raise ValueError('SCORE_INVALID')
    resources.complete('forward_rows', groups)
    resources.check()
    return value

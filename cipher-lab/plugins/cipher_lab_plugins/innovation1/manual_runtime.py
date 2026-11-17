"""Saved-input manual runtime only; no provider/generator dependency."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .batch import Batch
from .checkpoint import SLOTS, decode, encode
from .config import Recipe, Task


class ManualRuntime:
    backend = 'demo'

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self.states = {s: {'manual_steps': torch.tensor(0, dtype=torch.int64)} for s in SLOTS}
        self.restored: list[dict[str, Any]] = []
        self.consumed: dict[tuple[str, int], list[tuple[int, int]]] = {}

    def metadata(self) -> dict[str, Any]:
        return {slot: dict(parameters=0, buffers=0, state_bytes=8, manual_state_only=True)
                for slot in SLOTS}

    def prepare(self, batch: Batch) -> tuple[Any, Any]:
        return batch.tensors()

    def update(self, slot: str, task: Task, x: Any, y: Any, lr: float,
               resources: Any) -> float:
        resources.reserve('forward_rows', int(y.shape[0]))
        # Hand-written constant outputs; no module invocation/optimizer/backward.
        resources.complete('forward_rows', int(y.shape[0]))
        own = int(self.states[slot]['manual_steps'])
        cycle = own // 4 if slot == 'M-best' else own
        self.consumed.setdefault((task.cipher, cycle), []).append((id(x), id(y)))
        self.states[slot]['manual_steps'] += 1
        return 0.6931471805599453

    def predict(self, slot: str, task: Task, x: Any) -> Any:
        # Same DEV score at every epoch intentionally proves earliest-tie selection.
        return np.full((x.shape[0], 1), 0.0 if slot == 'M-best' else -0.1)

    def checkpoint(self, slot: str, metadata: dict[str, Any]) -> bytes:
        return encode(metadata, self.states[slot])

    def release(self) -> None:
        self.states.clear()

    def restore(self, slot: str, raw: bytes, metadata: dict[str, Any]) -> None:
        template = {'manual_steps': torch.tensor(0, dtype=torch.int64)}
        self.states[slot] = decode(raw, metadata, template)
        self.restored.append(metadata)

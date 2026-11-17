"""Explicit artificial engineering runtime: no cipher, private entropy, NN or Adam.

Never selected by the ciphers backend. Dummy state and hand-written logits do
not represent a trained model; block/forward/update counts are simulated work.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .batch import Batch
from .config import Recipe, Task
from .manual_runtime import ManualRuntime as ManualRuntime


class ManualProvider:
    def __init__(self, recipe: Recipe, resources: Any) -> None:
        self.recipe, self.resources = recipe, resources
        self.train_calls: list[tuple[str, int]] = []
        self.test_calls: list[str] = []
        self.panels: dict[tuple[str, str], Batch] = {}

    def _make(self, task: Task, split: str, index: int, groups: int) -> Batch:
        self.resources.reserve('real_blocks', 8 * groups)
        # Hand-filled PUBLIC fixture, not a fake oracle injected into ciphers.
        packed = np.zeros((groups, 4, 2, 8), dtype=np.uint8)
        labels = np.asarray([0, 1] * (groups // 2), dtype=np.uint8).reshape(-1, 1)
        packed[:, 0, 0, 0] = labels[:, 0]
        batch = Batch(packed, labels, task, split, index)
        self.resources.complete('real_blocks', 8 * groups)
        return batch

    def train(self, task: Task, cycle: int) -> Batch:
        if (task.cipher, cycle) in self.train_calls:
            raise ValueError('MANUAL_REPLAY')
        self.train_calls.append((task.cipher, cycle))
        return self._make(task, 'train', cycle, self.recipe.training.batch_groups)

    def prepare_dev(self) -> dict[str, Batch]:
        return {t.cipher: self.dev(t) for t in self.recipe.tasks}

    def dev(self, task: Task) -> Batch:
        return self._panel(task, 'dev', self.recipe.training.dev_groups)

    def test(self, task: Task) -> Batch:
        self.test_calls.append(task.cipher)
        return self._panel(task, 'test', self.recipe.training.test_groups)

    def _panel(self, task: Task, split: str, groups: int) -> Batch:
        key = (task.cipher, split)
        if key not in self.panels:
            self.panels[key] = self._make(task, split, 0, groups)
        return self.panels[key]

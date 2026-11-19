"""Layout training adapter, only imported after existing device admission."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from cipher_lab.representations.tasks import public_features

from ..config import Task
from ..models import TorchRuntime
from ..single_train.config import ARCHITECTURE, BACKEND, DEVICE, TASK, SingleRecipe
from ..single_train.data import PairBatch
from .config import SEEDS, SLOTS
from .models import LayoutProbe, build_pair, require_fp32
from .weights import FIXED_BUFFERS, decode


def _fresh_pair(check: Callable[[], None]) -> dict[str, LayoutProbe]:
    check()
    require_fp32()  # before build_pair, not just before the device cast
    pair = dict(zip(SLOTS, build_pair(init_seed=SEEDS.init), strict=True))
    for arm in pair.values():
        if set(dict(arm.named_buffers())) != FIXED_BUFFERS:
            raise ValueError('LAYOUT_BUFFERS')
    check()
    return pair


class Runtime(TorchRuntime):
    def __init__(self, check: Callable[[], None]) -> None:
        super().__init__(ARCHITECTURE, DEVICE, inference_only=True)
        pair = _fresh_pair(check)
        self._load_templates: dict[str, LayoutProbe] | None = None
        self.public = torch.tensor(public_features(TASK.cipher), device=self.device,
                                   dtype=torch.float32)
        o = ARCHITECTURE.optimizer
        for slot, model in pair.items():
            check()
            self.models[slot] = model.to(device=self.device, dtype=torch.float32)
            check()
            self.optimizers[slot] = torch.optim.Adam(
                model.parameters(), lr=o.lr_max, betas=o.betas, eps=o.eps,
                weight_decay=o.weight_decay, amsgrad=o.amsgrad)
            check()

    def prepare(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, PairBatch):
            raise ValueError('PAIR_BATCH')
        bits, labels = batch.tensors()
        return bits.to(self.device), labels.to(self.device)

    def _forward(self, slot: str, task: Task, x: Any) -> Any:
        if task != TASK:
            raise ValueError('TASK_MISMATCH')
        return self.models[slot](x, self.public, rounds=task.rounds)

    def load_best(self, slot: str, raw: bytes, recipe: SingleRecipe, source: dict[str, str],
                  epoch: int, score: float, check: Callable[[], None]) -> None:
        # The shared loop releases both training models before any reload. New
        # slot-specific templates start on CPU; neither is made from loaded state.
        if slot in self.models or self.optimizers:
            raise ValueError('RELOAD_NOT_RELEASED')
        if self._load_templates is None:
            self._load_templates = _fresh_pair(check)
        model = self._load_templates.pop(slot)
        state = decode(raw, recipe, source, BACKEND, slot, epoch, score, model.state_dict())
        check()
        model.load_state_dict(state, strict=True)
        check()
        self.models[slot] = model.to(device=self.device, dtype=torch.float32).eval()
        check()

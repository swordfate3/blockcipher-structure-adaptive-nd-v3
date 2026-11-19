"""Two-model gradient-capable adapter, constructed only after GPU proof."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from cipher_lab.representations.dedicated import author_pair_layout
from cipher_lab.representations.tasks import public_features

from ..config import Task
from ..models import TorchRuntime, construct
from ..single_pair.models import computation_slots, dedicated_logits, shared_logits
from .config import ARCHITECTURE, BACKEND, DEVICE, SLOTS, TASK, SingleRecipe
from .data import PairBatch
from .weights import decode


class Runtime(TorchRuntime):
    def __init__(self, check: Callable[[], None]) -> None:
        super().__init__(ARCHITECTURE, DEVICE, inference_only=True)
        check()
        self.public = torch.tensor(public_features(TASK.cipher), device=self.device,
                                   dtype=torch.float32)
        o = ARCHITECTURE.optimizer
        for slot in SLOTS:
            check()
            model = construct(ARCHITECTURE, slot).to(device=self.device, dtype=torch.float32)
            self.models[slot] = model
            check()
            self.optimizers[slot] = torch.optim.Adam(
                model.parameters(), lr=o.lr_max, betas=o.betas, eps=o.eps,
                weight_decay=o.weight_decay, amsgrad=o.amsgrad)
            check()

    def prepare(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, PairBatch):
            raise ValueError('PAIR_BATCH')
        bits, labels = batch.tensors()
        return computation_slots(bits).to(self.device), labels.to(self.device)

    def _forward(self, slot: str, task: Task, x: Any) -> Any:
        if task != TASK:
            raise ValueError('TASK_MISMATCH')
        model = self.models[slot]
        if slot == SLOTS[0]:
            return shared_logits(model, x, self.public, rounds=task.rounds)
        if slot == SLOTS[1]:
            # Same encoder (including live BN in update()) and same per-pair head.
            return dedicated_logits(model, x, author_pair_layout)
        raise ValueError('MODEL_SLOT')

    def load_best(self, slot: str, raw: bytes, recipe: SingleRecipe, source: dict[str, str],
                  epoch: int, score: float, check: Callable[[], None]) -> None:
        model = construct(ARCHITECTURE, slot)
        check()
        state = decode(raw, recipe, source, BACKEND, slot, epoch, score, model.state_dict())
        check()
        model.load_state_dict(state, strict=True)
        check()
        self.models[slot] = model.to(device=self.device, dtype=torch.float32).eval()
        check()

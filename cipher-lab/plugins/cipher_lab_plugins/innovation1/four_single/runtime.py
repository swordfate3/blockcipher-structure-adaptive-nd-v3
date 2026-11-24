"""Bind a single M to the existing FP32 Adam/clip/BCE runtime."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from torch import Tensor

from cipher_lab.representations.tasks import public_features

from ..config import Task
from ..models import TorchRuntime
from ..single_train.config import DEVICE
from ..single_train.data import PairBatch
from .config import SLOT, FourSingleRecipe
from .models import SinglePairM, build, geometry, load_state


class Runtime(TorchRuntime):
    def __init__(self, recipe: FourSingleRecipe, check: Callable[[], None]) -> None:
        super().__init__(recipe.streams, DEVICE, inference_only=True)
        self.contract = recipe
        check()
        model = self._build()
        self.tables = geometry(model)
        self.models[SLOT] = model.to(self.device, dtype=torch.float32)
        self.public = {t: torch.tensor(public_features(t.cipher), device=self.device,
                                       dtype=torch.float32) for t in recipe.tasks}
        check()
        self.add_adam(SLOT)
        check()

    def _build(self) -> SinglePairM:
        return build(self.contract)

    def metadata(self) -> dict[str, Any]:
        value = super().metadata()
        value[SLOT]['geometry'] = self.tables
        return value

    def prepare(self, batch: Any) -> tuple[Tensor, Tensor]:
        if not isinstance(batch, PairBatch):
            raise ValueError('PAIR_BATCH')
        x, y = batch.tensors()
        return x.to(self.device), y.to(self.device)

    def _forward(self, slot: str, task: Task, x: Any) -> Any:
        return self.models[slot](x, self.public[task], rounds=task.rounds)

    def routes(self) -> dict[str, dict[str, float]]:
        model = cast(SinglePairM, self.models[SLOT])
        return {task.cipher: model.route_weights(card) for task, card in self.public.items()}

    def restore(self, slot: str, raw: bytes, metadata: dict[str, Any]) -> None:
        if slot != SLOT or self.models or self.optimizers:
            raise ValueError('RELOAD_NOT_RELEASED')
        model = self._build()
        load_state(model, raw, metadata)
        self.models[SLOT] = model.to(self.device, dtype=torch.float32).eval()

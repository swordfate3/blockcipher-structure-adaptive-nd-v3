"""Two independent FP32 runtimes and exact slot/pool/geometry checkpoint binding."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from torch import Tensor

from cipher_lab.representations.tasks import public_features

from ..config import Task, canonical
from ..four_single.models import SinglePairM, geometry, load_state
from ..four_single.runtime import Runtime
from ..models import TorchRuntime
from ..single_train.config import BACKEND, DEVICE, SingleRecipe
from ..single_train.weights import bound_identity
from .config import (
    NEIGHBORHOOD,
    NIBBLE_BUFFER,
    SLOTS,
    TABLE_SHA256,
    TASK,
    NeighborhoodRecipe,
    official,
)
from .model import build


class NeighborhoodRuntime(TorchRuntime):
    prepare = Runtime.prepare

    def __init__(self, recipe: NeighborhoodRecipe, check: Callable[[], None],
                 pools: Callable[[], dict[str, Any]], backend: str = BACKEND) -> None:
        super().__init__(recipe.streams, DEVICE, inference_only=True)
        self.contract, self._pools, self.backend = recipe, pools, backend
        self.public = torch.tensor(public_features(TASK.cipher), device=self.device,
                                   dtype=torch.float32)
        self.tables: dict[str, dict[str, str]] = {}
        for slot, expected_table in zip(SLOTS, TABLE_SHA256, strict=True):
            check()
            model = self._build(slot)
            tables = geometry(model)
            if tables.get(NIBBLE_BUFFER) != expected_table:
                raise ValueError('MODEL_GEOMETRY')
            self.tables[slot] = tables
            self.models[slot] = model.to(self.device, dtype=torch.float32)
            self.add_adam(slot)
            check()

    def _build(self, slot: str) -> SinglePairM:
        return build(self.contract, slot)

    def metadata(self) -> dict[str, Any]:
        value = super().metadata()
        for slot in SLOTS:
            value[slot]['geometry'] = self.tables[slot]
        return value

    def _forward(self, slot: str, task: Task, x: Tensor) -> Any:
        if task != TASK:
            raise ValueError('MODEL_TASK')
        return self.models[slot](x, self.public, rounds=task.rounds)

    def routes(self, slot: str) -> dict[str, float]:
        return cast(SinglePairM, self.models[slot]).route_weights(self.public)

    def identity(self, recipe: SingleRecipe, source: dict[str, str], backend: str,
                 slot: str, epoch: int, score: float) -> dict[str, Any]:
        contract = self.contract
        if (type(recipe) is not NeighborhoodRecipe or recipe != contract or backend != self.backend
                or canonical(recipe.to_dict()) != canonical(contract.to_dict())):
            raise ValueError('CHECKPOINT_IDENTITY')
        if backend == BACKEND:
            official(contract, contract.run_id)
        meta = bound_identity(recipe, source, backend, slot, epoch, score, slots=SLOTS,
            fixed=NEIGHBORHOOD, schema=1, method=contract.method, run_id=contract.run_id)
        return dict(meta, geometry=self.tables[slot], pools=self._pools())

    def load_best(self, slot: str, raw: bytes, recipe: SingleRecipe, source: dict[str, str],
                  epoch: int, score: float, check: Callable[[], None]) -> None:
        if (slot not in SLOTS or self.optimizers
                or set(self.models) != set(SLOTS[:SLOTS.index(slot)])):
            raise ValueError('RELOAD_NOT_RELEASED')
        expected = self.identity(recipe, source, self.backend, slot, epoch, score)
        model = self._build(slot)
        load_state(model, raw, expected)
        check()
        self.models[slot] = model.to(self.device, dtype=torch.float32).eval()
        check()

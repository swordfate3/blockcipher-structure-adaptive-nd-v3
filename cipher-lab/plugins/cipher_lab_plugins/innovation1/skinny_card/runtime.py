"""Only the Top2 scorer input differs between independently owned models."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from torch import Tensor

from cipher_lab.representations.tasks import public_features

from ..config import Task, canonical, validate_run_id
from ..four_single.models import SinglePairM, build, geometry, load_state
from ..four_single.runtime import Runtime
from ..models import TorchRuntime
from ..serial_bc.models import NoCardM
from ..single_train.config import BACKEND, DEVICE, TASK, SingleRecipe
from ..single_train.weights import bound_identity
from .config import CARD, SLOTS, CardRecipe, official


class CardRuntime(TorchRuntime):
    prepare = Runtime.prepare  # exactly the same real single-pair tensor preparation

    def __init__(self, recipe: CardRecipe, check: Callable[[], None],
                 pools: Callable[[], dict[str, Any]], backend: str = BACKEND) -> None:
        super().__init__(recipe.streams, DEVICE, inference_only=True)
        self.contract, self._pools, self.backend = recipe, pools, backend
        self.tables: dict[str, dict[str, str]] = {}
        self.public = torch.tensor(public_features(TASK.cipher), device=self.device,
                                   dtype=torch.float32)
        for slot in SLOTS:
            check()
            model = self._build(slot)
            self.tables[slot] = geometry(model)
            self.models[slot] = model.to(self.device, dtype=torch.float32)
            self.add_adam(slot)
            check()

    def _build(self, slot: str) -> SinglePairM:
        if slot not in SLOTS:
            raise ValueError('MODEL_SLOT')
        return build(self.contract, SinglePairM if slot == SLOTS[0] else NoCardM)

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
        if (type(recipe) is not CardRecipe or recipe != contract or backend != self.backend
                or canonical(recipe.to_dict()) != canonical(contract.to_dict())):
            raise ValueError('CHECKPOINT_IDENTITY')
        validate_run_id(contract.run_id)
        if backend == BACKEND:
            official(contract, contract.run_id)
        meta = bound_identity(recipe, source, backend, slot, epoch, score, slots=SLOTS,
                              fixed=CARD, schema=1, method=contract.method, run_id=contract.run_id)
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

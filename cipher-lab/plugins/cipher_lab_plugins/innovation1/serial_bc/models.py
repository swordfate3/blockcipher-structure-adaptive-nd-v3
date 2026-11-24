"""C changes only the gate input; retain real cards and original geometry."""
from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from cipher_lab.models.shared_m.router import PublicRouter, Route
from cipher_lab.representations.tasks import validate_public

from ..config import canonical
from ..four_single.config import SLOT
from ..four_single.models import SinglePairM, build
from ..four_single.runtime import Runtime
from .config import SerialRecipe


def constant_route(router: PublicRouter) -> Route:
    # The public router's gate and exact stable Top2/softmax, without passing
    # a synthetic value through public card validation. No detach/no-grad here.
    first = cast(nn.Linear, router.gate[0])
    scores = cast(Tensor, router.gate(first.weight.new_full((13,), .5)))
    if not bool(torch.isfinite(scores).all()):
        raise ValueError('router scores must be finite')
    indices = torch.argsort(scores, descending=True, stable=True)[:2]
    return Route(indices, scores[indices].softmax(dim=0))


class NoCardM(SinglePairM):
    def _route(self, public: Tensor) -> Route:
        # The real card is still required by the forward and route diagnostic.
        # Neither card values nor rounds are used by the gate in C.
        validate_public(public)
        return constant_route(self.shared.router)


class SerialRuntime(Runtime):
    contract: SerialRecipe

    def _build(self) -> SinglePairM:
        mode = self.contract.routing_mode
        if mode == 'public-card':
            return build(self.contract)
        if mode == 'constant-no-card':
            return build(self.contract, NoCardM)
        raise ValueError('ROUTING_MODE')

    def restore(self, slot: str, raw: bytes, metadata: dict[str, object]) -> None:
        if (slot != SLOT or metadata.get('slot') != SLOT
                or type(metadata.get('stage')) is not int or metadata['stage'] != 4
                or metadata.get('routing_mode') != self.contract.routing_mode
                or metadata.get('run_id') != self.contract.run_id
                or canonical(metadata.get('recipe')) != canonical(self.contract.to_dict())):
            raise ValueError('CHECKPOINT_IDENTITY')
        super().restore(slot, raw, metadata)

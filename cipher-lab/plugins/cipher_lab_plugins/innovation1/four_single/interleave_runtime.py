"""Reuse the fixed B/C subsets and one shared runtime for the interleaved recipe."""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from cipher_lab.representations.tasks import public_features

from ..present_neighborhood.config import PRESENT_CARD
from ..present_neighborhood.layout import verified_card
from ..serial_bc.data import SerialPools
from .interleave_config import PRESENT_BUFFER, InterleaveRecipe
from .models import SinglePairM, build
from .runtime import Runtime


class InterleaveM(SinglePairM):
    extra_fixed_buffers = frozenset({PRESENT_BUFFER})
    present_structure: Tensor

    def __init__(self, shared: nn.Module) -> None:
        super().__init__(shared)
        default = self.shared.stems['nibble'].get_buffer('indices').tolist()
        table = verified_card(PRESENT_CARD.p_layer, PRESENT_CARD.gather_rule,
                              model_default=tuple(tuple(row) for row in default))
        self.register_buffer(PRESENT_BUFFER, torch.tensor(table, dtype=torch.int64))

    def _present(self, raw: Tensor, channel: Tensor) -> Tensor:
        return self._gather(raw, channel, self.present_structure)

    def _nibble(self, public: Tensor) -> Callable[[Tensor, Tensor], Tensor] | None:
        if torch.equal(public, public.new_tensor(public_features('PRESENT'))):
            return self._present
        return super()._nibble(public)


class InterleaveRuntime(Runtime):
    def _build(self) -> SinglePairM:
        return build(self.contract, InterleaveM)


def assemble_pools(recipe: InterleaveRecipe, check: Callable[[], None]) -> SerialPools:
    return SerialPools(recipe.subset_recipe(), check, defer_activate=True)

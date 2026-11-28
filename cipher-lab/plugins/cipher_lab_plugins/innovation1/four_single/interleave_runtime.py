"""Reuse the fixed B/C subsets and one shared runtime for the interleaved recipe."""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from cipher_lab.representations.tasks import public_features

from ..present_neighborhood.config import PRESENT_CARD
from ..present_neighborhood.layout import verified_card
from ..serial_bc.data import SerialPools
from .interleave_config import (
    NIBBLE_BUFFERS,
    PRESENT_BUFFER,
    WORD_BUFFERS,
    FeistelWordRecipe,
    FullStructureInterleaveRecipe,
    InterleaveRecipe,
    StructureInterleaveRecipe,
)
from .models import SinglePairM, build
from .replay import ReplayPools
from .runtime import Runtime
from .structure_cards import WORD_ROTATIONS, verified_tables, verified_word_tables


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


class StructureCardM(SinglePairM):
    extra_fixed_buffers = frozenset({PRESENT_BUFFER})

    def __init__(self, shared: nn.Module) -> None:
        tables = verified_tables()
        super().__init__(shared, column_table=torch.tensor(tables['SKINNY'], dtype=torch.int64))
        # The two linear layouts have identical pinned tables and share one buffer.
        self.shared.stems['nibble'].register_buffer(
            'indices', torch.tensor(tables['SIMECK'], dtype=torch.int64))
        self.register_buffer(PRESENT_BUFFER, torch.tensor(tables['PRESENT'], dtype=torch.int64))

    def _nibble(self, public: Tensor) -> Callable[[Tensor, Tensor], Tensor] | None:
        for cipher, name in NIBBLE_BUFFERS.items():
            if torch.equal(public, public.new_tensor(public_features(cipher))):
                table = self.get_buffer(name)
                return lambda raw, channel: self._gather(raw, channel, table)
        return None


class FeistelWordM(StructureCardM):
    extra_fixed_buffers = StructureCardM.extra_fixed_buffers | frozenset(WORD_BUFFERS.values())

    def __init__(self, shared: nn.Module) -> None:
        super().__init__(shared)
        for cipher, table in verified_word_tables().items():
            public = public_features(cipher)
            if tuple(int(v * 32) for v in public[8:11]) != WORD_ROTATIONS[cipher]:
                raise ValueError('STRUCTURE_CARD_WORD_ROTATIONS')
            self.register_buffer(WORD_BUFFERS[cipher], torch.tensor(table, dtype=torch.int64))

    def _word(self, public: Tensor) -> Callable[[Tensor, Tensor], Tensor] | None:
        for cipher, name in WORD_BUFFERS.items():
            if torch.equal(public, public.new_tensor(public_features(cipher))):
                table = self.get_buffer(name)
                return lambda raw, channel: self._gather(raw, channel, table, stem_name='word')
        return None


class InterleaveRuntime(Runtime):
    def _build(self) -> SinglePairM:
        if isinstance(self.contract, FeistelWordRecipe):
            return build(self.contract, FeistelWordM)
        if isinstance(self.contract, StructureInterleaveRecipe):
            return build(self.contract, StructureCardM)
        return build(self.contract, InterleaveM)


def assemble_pools(recipe: InterleaveRecipe, check: Callable[[], None]) -> ReplayPools:
    if isinstance(recipe, FullStructureInterleaveRecipe):
        return ReplayPools(recipe.archive_reader_recipe(), check)
    return SerialPools(recipe.subset_recipe(), check, defer_activate=True)

"""Fuse public position features before the original convolutional suffix."""
from collections import OrderedDict
from typing import cast

import torch
from torch import Tensor, nn

from ..present_card_fresh import model as pure
from .wiring import gather_for


class Fusion(nn.Module):
    gather: Tensor

    def __init__(self, gather: tuple[int, ...]) -> None:
        super().__init__()
        self.register_buffer('gather', torch.tensor(gather, dtype=torch.long, device='cpu'))
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Conv1d(96, 96, 1, bias=False, device='cpu', dtype=torch.float32)
        nn.init.zeros_(self.projection.weight)

    def forward(self, h: Tensor) -> Tensor:
        return h + cast(Tensor, self.projection(h.index_select(-1, self.gather)))


class FusedSuffix(nn.Sequential):
    fusion: Fusion

    def __init__(self, suffix: nn.Sequential, gather: tuple[int, ...]) -> None:
        super().__init__(OrderedDict([('fusion', Fusion(gather)), *suffix.named_children()]))


def add_fusion(net: pure.Classifier, slot: str) -> pure.Classifier:
    gather = gather_for(slot)
    if gather:
        net.pair.shared.suffix = FusedSuffix(net.pair.shared.suffix, gather)
    return net


def build(slot: str) -> pure.Classifier:
    return add_fusion(pure.build(), slot)

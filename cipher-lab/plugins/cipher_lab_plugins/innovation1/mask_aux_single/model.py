"""Shared-M trunk with an independent class logit for each C7 pair."""
from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from ..prefix_bridge import config
from ..prefix_bridge.model import PrefixSharedM
from ..prefix_bridge.model import build as build_classifier


class PairClassifier(nn.Module):
    def __init__(self, classifier: PrefixSharedM) -> None:
        super().__init__()
        self.classifier = classifier

    def _features(self, bits: Tensor) -> Tensor:
        n = len(bits)
        if bits.ndim != 4 or bits.shape[1:] != (1, 6, 64) or not n or n % 4:
            raise ValueError('PAIR_LAYOUT')
        # Four slots are only a reshape accepted by the old public-card trunk.
        # Stems and suffix flatten these slots before any per-row GroupNorm.
        hidden = self.classifier.features(bits.reshape(n // 4, 4, 6, 64))
        if hidden.shape != (n // 4, 4, config.CHANNELS, 64):
            raise ValueError('PAIR_FEATURES')
        return hidden.reshape(n, config.CHANNELS, 64)

    def forward(self, bits: Tensor) -> Tensor:
        flat = self._features(bits).flatten(1)
        readout = self.classifier.readout
        # Never call the old pooled readout. Both branches act on the same pair.
        return cast(Tensor, readout.base(flat)) + cast(Tensor, readout.residual(flat))


class PairAux(PairClassifier):
    def __init__(self, classifier: PrefixSharedM) -> None:
        super().__init__(classifier)
        self.aux_head = nn.Conv1d(config.CHANNELS, 2, kernel_size=1)

    def masked_logits(self, masked: Tensor) -> Tensor:
        return cast(Tensor, self.aux_head(self._features(masked))).unsqueeze(1)


def build(auxiliary: bool) -> PairClassifier:
    classifier = build_classifier(config.INIT_SEED)
    if not auxiliary:
        return PairClassifier(classifier)
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.default_generator.manual_seed(config.INIT_SEED)
        return PairAux(classifier).float()

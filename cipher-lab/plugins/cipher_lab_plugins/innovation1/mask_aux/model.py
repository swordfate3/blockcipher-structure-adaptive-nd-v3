"""Complete-view classifier and a separate training-only readout of its trunk."""
from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from ..prefix_bridge import config
from ..prefix_bridge.model import PrefixSharedM
from ..prefix_bridge.model import build as build_classifier


class MaskAux(nn.Module):
    def __init__(self, classifier: PrefixSharedM) -> None:
        super().__init__()
        self.classifier = classifier
        self.aux_head = nn.Conv1d(config.CHANNELS, 2, kernel_size=1)

    def forward(self, full: Tensor) -> Tensor:
        return cast(Tensor, self.classifier(full))

    def masked_logits(self, masked: Tensor) -> Tensor:
        features = self.classifier.features(masked)
        return cast(Tensor, self.aux_head(features.reshape(-1, config.CHANNELS, 64))).reshape(
            -1, 4, 2, 64)


def build() -> MaskAux:
    classifier = build_classifier(config.INIT_SEED)
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.default_generator.manual_seed(config.INIT_SEED)
        return MaskAux(classifier).float()

"""The original full classifier plus a shared-trunk masked auxiliary view."""
from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from cipher_lab.models.shared_m.components import GeometryStem
from cipher_lab.representations.neighborhoods import gather_neighborhood

from ..present_card_fresh import config as pure_config
from ..present_card_fresh import model as pure
from . import config


class MaskedCard(nn.Module):
    def __init__(self, classifier: pure.Classifier) -> None:
        super().__init__()
        self.classifier = classifier
        self.flags = nn.ModuleDict({name: nn.Conv1d(config.FLAG_WIDTH, config.WIDTH, 1, bias=False)
                                    for name, _ in pure_config.ROUTE})
        for projection in self.flags.values():
            nn.init.zeros_(cast(nn.Conv1d, projection).weight)
        self.aux_head = nn.Conv1d(config.WIDTH, config.HEADS, 1)

    def forward(self, bits: Tensor) -> Tensor:
        return self.classifier.forward(bits)

    def masked_logits(self, masked: Tensor) -> Tensor:
        if (not self.training or masked.ndim != 4 or masked.shape[1:] != (1, 6, 64)
                or not len(masked) or len(masked) % 4 or masked.dtype != torch.float32
                or masked.device != self.classifier.public.device):
            raise ValueError('MASKED_CARD_INPUT')
        slots = masked.reshape(-1, 4, 6, 64)
        flags, observations = slots[:, :, :3], slots[:, :, 3:]
        shared = self.classifier.pair.shared
        branch = []
        for name, weight in pure_config.ROUTE:
            stem = cast(GeometryStem, shared.stems[name])
            values = stem.network[0](gather_neighborhood(observations, stem.indices))
            markers = self.flags[name](gather_neighborhood(flags, stem.indices))
            branch.append(weight * stem.network[1:](values + markers))
        # PRESENT has a zero public relation input, in both views.
        hidden = shared.suffix(branch[0] + branch[1])
        return cast(Tensor, self.aux_head(hidden)).unsqueeze(1)


def build() -> MaskedCard:
    classifier = pure.build()
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.default_generator.manual_seed(config.MASK_SEED)
        model = MaskedCard(classifier)
    if sum(p.numel() for p in model.parameters()) != config.PARAMETERS:
        raise ValueError('MODEL_PARAMETERS')
    return model

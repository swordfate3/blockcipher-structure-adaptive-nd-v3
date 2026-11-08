"""Learned stable top2 routing, taking ONLY one public13 card."""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from cipher_lab.representations.tasks import validate_public


@dataclass(frozen=True)
class Route:
    indices: Tensor
    weights: Tensor


class PublicRouter(nn.Module):
    def __init__(self, hidden_width: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(13, hidden_width), nn.ReLU(), nn.Linear(hidden_width, 3))

    def forward(self, public: Tensor) -> Route:
        validate_public(public)
        first = cast(nn.Linear, self.gate[0])
        if public.device != first.weight.device or public.dtype != first.weight.dtype:
            raise ValueError("public card and router must have same dtype/device")
        scores = cast(Tensor, self.gate(public))
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("router scores must be finite")
        indices = torch.argsort(scores, descending=True, stable=True)[:2]
        return Route(indices, scores[indices].softmax(dim=0))

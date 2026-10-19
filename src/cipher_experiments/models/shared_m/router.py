"""Task-update-level routing: public13 only, no sample-dependent input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from cipher_experiments.ciphers.tasks import validate_public
from cipher_experiments.models.shared_m.config import RouterConfig, validate_pair


@dataclass(frozen=True)
class Route:
    indices: torch.Tensor
    weights: torch.Tensor


class PublicRouter(nn.Module):
    def __init__(self, config: RouterConfig = RouterConfig()) -> None:  # noqa: B008
        # Frozen dataclass: safe immutable default, part of the explicit public API.
        super().__init__()
        self.config = config
        # Keep gate parameters in every control, so state import is always complete.
        self.gate = nn.Sequential(
            nn.Linear(13, config.hidden_width), nn.ReLU(), nn.Linear(config.hidden_width, 3)
        )

    def forward(
        self,
        public: torch.Tensor,
        noise: torch.Tensor | None = None,
        *,
        forced_pair: tuple[int, int] | None = None,
    ) -> Route:
        validate_public(public)
        if not public.is_floating_point():
            raise ValueError("public card must be floating point")
        if forced_pair is not None:
            pair = validate_pair(forced_pair)
            if not self.training or noise is not None:
                raise ValueError("warmup pair requires training and no noise")
            return Route(torch.tensor(pair, device=public.device), public.new_full((2,), 0.5))
        if noise is not None and (
            not self.training
            or noise.shape != (3,)
            or noise.device != public.device
            or not bool(torch.isfinite(noise).all())
        ):
            raise ValueError("noise must be a finite training-only same-device three-vector")
        if self.config.mode == "fixed":
            if noise is not None:
                raise ValueError("fixed routing forbids exploration noise")
            pair = validate_pair(self.config.fixed_pair)
            return Route(torch.tensor(pair, device=public.device), public.new_full((2,), 0.5))
        card = torch.zeros_like(public) if self.config.mode == "constant" else public
        scores = cast(torch.Tensor, self.gate(card))
        if noise is not None:
            scores = scores + noise
        indices = torch.argsort(scores, descending=True, stable=True)[:2]
        return Route(indices, scores[indices].softmax(dim=0))

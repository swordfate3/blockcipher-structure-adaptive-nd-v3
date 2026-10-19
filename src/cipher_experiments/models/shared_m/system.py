"""Independent shared distinguisher; no experiment inheritance or runner imports."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from cipher_experiments.models.shared_m.components import GeometryStem, Residual, SharedReadout
from cipher_experiments.models.shared_m.config import STEM_NAMES, ModelConfig
from cipher_experiments.models.shared_m.router import PublicRouter, Route
from cipher_experiments.representations.input import InputRepresentation


class _AverageFusion(nn.Conv1d):
    """Pointwise slot fusion with no random draws, even during construction."""

    def reset_parameters(self) -> None:
        channels = self.out_channels
        with torch.no_grad():
            self.weight.zero_()
            identity = torch.eye(channels, device=self.weight.device, dtype=self.weight.dtype)
            self.weight[:, :channels, 0].copy_(0.5 * identity)
            self.weight[:, channels:, 0].copy_(0.5 * identity)
            if self.bias is not None:
                self.bias.zero_()


@dataclass(frozen=True)
class ForwardOutput:
    logits: torch.Tensor
    auxiliary_loss: torch.Tensor


class SharedDistinguisher(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:  # noqa: B008
        # Frozen dataclass: safe immutable default, as declared by the public API.
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise ValueError("config must be a ModelConfig")
        self.config = config
        self.representation = InputRepresentation(config.representation)
        self.stems = nn.ModuleDict({name: GeometryStem(name, config) for name in STEM_NAMES})
        initial = copy.deepcopy(cast(GeometryStem, self.stems["local"]).network.state_dict())
        for name in STEM_NAMES[1:]:
            cast(GeometryStem, self.stems[name]).network.load_state_dict(initial)
        self.router = PublicRouter(config.router)
        self.suffix = nn.Sequential(
            *[Residual(k, config.channels, config.norm_groups) for k in config.suffix_kernels]
        )
        self.readout = SharedReadout(config)
        self.fusion: nn.Conv1d | None = None
        if config.view_mode != "single":
            self.fusion = _AverageFusion(2 * config.channels, config.channels, 1)
        self.last_route: Route | None = None

    def route(
        self,
        public: torch.Tensor,
        noise: torch.Tensor | None = None,
        *,
        forced_pair: tuple[int, int] | None = None,
    ) -> Route:
        return cast(Route, self.router(public, noise, forced_pair=forced_pair))

    def forward(
        self,
        bits: torch.Tensor,
        public: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        forced_pair: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        # A failed forward must not leave a stale route for auxiliary penalties.
        self.last_route = None
        if self.config.view_mode == "single":
            represented = self.representation(bits, public)
            route = self.route(public, noise, forced_pair=forced_pair)
            features: list[torch.Tensor] = []
            for index in route.indices.tolist():
                features.append(
                    self.stems[STEM_NAMES[index]](represented.canonical, represented.channel)
                )
            # Preserve legacy sum's operation order (including the initial +0).
            mixed = cast(torch.Tensor, sum(route.weights[i] * h for i, h in enumerate(features)))
        else:
            slots = self.representation.dual_views(bits, public, view_mode=self.config.view_mode)
            route = self.route(public, noise, forced_pair=forced_pair)
            assert self.fusion is not None
            slot_features: list[torch.Tensor] = []
            for slot in slots:
                features = [
                    self.stems[STEM_NAMES[index]](
                        slot.canonical.to(dtype=self.fusion.weight.dtype),
                        slot.channel.to(dtype=self.fusion.weight.dtype),
                    )
                    for index in route.indices.tolist()
                ]
                slot_features.append(
                    cast(torch.Tensor, sum(route.weights[i] * h for i, h in enumerate(features)))
                )
            # Concatenate channels, never pair/bit axes: [B,4,2*C,64] -> [B*4,2*C,64].
            mixed = cast(
                torch.Tensor,
                self.fusion(
                    torch.cat(slot_features, dim=2).reshape(-1, 2 * self.config.channels, 64)
                ),
            )
        hidden = self.suffix(mixed.reshape(-1, self.config.channels, 64)).reshape(
            -1, 4, self.config.channels, 64
        )
        logits = cast(torch.Tensor, self.readout(hidden))
        self.last_route = route
        return logits

    def kernel_penalty(self) -> torch.Tensor:
        """Legacy SkinnyPairMoE L2 scope (+ fusion in dual), NOT all trainable kernels.

        The inherited ComponentMoE implementation includes selected stem main
        networks, suffix and linear base head ONLY. Relation projections and
        nonlinear residual head were added in subclasses without overriding it;
        they (and the router) must remain excluded to reproduce that objective.
        """
        if self.last_route is None:
            raise ValueError("kernel penalty requires a successful forward route")
        modules: list[nn.Module] = [
            cast(GeometryStem, self.stems[STEM_NAMES[i]]).network
            for i in self.last_route.indices.tolist()
        ]
        modules += [self.suffix, self.readout.base]
        # Dual scope adds fusion only; the shared stems are counted once, not per slot.
        if self.fusion is not None:
            modules.append(self.fusion)
        return torch.stack(
            [
                layer.weight.square().sum()
                for module in modules
                for layer in module.modules()
                if isinstance(layer, (nn.Conv1d, nn.Linear))
            ]
        ).sum()

    def forward_output(
        self,
        bits: torch.Tensor,
        public: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        forced_pair: tuple[int, int] | None = None,
    ) -> ForwardOutput:
        """Training interface: auxiliary_loss is unweighted mode-specific kernel L2."""
        logits = self(bits, public, noise=noise, forced_pair=forced_pair)
        return ForwardOutput(logits, self.kernel_penalty())


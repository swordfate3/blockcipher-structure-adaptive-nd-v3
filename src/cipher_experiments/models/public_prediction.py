"""Public-input research predictors with explicit, key-free constructor contracts."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from ..representations.prediction_public import (
    public_interactions,
    zero_fixed_background,
    zero_round_constant_reference,
)


def _integer(value: int, name: str, minimum: int, maximum: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in int")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _choice(value: str, name: str, choices: tuple[str, ...]) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a built-in str")
    if value not in choices:
        raise ValueError(f"invalid {name}")


class PublicPredictor(nn.Module):
    """Map FP32 binary plaintexts to absolute-label logits, without secret inputs.

    Background match-direct changes the initial first bias exactly once. A
    subsequent strict state load overwrites that bias; forward never changes it.
    The checkpoint SHA is assigned externally by the strict loader, not here.
    """

    indices: Tensor
    checkpoint_config_sha256: str

    def __init__(
        self,
        hidden_width: int = 128,
        *,
        model_kind: str,
        public_input: str,
        public_readout: str,
        background_init: str,
        rounds: int,
        target_bit: int = 0,
        plaintext_space: str = "full",
        plaintext_fixed_value: int = 0,
        l1_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        _integer(hidden_width, "hidden_width", 1, 4096)
        _integer(rounds, "rounds", 3, 10)
        _integer(target_bit, "target_bit", 0, 0)
        _integer(plaintext_fixed_value, "plaintext_fixed_value", 0, 0xFFFFFFFF)
        _choice(model_kind, "model_kind", ("public-mlp", "public-linear"))
        _choice(public_input, "public_input", ("identity", "background-zero", "simon-interactions"))
        _choice(public_readout, "public_readout", ("direct", "zero-round-constant-residual"))
        _choice(background_init, "background_init", ("standard", "match-direct"))
        _choice(plaintext_space, "plaintext_space", ("full", "high32-zero"))
        if type(l1_penalty) not in (int, float):
            raise TypeError("l1_penalty must be a built-in int or float")
        # Exact allowed values also reject NaN/inf without accepting conversions.
        expected_l1 = 1e-4 if model_kind == "public-linear" else 0.0
        if l1_penalty != expected_l1:
            raise ValueError("invalid l1_penalty for model_kind")

        if public_input == "background-zero":
            if (model_kind != "public-mlp" or rounds != 3
                    or plaintext_space != "high32-zero" or public_readout != "direct"):
                raise ValueError("background-zero requires public-mlp/r3/high32-zero/direct")
        else:
            if (rounds not in (6, 8, 10) or plaintext_space != "full"
                    or plaintext_fixed_value != 0 or background_init != "standard"):
                raise ValueError("public inputs require r6/8/10, full/fixed0, standard init")
        if model_kind == "public-linear" and (
            hidden_width != 128 or public_input != "simon-interactions"
            or public_readout != "zero-round-constant-residual" or background_init != "standard"
        ):
            raise ValueError("public-linear requires width128/interactions/residual/standard")

        self.hidden_width = hidden_width
        self.model_kind = model_kind
        self.public_input = public_input
        self.public_readout = public_readout
        self.background_init = background_init
        self.rounds = rounds
        self.target_bit = target_bit
        self.plaintext_space = plaintext_space
        self.plaintext_fixed_value = plaintext_fixed_value
        self.l1_penalty = float(l1_penalty)
        self.input_width = 96 if public_input == "simon-interactions" else 64
        if model_kind == "public-mlp":
            # Preserve the old registration order and RNG draws for equal shapes.
            self.layers = nn.Sequential(
                nn.Linear(self.input_width, hidden_width, dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(hidden_width, hidden_width, dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(hidden_width, 1, dtype=torch.float32),
            )
        else:
            self.layers = nn.Sequential(nn.Linear(96, 1, dtype=torch.float32))
        self.register_buffer("indices", torch.arange(64), persistent=False)
        if background_init == "match-direct":
            first = cast(nn.Linear, self.layers[0])
            fixed_bits = torch.tensor(
                [(plaintext_fixed_value >> shift) & 1 for shift in range(31, -1, -1)],
                dtype=torch.float32, device=first.weight.device,
            )
            with torch.no_grad():
                first.bias.add_(first.weight[:, :32] @ fixed_bits)

    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if (inputs.layout != torch.strided or inputs.ndim != 2
                or inputs.shape[0] < 1 or inputs.shape[1] != 64):
            raise ValueError("inputs must have nonempty shape [N,64]")
        if inputs.dtype != torch.float32:
            raise TypeError("inputs must have dtype torch.float32")
        if inputs.device != next(self.parameters()).device:
            raise ValueError("inputs and model must be on the same device")
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("inputs must be finite")
        if not bool(((inputs == 0) | (inputs == 1)).all()):
            raise ValueError("inputs must contain binary values")
        if self.public_input == "background-zero":
            features = zero_fixed_background(
                inputs, plaintext_space=self.plaintext_space, fixed_value=self.plaintext_fixed_value
            )
        elif self.public_input == "simon-interactions":
            features = public_interactions(inputs)
        else:
            features = inputs
        logits = self.layers(features)
        if self.public_readout == "zero-round-constant-residual":
            reference = zero_round_constant_reference(
                inputs, rounds=self.rounds, target_bit=self.target_bit
            )
            logits = (1.0 - 2.0 * reference) * logits
        if logits.shape != (inputs.shape[0], 1) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("model produced invalid logits")
        return cast(Tensor, logits)

    def regularization_loss(self) -> Tensor:
        """Weight-only L1 for the research linear model, no parameter mutation."""
        if self.model_kind == "public-linear":
            first = cast(nn.Linear, self.layers[0])
            return self.l1_penalty * first.weight.abs().sum()
        return next(self.parameters()).new_zeros(())

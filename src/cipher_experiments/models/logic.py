"""Differentiable soft-logic plaintext bit predictor, independent of cipher/data code."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from .prediction import (
    _validate_builtin_int,
    _validate_plaintexts,
    public_reference,
    support_indices,
)


class SoftGateLayer(nn.Module):
    """Mix all 16 two-input Boolean functions over independent Bernoulli wires.

    For inputs ``a`` and ``b``, each truth table is evaluated as the expectation
    over independently sampled Bernoulli inputs. This relaxation does not model
    joint correlations between wires. A learned softmax over truth tables mixes
    those 16 expected values; the result remains in [0, 1].
    """

    gate_logits: nn.Parameter
    left_indices: Tensor
    right_indices: Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = _validate_builtin_int(in_features, "in_features", 1, 4096)
        self.out_features = _validate_builtin_int(out_features, "out_features", 1, 4096)

        # Left wires cover input coordinates in order. Right wires are sampled
        # from the caller-seeded global Torch RNG, with no self-pairs where avoidable.
        left = torch.arange(out_features, dtype=torch.long).remainder(in_features)
        right = torch.randint(in_features, (out_features,), dtype=torch.long)
        if in_features > 1:
            right = torch.where(right == left, (right + 1).remainder(in_features), right)
        self.register_buffer("left_indices", left, persistent=True)
        self.register_buffer("right_indices", right, persistent=True)

        self.gate_logits = nn.Parameter(torch.empty(out_features, 16, dtype=torch.float32))
        nn.init.normal_(self.gate_logits, mean=0.0, std=0.05)
        # Local convention: table bits correspond to corners 00,01,10,11 in
        # ascending bit significance. Thus A is mask 12 and B is mask 10.
        # A mild A preference preserves initial input gradients without fixing gates.
        with torch.no_grad():
            self.gate_logits[:, 12].add_(4.0)

    @staticmethod
    def gate_values(a: Tensor, b: Tensor) -> Tensor:
        """Evaluate every truth table, returning values in the final dimension.

        The four truth-table entries are ordered by input corners 00, 01, 10,
        11, and corner ``k`` is stored in mask bit ``k`` (00 is least
        significant). Under this local convention A and B have IDs 12 and 10;
        these are not IDs from any other convention.
        """
        if not isinstance(a, Tensor) or not isinstance(b, Tensor):
            raise TypeError("gate inputs must be torch.Tensor values")
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise TypeError("gate inputs must have dtype torch.float32")
        if a.device != b.device:
            raise ValueError("gate inputs must be on the same device")
        for value in (a, b):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("gate inputs must be finite")
            if not bool(((value >= 0) & (value <= 1)).all()):
                raise ValueError("gate inputs must be in [0,1]")
        if a.shape != b.shape:
            raise ValueError("gate inputs must have matching shapes")

        corner_probabilities = torch.stack(
            ((1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b), dim=-1
        )
        masks = torch.arange(16, dtype=torch.long, device=a.device)
        corners = torch.arange(4, dtype=torch.long, device=a.device)
        truth_tables = ((masks[:, None] >> corners[None, :]) & 1).to(dtype=a.dtype)
        return corner_probabilities @ truth_tables.transpose(0, 1)

    def validate_wiring(self) -> None:
        """Reject malformed/corrupt persistent wire buffers."""
        expected_shape = (self.out_features,)
        for name, wires in (
            ("left_indices", self.left_indices),
            ("right_indices", self.right_indices),
        ):
            if not isinstance(wires, Tensor) or wires.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")
            if wires.dtype != torch.long:
                raise TypeError(f"{name} must have dtype torch.long")
            if wires.device != self.gate_logits.device:
                raise ValueError(f"{name} and gate parameters must be on the same device")
            if not bool(((wires >= 0) & (wires < self.in_features)).all()):
                raise ValueError(f"{name} contains an out-of-range wire index")
        if self.in_features > 1 and bool((self.left_indices == self.right_indices).any()):
            raise ValueError("wire pairs must not connect an input to itself")

    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if (
            inputs.layout != torch.strided
            or inputs.ndim != 2
            or inputs.shape[0] < 1
            or inputs.shape[1] != self.in_features
        ):
            raise ValueError(f"inputs must have nonempty shape [N,{self.in_features}]")
        if inputs.dtype != torch.float32:
            raise TypeError("inputs must have dtype torch.float32")
        if inputs.device != self.gate_logits.device:
            raise ValueError("inputs and model must be on the same device")
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("inputs must be finite")
        if not bool(((inputs >= 0) & (inputs <= 1)).all()):
            raise ValueError("inputs must be in [0,1]")
        self.validate_wiring()

        a = inputs.index_select(1, self.left_indices)
        b = inputs.index_select(1, self.right_indices)
        values = self.gate_values(a, b)
        outputs = (values * torch.softmax(self.gate_logits, dim=-1).unsqueeze(0)).sum(dim=-1)
        if not bool(torch.isfinite(outputs).all()) or not bool(
            ((outputs >= 0) & (outputs <= 1)).all()
        ):
            raise RuntimeError("soft gate produced invalid bounded outputs")
        return outputs


class LogicPredictor(nn.Module):
    """Predict one absolute plaintext-conditioned output-bit logit.

    Inputs are binary float32 plaintexts. Hidden soft-gate activations stay
    continuous in [0,1]; evaluation does not harden gates or argmax truth tables.
    With direct output, three two-input multilinear gate layers have real
    polynomial degree at most eight in their selected inputs. This is not the
    same as GF(2) algebraic degree and does not guarantee target expressivity;
    public-residual sign modulation can raise the overall degree. This
    independent architecture does not claim to reproduce another model.
    """

    indices: Tensor
    checkpoint_config_sha256: str

    def __init__(
        self,
        hidden_width: int = 64,
        *,
        rounds: int = 3,
        target_bit: int = 0,
        feature_mode: str = "full",
        prediction_mode: str = "direct",
    ) -> None:
        super().__init__()
        hidden_width = _validate_builtin_int(hidden_width, "hidden_width", 1, 4096)
        rounds = _validate_builtin_int(rounds, "rounds", 1, 44)
        target_bit = _validate_builtin_int(target_bit, "target_bit", 0, 63)
        if feature_mode not in ("full", "support"):
            raise ValueError("feature_mode must be 'full' or 'support'")
        if prediction_mode not in ("direct", "public-residual"):
            raise ValueError("prediction_mode must be 'direct' or 'public-residual'")
        if prediction_mode == "public-residual" and rounds > 4:
            raise ValueError("public-residual mode supports rounds 1..4")

        selected = (
            tuple(range(64))
            if feature_mode == "full"
            else support_indices(rounds, target_bit)
        )
        self.hidden_width = hidden_width
        self.input_width = len(selected)
        if hidden_width < self.input_width:
            raise ValueError(
                "hidden_width must be at least input_width to cover all selected features"
            )
        self.rounds = rounds
        self.target_bit = target_bit
        self.feature_mode = feature_mode
        self.prediction_mode = prediction_mode
        self.register_buffer(
            "indices", torch.tensor(selected, dtype=torch.long), persistent=False
        )
        self.first_gate = SoftGateLayer(self.input_width, hidden_width)
        self.second_gate = SoftGateLayer(hidden_width, hidden_width)
        self.third_gate = SoftGateLayer(hidden_width, hidden_width)
        self.head = nn.Linear(hidden_width, 1)
        self.checkpoint_config_sha256 = ""

    @property
    def gate_layers(self) -> tuple[SoftGateLayer, SoftGateLayer, SoftGateLayer]:
        """Return the three trainable gates in forward order."""
        return self.first_gate, self.second_gate, self.third_gate

    def validate_wiring(self) -> None:
        """Validate persistent wiring after construction or strict checkpoint loading."""
        for layer in self.gate_layers:
            layer.validate_wiring()

    def forward(self, inputs: Tensor) -> Tensor:
        _validate_plaintexts(inputs)
        if inputs.device != next(self.parameters()).device:
            raise ValueError("inputs and model must be on the same device")
        self.validate_wiring()
        hidden = inputs.index_select(1, self.indices)
        for layer in self.gate_layers:
            hidden = layer(hidden)
        logits = self.head(hidden)
        if self.prediction_mode == "public-residual":
            reference = public_reference(inputs, self.rounds, self.target_bit)
            logits = (1.0 - 2.0 * reference.to(dtype=logits.dtype)) * logits
        if logits.shape != (inputs.shape[0], 1) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("model produced invalid logits")
        return cast(Tensor, logits)

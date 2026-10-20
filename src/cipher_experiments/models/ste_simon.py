"""Straight-through hard SIMON output-bit circuit with trainable symbolic constants."""

from __future__ import annotations

import torch
from torch import Tensor

from .prediction import _validate_plaintexts
from .structured_simon import StructuredSimonPredictor


class STESimonPredictor(StructuredSimonPredictor):
    """Hard-forward predictor with sigmoid-surrogate gradients for key constants.

    The inherited public cone and its plaintext wiring stay fixed. Only the
    inherited ``key_logits`` parameter is trainable; its thresholded values are
    used in the forward pass while sigmoid values provide a straight-through
    gradient surrogate. Returned logits are exactly -2/+2, not calibrated
    probabilities. The surrogate gradient is biased and is not the Boolean
    circuit's true derivative.
    """

    def __init__(self, *, rounds: int = 3, target_bit: int = 0) -> None:
        super().__init__(rounds=rounds, target_bit=target_bit)
        with torch.no_grad():
            self.key_logits.mul_(0.05)

    def _evaluate_bit(self, inputs: Tensor, key_values: Tensor) -> Tensor:
        """Evaluate the inherited Boolean DAG using the supplied key leaves."""
        _validate_plaintexts(inputs)
        if inputs.device != self.key_logits.device:
            raise ValueError("inputs and model must be on the same device")

        selected_inputs = inputs.index_select(1, self.indices)
        values: list[Tensor] = []
        for node in self.cone.nodes:
            if node.kind == "plaintext":
                (coordinate,) = node.coordinate
                values.append(selected_inputs[:, self._plaintext_positions[coordinate]])
            elif node.kind == "round_key":
                values.append(
                    key_values[self._key_positions[(node.coordinate[0], node.coordinate[1])]]
                    .expand(inputs.shape[0])
                )
            else:
                left, right = node.args
                left_values, right_values = values[left], values[right]
                if node.kind == "and":
                    values.append(left_values * right_values)
                elif node.kind == "xor":
                    values.append(left_values + right_values - 2.0 * left_values * right_values)
                else:  # SimonCone validates node kinds before this module receives it.
                    raise RuntimeError(f"unsupported node in fixed SIMON cone: {node.kind!r}")

        return values[self.cone.output].unsqueeze(1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return exact hard-bit logits with a sigmoid straight-through surrogate."""
        hard = (self.key_logits > 0).to(dtype=torch.float32)
        relaxed = torch.sigmoid(self.key_logits)
        key_values = hard + (relaxed - relaxed.detach())
        bit = self._evaluate_bit(inputs, key_values)
        logits = 2.0 * (2.0 * bit - 1.0)
        if logits.dtype != torch.float32 or logits.shape != (inputs.shape[0], 1):
            raise RuntimeError("model produced invalid logits")
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("model produced non-finite logits")
        return logits

    def hard_logits(self, inputs: Tensor) -> Tensor:
        """Evaluate the Boolean circuit without a gradient surrogate (ties are bit 0)."""
        key_values = (self.key_logits > 0).to(dtype=torch.float32)
        bit = self._evaluate_bit(inputs, key_values)
        logits = 2.0 * (2.0 * bit - 1.0)
        if logits.dtype != torch.float32 or logits.shape != (inputs.shape[0], 1):
            raise RuntimeError("model produced invalid hard logits")
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("model produced non-finite hard logits")
        return logits

"""Differentiable predictor over a fixed public SIMON output-bit cone."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..representations.simon_cone import SimonCone, simon_cone
from .prediction import _validate_builtin_int, _validate_plaintexts


class StructuredSimonPredictor(nn.Module):
    """Learn only symbolic round-key effects in a fixed SIMON64/128 bit cone.

    The public cone and plaintext wiring are fixed at construction. Each unique
    symbolic round-key coordinate has one trainable logit; neither actual key
    material nor cipher labels enter this model.
    """

    indices: Tensor
    checkpoint_config_sha256: str
    key_logits: nn.Parameter

    def __init__(self, *, rounds: int = 3, target_bit: int = 0) -> None:
        super().__init__()
        rounds = _validate_builtin_int(rounds, "rounds", 1, 16)
        target_bit = _validate_builtin_int(target_bit, "target_bit", 0, 63)
        cone = simon_cone(rounds, target_bit)
        key_coordinates = cone.round_key_coordinates
        if not key_coordinates:
            raise ValueError("the selected cone contains no learnable round-key coordinates")

        self.rounds = rounds
        self.target_bit = target_bit
        self.cone: SimonCone = cone
        self.key_coordinates = key_coordinates
        self._plaintext_positions = {
            coordinate: position for position, coordinate in enumerate(cone.plaintext_indices)
        }
        self._key_positions = {
            coordinate: position for position, coordinate in enumerate(key_coordinates)
        }
        self.register_buffer(
            "indices", torch.tensor(cone.plaintext_indices, dtype=torch.long), persistent=False
        )

        signs = torch.randint(0, 2, (len(key_coordinates),), dtype=torch.long)
        signs = (2 * signs - 1).to(dtype=torch.float32)
        noise = torch.randn(len(key_coordinates), dtype=torch.float32) * 0.05
        self.key_logits = nn.Parameter(2.0 * signs + noise)
        self.checkpoint_config_sha256 = ""

    def _probabilities(self, inputs: Tensor, *, hard: bool) -> Tensor:
        """Evaluate the fixed DAG with soft or explicitly thresholded key leaves."""
        _validate_plaintexts(inputs)
        if inputs.device != self.key_logits.device:
            raise ValueError("inputs and model must be on the same device")

        selected_inputs = inputs.index_select(1, self.indices)
        if hard:
            key_values = (self.key_logits > 0).to(dtype=torch.float32)
        else:
            key_values = torch.sigmoid(self.key_logits).to(dtype=torch.float32)

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

    @staticmethod
    def _stable_logits(probabilities: Tensor) -> Tensor:
        probabilities = probabilities.clamp(min=1e-6, max=1.0 - 1e-6)
        return torch.log(probabilities) - torch.log1p(-probabilities)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return finite FP32 logits using differentiable sigmoid key leaves."""
        logits = self._stable_logits(self._probabilities(inputs, hard=False))
        if logits.dtype != torch.float32 or logits.shape != (inputs.shape[0], 1):
            raise RuntimeError("model produced invalid logits")
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("model produced non-finite logits")
        return logits

    def hard_logits(self, inputs: Tensor) -> Tensor:
        """Explicitly evaluate with key bits ``key_logits > 0`` (zero ties are 0)."""
        logits = self._stable_logits(self._probabilities(inputs, hard=True))
        if logits.dtype != torch.float32 or logits.shape != (inputs.shape[0], 1):
            raise RuntimeError("model produced invalid hard logits")
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("model produced non-finite hard logits")
        return logits


class HardSimonPredictor(StructuredSimonPredictor):
    """Frozen Boolean circuit selected solely by TRAIN consistency search."""

    def __init__(self, *, rounds: int = 3, target_bit: int = 0) -> None:
        super().__init__(rounds=rounds, target_bit=target_bit)
        with torch.no_grad():
            self.key_logits.fill_(-1.)
        self.key_logits.requires_grad_(False)

    def set_assignment(self, assignment: int) -> None:
        if type(assignment) is not int or not 0 <= assignment < 2 ** len(self.key_coordinates):
            raise ValueError("assignment is outside the symbolic constant domain")
        with torch.no_grad():
            self.key_logits.copy_(torch.tensor(
                [1. if (assignment >> i) & 1 else -1. for i in range(len(self.key_coordinates))],
                device=self.key_logits.device))

    def validate_assignment(self) -> None:
        if not bool(((self.key_logits == -1.) | (self.key_logits == 1.)).all()):
            raise ValueError("hard assignment state must contain only -1/+1")

    def forward(self, inputs: Tensor) -> Tensor:
        return self.hard_logits(inputs)

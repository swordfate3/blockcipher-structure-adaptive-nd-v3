"""Fixed 8-bit soft-gate circuit predictor, independent of cipher/data code."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from .logic import SoftGateLayer
from .prediction import _validate_plaintexts

_ALIGNED_PAIRS: tuple[
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
] = (
    ((0, 2, 4, 6), (1, 3, 5, 7)),
    ((0, 2), (1, 3)),
    ((0,), (1,)),
)
_CANONICAL_SHAPES = ((8, 4), (4, 2), (2, 1))


class CircuitPredictor(nn.Module):
    """Predict a single plaintext-conditioned logit using an 8→4→2→1 circuit.

    Only the first eight plaintext bits are used, in their original order.
    Persistent gate wiring is either the fixed aligned pairing or one seeded
    perfect matching per layer; all gates retain ``SoftGateLayer``'s existing
    initialization. The scalar readout starts at weight 4 and bias -2.
    """

    indices: Tensor
    checkpoint_config_sha256: str

    def __init__(self, *, wiring: str = "aligned") -> None:
        super().__init__()
        if not isinstance(wiring, str):
            raise TypeError("wiring must be a string")
        if wiring not in ("aligned", "random"):
            raise ValueError("wiring must be 'aligned' or 'random'")
        self.wiring = wiring
        self.register_buffer("indices", torch.arange(8, dtype=torch.long), persistent=False)

        # Build every gate before sampling replacement wire matchings. This keeps
        # gate parameter initialization independent of the wiring choice.
        self.first_gate = SoftGateLayer(8, 4)
        self.second_gate = SoftGateLayer(4, 2)
        self.third_gate = SoftGateLayer(2, 1)

        for layer_index, layer in enumerate(self.gate_layers):
            # Consume the same permutation RNG draws for aligned and random
            # wiring, so a caller's seed yields identical trainable parameters.
            permutation = torch.randperm(layer.in_features, dtype=torch.long)
            if wiring == "aligned":
                left_values, right_values = _ALIGNED_PAIRS[layer_index]
                left = torch.tensor(left_values, dtype=torch.long)
                right = torch.tensor(right_values, dtype=torch.long)
            else:
                pairs = permutation.reshape(layer.out_features, 2)
                left = pairs[:, 0]
                right = pairs[:, 1]
            with torch.no_grad():
                layer.left_indices.copy_(left)
                layer.right_indices.copy_(right)

        self.head = nn.Linear(1, 1)
        with torch.no_grad():
            self.head.weight.fill_(4.0)
            self.head.bias.fill_(-2.0)
        self.checkpoint_config_sha256 = ""

    @property
    def gate_layers(self) -> tuple[SoftGateLayer, SoftGateLayer, SoftGateLayer]:
        """Return the three trainable gates in forward order."""
        return self.first_gate, self.second_gate, self.third_gate

    def validate_wiring(self) -> None:
        """Validate canonical shapes and full, nonduplicated layer coverage."""
        for index, (layer, expected_shape) in enumerate(
            zip(self.gate_layers, _CANONICAL_SHAPES, strict=True)
        ):
            expected_inputs, expected_outputs = expected_shape
            if (layer.in_features, layer.out_features) != expected_shape:
                raise ValueError(
                    f"gate layer {index} must have canonical shape "
                    f"{expected_inputs}->{expected_outputs}"
                )
            if layer.gate_logits.shape != (expected_outputs, 16):
                raise ValueError(f"gate layer {index} has noncanonical gate parameter shape")
            layer.validate_wiring()
            wires = torch.cat((layer.left_indices, layer.right_indices))
            coverage = torch.bincount(wires, minlength=expected_inputs)
            if not bool((coverage == 1).all()):
                raise ValueError(f"gate layer {index} wires must cover every input exactly once")
            if self.wiring == "aligned":
                left, right = _ALIGNED_PAIRS[index]
                if (layer.left_indices.tolist() != list(left)
                        or layer.right_indices.tolist() != list(right)):
                    raise ValueError(f"aligned gate layer {index} has noncanonical wiring")

        if (
            not isinstance(self.head, nn.Linear)
            or self.head.in_features != 1
            or self.head.out_features != 1
            or self.head.weight.shape != (1, 1)
            or self.head.bias is None
            or self.head.bias.shape != (1,)
        ):
            raise ValueError("readout must be a biased nn.Linear(1, 1)")

    def forward(self, inputs: Tensor) -> Tensor:
        _validate_plaintexts(inputs)
        if inputs.device != next(self.parameters()).device:
            raise ValueError("inputs and model must be on the same device")
        self.validate_wiring()
        hidden = inputs.index_select(1, self.indices)
        for layer in self.gate_layers:
            hidden = layer(hidden)
        logits = self.head(hidden)
        if logits.shape != (inputs.shape[0], 1) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("model produced invalid logits")
        return cast(Tensor, logits)

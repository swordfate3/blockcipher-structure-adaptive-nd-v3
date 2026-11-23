"""MLP for raw plaintext bits, single block or concatenated pair."""
from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

# Public block geometry only; importing the MLP does not import old recipes/modes.
BITS = 64
CONCAT_INPUT_BITS = 2 * BITS


class DirectBlockMLP(nn.Module):
    def __init__(self, *, input_width: int = BITS, output_width: int = BITS,
                 device: str | torch.device = 'cpu') -> None:
        super().__init__()
        if type(input_width) is not int or input_width not in (BITS, CONCAT_INPUT_BITS):
            raise ValueError('input width must be 64 or 128')
        if type(output_width) is not int or output_width not in (8, BITS, CONCAT_INPUT_BITS):
            raise ValueError('output width must be 8, 64 or 128')
        self._input_width, self._output_width = input_width, output_width
        self._initial_device = torch.device(device)
        self.layers = nn.Sequential(
            nn.Linear(input_width, 128, device=device, dtype=torch.float32), nn.ReLU(),
            nn.Linear(128, 128, device=device, dtype=torch.float32), nn.ReLU(),
            nn.Linear(128, output_width, device=device, dtype=torch.float32),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if (not isinstance(inputs, Tensor) or inputs.layout != torch.strided
                or inputs.ndim != 2 or not inputs.shape[0]
                or inputs.shape[1] != self._input_width):
            raise ValueError(f'expected nonempty plaintext shape [N,{self._input_width}]')
        first = next(self.parameters(), None)
        device = first.device if first is not None else self._initial_device
        if inputs.device != device or inputs.dtype != torch.float32:
            kind = 'CPU' if device.type == 'cpu' else 'same-device'
            raise ValueError(f'expected {kind} FP32 plaintext')
        if not bool(((inputs == 0) | (inputs == 1)).all()):
            raise ValueError('expected binary plaintext')
        logits = cast(Tensor, self.layers(inputs))
        if (logits.shape != (inputs.shape[0], self._output_width)
                or not bool(logits.isfinite().all())):
            raise ValueError('invalid logits')
        return logits

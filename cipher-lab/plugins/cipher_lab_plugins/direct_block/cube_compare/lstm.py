"""A stateless two-layer LSTM over paired MSB0 plaintext bit positions."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class CubeLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(2, 128, num_layers=2, batch_first=True, bias=True,
                            dropout=0., bidirectional=False, proj_size=0)
        self.head = nn.Linear(128, 128)

    def forward(self, inputs: Tensor) -> Tensor:
        # P || P′: each time step sees the same bit position in both blocks.
        sequence = torch.stack((inputs[:, :64], inputs[:, 64:]), dim=-1)
        shape = (2, inputs.shape[0], 128)
        h0, c0 = inputs.new_zeros(shape), inputs.new_zeros(shape)
        _, (hn, _) = self.lstm(sequence, (h0, c0))
        logits: Tensor = self.head(hn[-1])
        return logits

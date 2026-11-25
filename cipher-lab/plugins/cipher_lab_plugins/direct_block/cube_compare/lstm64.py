"""Stateless LSTM over one MSB0 plaintext bit per step."""
from __future__ import annotations

from torch import Tensor, nn


class DirectLSTM64(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, 128, num_layers=2, batch_first=True, bias=True,
                            dropout=0., bidirectional=False, proj_size=0)
        self.head = nn.Linear(128, 64)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != 64:
            raise ValueError('DirectLSTM64 expects inputs shaped [N, 64]')
        shape = (2, inputs.shape[0], 128)
        h0, c0 = inputs.new_zeros(shape), inputs.new_zeros(shape)
        _, (hn, _) = self.lstm(inputs.unsqueeze(-1), (h0, c0))
        logits: Tensor = self.head(hn[-1])
        return logits

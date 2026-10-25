"""A key-free, differentiable unrolled SIMON-shaped bit predictor."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .prediction import _validate_builtin_int, _validate_plaintexts


class UnrolledSimonPredictor(nn.Module):
    """Predict one output bit with a shared learned cell over several rounds.

    Inputs are public 64-bit plaintexts only.  Each position's cell sees four
    centered local state values and a learned round/position condition.  The
    intermediate sigmoid states are continuous relaxations, not exact cipher
    states, and the learned conditions are not actual key material. Both
    plaintext-xor modes sign the selected two-round high-half logit by its
    corresponding plaintext bit; ``plaintext-xor-no-final-low`` additionally
    replaces only the final cell's centered low-state input channel with zero.
    Targets remain absolute labels.
    """

    conditions: nn.Parameter
    indices: Tensor
    checkpoint_config_sha256: str

    def __init__(
        self,
        hidden_width: int = 32,
        *,
        rounds: int = 3,
        target_bit: int = 0,
        prediction_mode: str = "direct",
    ) -> None:
        super().__init__()
        hidden_width = _validate_builtin_int(hidden_width, "hidden_width", 1, 4096)
        rounds = _validate_builtin_int(rounds, "rounds", 1, 8)
        target_bit = _validate_builtin_int(target_bit, "target_bit", 0, 63)
        xor_modes = ("plaintext-xor", "plaintext-xor-no-final-low")
        if prediction_mode not in ("direct", *xor_modes):
            raise ValueError("prediction_mode must be direct or a plaintext-xor mode")
        if prediction_mode in xor_modes and (rounds != 2 or target_bit >= 32):
            raise ValueError("plaintext-xor modes require rounds=2 and target_bit in the high half")
        if prediction_mode == "direct" and rounds == 1 and target_bit >= 32:
            raise ValueError(
                "one-round low-half outputs are copied inputs with no trainable output"
            )

        self.hidden_width = hidden_width
        self.rounds = rounds
        self.target_bit = target_bit
        self.feature_mode = "full"
        self.prediction_mode = prediction_mode
        self.checkpoint_config_sha256 = ""
        self.register_buffer("indices", torch.arange(64), persistent=False)

        self.cell = nn.Sequential(
            nn.Linear(8, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 1),
        )
        self.conditions = nn.Parameter(torch.randn(rounds, 32, 4) * 0.05)

    @staticmethod
    def _rotate_left(values: Tensor, amount: int) -> Tensor:
        """Rotate a 32-position state so output position i reads i+amount."""
        return torch.cat((values[:, amount:], values[:, :amount]), dim=1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return one finite float32 logit per plaintext row."""
        _validate_plaintexts(inputs)
        if inputs.device != self.conditions.device:
            raise ValueError("inputs and model must be on the same device")

        high, low = inputs[:, :32], inputs[:, 32:]
        previous_logits: Tensor | None = None
        logits: Tensor | None = None
        for round_index in range(self.rounds):
            # Center each public local state value from {0,1} to {-1,+1}.
            local_state = torch.stack(
                (
                    low,
                    self._rotate_left(high, 1),
                    self._rotate_left(high, 8),
                    self._rotate_left(high, 2),
                ),
                dim=-1,
            )
            local_state = local_state.mul(2.0).sub(1.0)
            if (self.prediction_mode == "plaintext-xor-no-final-low"
                    and round_index == self.rounds - 1):
                local_state = torch.cat(
                    (torch.zeros_like(local_state[..., :1]), local_state[..., 1:]), dim=-1
                )
            round_condition = (
                self.conditions[round_index].unsqueeze(0).expand(inputs.shape[0], -1, -1)
            )
            cell_inputs = torch.cat((local_state, round_condition), dim=-1)
            logits = self.cell(cell_inputs).squeeze(-1)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("model produced non-finite logits")

            previous_logits = logits if round_index == self.rounds - 2 else previous_logits
            low, high = high, torch.sigmoid(logits)

        if logits is None:
            raise RuntimeError("model has no output round")
        if self.target_bit < 32:
            selected = logits[:, self.target_bit : self.target_bit + 1]
        else:
            if previous_logits is None:
                raise RuntimeError("low-half output has no preceding trainable round")
            bit = self.target_bit - 32
            selected = previous_logits[:, bit : bit + 1]
        if self.prediction_mode in ("plaintext-xor", "plaintext-xor-no-final-low"):
            plaintext_bit = inputs[:, self.target_bit : self.target_bit + 1]
            selected = (1.0 - 2.0 * plaintext_bit) * selected
        if selected.shape != (inputs.shape[0], 1) or not bool(torch.isfinite(selected).all()):
            raise RuntimeError("model produced invalid logits")
        return selected

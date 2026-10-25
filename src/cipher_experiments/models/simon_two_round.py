"""Key-free public-feature predictors specialized to two SIMON rounds."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from .prediction import _validate_builtin_int, _validate_plaintexts


class _TwoRoundPredictor(nn.Module):
    """Shared public-bit indexing and input validation for fixed two-round models."""

    rounds: int
    target_bit: int
    indices: Tensor
    checkpoint_config_sha256: str

    def __init__(self, *, rounds: int, target_bit: int) -> None:
        super().__init__()
        self.rounds = _validate_builtin_int(rounds, "rounds", 2, 2)
        self.target_bit = _validate_builtin_int(target_bit, "target_bit", 0, 31)
        self.checkpoint_config_sha256 = ""

        q_positions = tuple((target_bit + offset) % 32 for offset in (1, 8, 2))
        dependencies = {target_bit}
        for q in q_positions:
            dependencies.add(32 + q)
            dependencies.update((q + offset) % 32 for offset in (1, 8, 2))
        ordered = tuple(sorted(dependencies))
        self.register_buffer("indices", torch.tensor(ordered, dtype=torch.long), persistent=False)
        self._input_positions = {absolute: position for position, absolute in enumerate(ordered)}
        self._q_positions = q_positions

    def _selected_inputs(self, inputs: Tensor) -> Tensor:
        _validate_plaintexts(inputs)
        if inputs.device != next(self.parameters()).device:
            raise ValueError("inputs and model must be on the same device")
        return inputs.index_select(1, self.indices)

    def _columns(self, selected: Tensor, absolute_indices: tuple[int, ...]) -> Tensor:
        positions = [self._input_positions[index] for index in absolute_indices]
        return selected[:, positions]

    def _plaintext_sign(self, selected: Tensor) -> Tensor:
        bit = self._columns(selected, (self.target_bit,))
        return 1.0 - 2.0 * bit


class TwoRoundTablePredictor(_TwoRoundPredictor):
    """Four trainable residual logits over public two-round features.

    For each target bit, public U=Y XOR F(X) gives A=U[j+1], B=U[j+8], and
    T=X[j] XOR U[j+2]. The output is (1-2T)*table_logits[2A+B]. The table is
    initialized at zero and learns only from caller-provided TRAIN labels.
    """

    table_logits: nn.Parameter

    def __init__(self, *, rounds: int = 2, target_bit: int = 0) -> None:
        super().__init__(rounds=rounds, target_bit=target_bit)
        self.table_logits = nn.Parameter(torch.zeros(4, dtype=torch.float32))

    def public_table_features(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return public table-cell indices and T bits, without labels or cipher calls."""
        selected = self._selected_inputs(inputs)
        high_terms: list[Tensor] = []
        residual_terms: list[Tensor] = []
        for q in self._q_positions:
            y_q = self._columns(selected, (32 + q,))
            x_terms = self._columns(
                selected, tuple((q + offset) % 32 for offset in (1, 8, 2))
            )
            x_1, x_8, x_2 = x_terms[:, 0:1], x_terms[:, 1:2], x_terms[:, 2:3]
            and_term = x_1 * x_8
            f_q = x_2 + and_term - 2.0 * x_2 * and_term
            u_q = y_q + f_q - 2.0 * y_q * f_q
            high_terms.append(u_q)
            if q == (self.target_bit + 2) % 32:
                residual_terms.append(u_q)

        if len(high_terms) != 3 or len(residual_terms) != 1:
            raise RuntimeError("public two-round features are incomplete")
        a = high_terms[0]
        b = high_terms[1]
        t = self._columns(selected, (self.target_bit,)) + residual_terms[0]
        t = t - 2.0 * self._columns(selected, (self.target_bit,)) * residual_terms[0]
        cell_indices = (2 * a + b).to(dtype=torch.long).squeeze(1)
        return cell_indices, t

    def forward(self, inputs: Tensor) -> Tensor:
        cell_indices, residual_bit = self.public_table_features(inputs)
        residual_logits = self.table_logits.index_select(0, cell_indices).unsqueeze(1)
        logits = (1.0 - 2.0 * residual_bit) * residual_logits
        if logits.shape != (inputs.shape[0], 1) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("model produced invalid logits")
        return logits


class TwoRoundLocalPredictor(_TwoRoundPredictor):
    """Three independent public 4→16→1 local MLPs and a 3→8→1 combiner."""

    def __init__(self, *, rounds: int = 2, target_bit: int = 0) -> None:
        super().__init__(rounds=rounds, target_bit=target_bit)
        self.local_modules = nn.ModuleList(
            nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))
            for _ in range(3)
        )
        self.combiner = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, inputs: Tensor) -> Tensor:
        selected = self._selected_inputs(inputs)
        local_outputs: list[Tensor] = []
        for q, module in zip(self._q_positions, self.local_modules, strict=True):
            absolute = (32 + q, (q + 1) % 32, (q + 8) % 32, (q + 2) % 32)
            centered = self._columns(selected, absolute).mul(2.0).sub(1.0)
            local_outputs.append(cast(Tensor, module(centered)))
        hidden_bits = torch.sigmoid(torch.cat(local_outputs, dim=1)).mul(2.0).sub(1.0)
        combined = self.combiner(hidden_bits)
        if not isinstance(combined, Tensor):
            raise RuntimeError("combiner did not return a tensor")
        logits = combined * self._plaintext_sign(selected)
        if logits.shape != (inputs.shape[0], 1) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("model produced invalid logits")
        return logits


class TwoRoundVectorPredictor(_TwoRoundPredictor):
    """Three independent public 4→16→4 local MLPs and a 12→8→1 combiner."""

    def __init__(self, *, rounds: int = 2, target_bit: int = 0) -> None:
        super().__init__(rounds=rounds, target_bit=target_bit)
        self.local_modules = nn.ModuleList(
            nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 4))
            for _ in range(3)
        )
        self.combiner = nn.Sequential(nn.Linear(12, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, inputs: Tensor) -> Tensor:
        selected = self._selected_inputs(inputs)
        local_outputs: list[Tensor] = []
        for q, module in zip(self._q_positions, self.local_modules, strict=True):
            absolute = (32 + q, (q + 1) % 32, (q + 8) % 32, (q + 2) % 32)
            centered = self._columns(selected, absolute).mul(2.0).sub(1.0)
            local_outputs.append(cast(Tensor, module(centered)))
        if any(output.shape != (inputs.shape[0], 4) for output in local_outputs):
            raise RuntimeError("local modules must produce [N,4] logits")
        hidden_bits = torch.sigmoid(torch.cat(local_outputs, dim=1)).mul(2.0).sub(1.0)
        combined = self.combiner(hidden_bits)
        if not isinstance(combined, Tensor):
            raise RuntimeError("combiner did not return a tensor")
        logits = combined * self._plaintext_sign(selected)
        if logits.shape != (inputs.shape[0], 1) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("model produced invalid logits")
        return logits

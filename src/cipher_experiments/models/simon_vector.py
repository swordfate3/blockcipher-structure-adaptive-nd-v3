"""Typed vector computation over the public SIMON bit0 target cone."""
from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..representations.simon_vector import simon_vector_plan
from .prediction import _validate_builtin_int, _validate_plaintexts


def bound_vector(values: Tensor) -> Tensor:
    """Project each latent vector onto the L2 ball of radius2.

    Identity for sign-encoded exact Boolean vectors (+/-1,0,...); not LayerNorm
    and not a probability interpretation. Only forward activations are bounded,
    not parameters, gradients or a guarantee of good optimization.
    """
    return cast(Tensor, values / (values.norm(dim=-1, keepdim=True) / 2.).clamp_min(1.))


class SymmetricVectorGate(nn.Module):
    """Shared linear and multiplicative paths; symmetric in the two parents."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)
        self.product = nn.Linear(width, width, bias=False)
        self.project = nn.Linear(width, width, bias=False)
        self.mixed = nn.Linear(width, width, bias=False)
        self.bias = nn.Parameter(torch.zeros(width))
        with torch.no_grad():
            for module, diagonal in ((self.linear, .5), (self.product, .1),
                                     (self.project, .5), (self.mixed, .1)):
                module.weight.normal_(0., .02 / math.sqrt(width))
                module.weight.add_(torch.eye(width) * diagonal)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        value = (self.linear(a + b) + self.product(a * b)
                 + self.mixed(self.project(a) * self.project(b)) + self.bias)
        return bound_vector(cast(Tensor, value))


class SimonVectorPredictor(nn.Module):
    """Plaintexts->[N,1], no true key/state/teachers and no global readout bypass.

    Rounds 5 and 6 select the compiled public plan; the default stays 6 so the
    six-round initialization order and RNG consumption are unchanged.
    """

    indices: Tensor
    output_index: Tensor
    checkpoint_config_sha256: str

    def __init__(self, hidden_width: int = 32, *, rounds: int = 6,
                 target_bit: int = 0) -> None:
        super().__init__()
        _validate_builtin_int(hidden_width, "hidden_width", 32, 32)
        _validate_builtin_int(rounds, "rounds", 5, 6)
        _validate_builtin_int(target_bit, "target_bit", 0, 0)
        self.hidden_width = hidden_width
        self.rounds, self.target_bit = rounds, target_bit
        self.feature_mode, self.prediction_mode = "full", "direct"
        self.checkpoint_config_sha256 = ""
        plan = simon_vector_plan(rounds)
        self.plan = plan
        self.register_buffer("indices", torch.tensor(plan.plaintext_indices), persistent=True)
        self.register_buffer("symbolic_coordinates", torch.tensor(plan.symbolic_coordinates),
                             persistent=True)
        self.register_buffer("output_index", torch.tensor([plan.output]), persistent=True)
        for i, group in enumerate(plan.groups):
            self.register_buffer(f"operands_{i}", torch.tensor(group.operands), persistent=True)
        self.and_gate = SymmetricVectorGate(hidden_width)
        self.xor_gate = SymmetricVectorGate(hidden_width)
        self.constants = nn.Parameter(
            torch.randn(len(plan.symbolic_coordinates), hidden_width) * .2)
        self.readout = nn.Linear(hidden_width, 1)

    def validate_wiring(self) -> None:
        plan = simon_vector_plan(self.rounds)
        expected = {"indices": torch.tensor(plan.plaintext_indices),
                    "symbolic_coordinates": torch.tensor(plan.symbolic_coordinates),
                    "output_index": torch.tensor([plan.output])}
        expected.update({f"operands_{i}": torch.tensor(g.operands)
                         for i, g in enumerate(plan.groups)})
        for name, canonical in expected.items():
            value = cast(Tensor, getattr(self, name))
            if (value.dtype != torch.long or value.shape != canonical.shape
                    or not torch.equal(value.detach().cpu(), canonical)):
                raise ValueError(f"noncanonical vector graph wiring: {name}")

    def forward(self, inputs: Tensor) -> Tensor:
        _validate_plaintexts(inputs)
        if inputs.device != self.constants.device:
            raise ValueError("inputs and model must be on the same device")
        signs = 1 - 2 * inputs.index_select(1, self.indices)
        plaintext = F.pad(signs.unsqueeze(-1), (0, self.hidden_width - 1))
        constants = bound_vector(self.constants).unsqueeze(0).expand(len(inputs), -1, -1)
        table = torch.cat((plaintext, constants), dim=1)
        for i, group in enumerate(self.plan.groups):
            operands = cast(Tensor, getattr(self, f"operands_{i}"))
            left = table.index_select(1, operands[:, 0])
            right = table.index_select(1, operands[:, 1])
            gate = self.and_gate if group.kind == "and" else self.xor_gate
            table = torch.cat((table, gate(left, right)), dim=1)
        output = self.readout(table.index_select(1, self.output_index).squeeze(1))
        if output.shape != (len(inputs), 1) or not bool(torch.isfinite(output).all()):
            raise ValueError("nonfinite or malformed vector graph output")
        return cast(Tensor, output)

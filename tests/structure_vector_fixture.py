"""Isolated artificial-circuit prototype; not a CLI model or cipher experiment."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Node:
    kind: str
    left: int = -1
    right: int = -1
    coordinate: int = -1


def artificial_graph() -> tuple[Node, ...]:
    """Fixed 12-input, two-hidden-constant reconvergent graph, not SIMON."""
    nodes = [Node("input", coordinate=i) for i in range(12)]
    nodes.extend([Node("constant", coordinate=i) for i in range(2)])

    def gate(kind: str, left: int, right: int) -> int:
        nodes.append(Node(kind, left, right))
        return len(nodes) - 1

    a, b, c, d = gate("xor", 0, 12), 1, 2, 3
    for j in range(8):
        branch = gate("and", a, b)
        mixed = gate("xor", d, branch)
        update = gate("xor", mixed, 4 + j)
        a, b, c, d = update, a, b, c
    gate("xor", a, 13)
    return tuple(nodes)


def all_inputs() -> Tensor:
    values = torch.arange(4096, dtype=torch.int64)
    return ((values[:, None] >> torch.arange(11, -1, -1)) & 1).float()


def artificial_labels(inputs: Tensor, constants: tuple[int, int] = (1, 0)) -> Tensor:
    """Independent integer recurrence, never model/graph evaluation."""
    x = inputs.to(torch.int64)
    a, b, c, d = x[:, 0] ^ constants[0], x[:, 1], x[:, 2], x[:, 3]
    for j in range(8):
        update = d ^ (a & b) ^ x[:, 4 + j]
        a, b, c, d = update, a, b, c
    return (a ^ constants[1]).float().unsqueeze(1)


class SymmetricVectorGate(nn.Module):
    """The proposed affine + Hadamard + projected-Hadamard unit, no activations."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)
        self.product = nn.Linear(width, width, bias=False)
        self.project = nn.Linear(width, width, bias=False)
        self.mixed = nn.Linear(width, width, bias=False)
        self.bias = nn.Parameter(torch.zeros(width))
        # A single predeclared near-averaging random initialization, not fitted
        # to any target truth table or constants. No exact gate initialization.
        with torch.no_grad():
            for module, diagonal in ((self.linear, .5), (self.product, .1),
                                     (self.project, .5), (self.mixed, .1)):
                module.weight.normal_(0., .02 / math.sqrt(width))
                module.weight.add_(torch.eye(width) * diagonal)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return cast(Tensor, self.linear(a + b) + self.product(a * b)
                    + self.mixed(self.project(a) * self.project(b)) + self.bias)


class VectorGraph(nn.Module):
    """Test-only graph: two shared gate types, no node embeddings/readout bypass."""

    def __init__(self, nodes: tuple[Node, ...], width: int = 32) -> None:
        super().__init__()
        self.nodes = nodes
        self.width = width
        self.and_gate = SymmetricVectorGate(width)
        self.xor_gate = SymmetricVectorGate(width)
        self.constants = nn.Parameter(torch.randn(2, width) * .2)
        self.readout = nn.Linear(width, 1)
        self.trace: list[Tensor] = []
        self.capture = False

    def forward(self, inputs: Tensor) -> Tensor:
        values: list[Tensor] = []
        for node in self.nodes:
            if node.kind == "input":
                value = F.pad(1 - 2 * inputs[:, node.coordinate:node.coordinate + 1],
                              (0, self.width - 1))
            elif node.kind == "constant":
                value = self.constants[node.coordinate].expand(len(inputs), -1)
            else:
                gate = self.and_gate if node.kind == "and" else self.xor_gate
                value = gate(values[node.left], values[node.right])
            if self.capture and value.requires_grad:
                value.retain_grad()
            values.append(value)
        self.trace = values if self.capture else []
        return cast(Tensor, self.readout(values[-1]))


class ScalarGraph(nn.Module):
    """Fixed soft Boolean gates, two unknown logits: a distinct strong prior."""

    def __init__(self, nodes: tuple[Node, ...]) -> None:
        super().__init__()
        self.nodes = nodes
        self.constants = nn.Parameter((2 * torch.randint(0, 2, (2,)).float() - 1) * 2)

    def forward(self, inputs: Tensor) -> Tensor:
        constants = torch.sigmoid(self.constants)
        values: list[Tensor] = []
        for node in self.nodes:
            if node.kind == "input":
                value = inputs[:, node.coordinate]
            elif node.kind == "constant":
                value = constants[node.coordinate].expand(len(inputs))
            else:
                a, b = values[node.left], values[node.right]
                value = a * b if node.kind == "and" else a + b - 2 * a * b
            values.append(value)
        probability = values[-1].clamp(1e-6, 1 - 1e-6).unsqueeze(1)
        return torch.log(probability) - torch.log1p(-probability)


def exact_witness(nodes: tuple[Node, ...], width: int = 32) -> VectorGraph:
    """Known artificial constants only; not an initialization for any fit."""
    model = VectorGraph(nodes, width).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.and_gate.linear.weight[0, 0] = .5
        model.and_gate.product.weight[0, 0] = -.5
        model.and_gate.bias[0] = .5
        model.xor_gate.product.weight[0, 0] = 1.
        model.constants[:, 0] = torch.tensor([-1., 1.], dtype=torch.float64)
        model.readout.weight[0, 0] = -4.
    return model


def depths(nodes: tuple[Node, ...]) -> list[int]:
    result: list[int] = []
    for node in nodes:
        result.append(0 if node.kind in ("input", "constant")
                      else 1 + max(result[node.left], result[node.right]))
    return result

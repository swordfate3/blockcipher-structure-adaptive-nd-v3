"""Levelized public six-round bit0 cone; no secret values or training state."""
from __future__ import annotations

from dataclasses import dataclass

from .simon_cone import simon_cone


@dataclass(frozen=True)
class VectorGateGroup:
    kind: str
    depth: int
    operands: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SimonVectorPlan:
    plaintext_indices: tuple[int, ...]
    symbolic_coordinates: tuple[tuple[int, int], ...]
    groups: tuple[VectorGateGroup, ...]
    output: int
    node_count: int
    depth: int


def simon_vector_plan() -> SimonVectorPlan:
    """Reorder the canonical SIMON6/bit0 DAG for parallel same-depth gates.

    Operand indices refer to an append-only vector table. Groups of the same
    depth read only earlier depths; shared ancestors/constant coordinates are
    retained exactly once. This compiles routing, never evaluates cipher bits.
    """
    cone = simon_cone(6, 0)
    depths: list[int] = []
    for node in cone.nodes:
        depths.append(0 if not node.args else 1 + max(depths[a] for a in node.args))
    plaintext = tuple(n.coordinate[0] for n in cone.nodes if n.kind == "plaintext")
    keys = tuple((n.coordinate[0], n.coordinate[1]) for n in cone.nodes
                 if n.kind == "round_key")
    # Place plaintext leaves first and symbolic leaves second, irrespective of
    # where the recursive cone builder first encountered each terminal.
    mapping: dict[int, int] = {}
    for kind in ("plaintext", "round_key"):
        for i, node in enumerate(cone.nodes):
            if node.kind == kind:
                mapping[i] = len(mapping)
    groups: list[VectorGateGroup] = []
    for depth in range(1, max(depths) + 1):
        for kind in ("and", "xor"):
            ids = [i for i, n in enumerate(cone.nodes) if n.kind == kind and depths[i] == depth]
            if not ids:
                continue
            pairs = tuple((mapping[cone.nodes[i].args[0]], mapping[cone.nodes[i].args[1]])
                          for i in ids)
            groups.append(VectorGateGroup(kind, depth, pairs))
            for i in ids:
                mapping[i] = len(mapping)
    return SimonVectorPlan(plaintext, keys, tuple(groups), mapping[cone.output],
                           len(mapping), max(depths))

"""Levelized public SIMON bit0 cones for five or six rounds; no secret values."""
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


def simon_vector_plan(rounds: int = 6) -> SimonVectorPlan:
    """Reorder the canonical SIMON bit0 DAG (rounds 5 or 6) into same-depth gates.

    Only rounds 5 and 6 are compiled; other counts and bool values are rejected.
    Operand indices refer to an append-only vector table. Groups of the same
    depth read only earlier depths; shared ancestors/constant coordinates are
    retained exactly once. This compiles routing, never evaluates cipher bits.
    """
    if type(rounds) is not int:
        raise TypeError("rounds must be a built-in int")
    if rounds not in (5, 6):
        raise ValueError("rounds must be 5 or 6")
    cone = simon_cone(rounds, 0)
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

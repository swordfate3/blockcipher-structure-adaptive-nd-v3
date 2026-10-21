"""Small symbolic SIMON64 output-bit cones for public wiring analysis.

Round-key nodes are symbolic placeholders: this module never samples or derives
key values and is not a key-recovery or cipher-evaluation interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

# Symbolic cones are pure structure: the analysis engines stay cheap well past the
# cipher's own 44 rounds, so this bound is about tractable analysis, not security.
_MAX_ROUNDS = 16


@dataclass(frozen=True)
class ConeNode:
    """One immutable terminal or binary gate in a topologically ordered cone."""

    kind: str
    args: tuple[int, ...] = ()
    coordinate: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not str:
            raise TypeError("kind must be a built-in str")
        if type(self.args) is not tuple or any(type(arg) is not int for arg in self.args):
            raise TypeError("args must be a tuple of built-in ints")
        if type(self.coordinate) is not tuple or any(
            type(value) is not int for value in self.coordinate
        ):
            raise TypeError("coordinate must be a tuple of built-in ints")


@dataclass(frozen=True)
class SimonCone:
    """A validated, immutable forward Boolean DAG for one SIMON output bit."""

    nodes: tuple[ConeNode, ...]
    output: int

    def __post_init__(self) -> None:
        self.validate()

    @property
    def plaintext_indices(self) -> tuple[int, ...]:
        """Sorted MSB-first plaintext coordinates present in the cone."""
        return tuple(
            sorted(
                node.coordinate[0]
                for node in self.nodes
                if node.kind == "plaintext"
            )
        )

    @property
    def round_key_coordinates(self) -> tuple[tuple[int, int], ...]:
        """Sorted symbolic (zero-based round, MSB-first bit) key placeholders."""
        return tuple(
            sorted(
                (node.coordinate[0], node.coordinate[1])
                for node in self.nodes
                if node.kind == "round_key"
            )
        )

    @property
    def gate_counts(self) -> dict[str, int]:
        """Counts of unique gates in this DAG, keyed by ``AND`` and ``XOR``."""
        return {
            "AND": sum(node.kind == "and" for node in self.nodes),
            "XOR": sum(node.kind == "xor" for node in self.nodes),
        }

    def validate(self) -> None:
        """Reject malformed, non-forward, out-of-range, or unreachable nodes."""
        if type(self.nodes) is not tuple:
            raise TypeError("nodes must be a tuple")
        if type(self.output) is not int:
            raise TypeError("output must be a built-in int")
        if not self.nodes or not 0 <= self.output < len(self.nodes):
            raise ValueError("output must index a node in the cone")

        for index, node in enumerate(self.nodes):
            if not isinstance(node, ConeNode):
                raise TypeError("nodes must contain ConeNode instances")
            if node.kind in ("plaintext", "round_key"):
                if node.args:
                    raise ValueError("terminal nodes must not have arguments")
                if node.kind == "plaintext":
                    if len(node.coordinate) != 1 or not 0 <= node.coordinate[0] < 64:
                        raise ValueError("plaintext coordinate must be in [0, 64)")
                elif (
                    len(node.coordinate) != 2
                    or not 0 <= node.coordinate[0] < _MAX_ROUNDS
                    or not 0 <= node.coordinate[1] < 32
                ):
                    raise ValueError("round-key coordinate must be in [0, 6) x [0, 32)")
            elif node.kind in ("and", "xor"):
                if node.coordinate:
                    raise ValueError("gate nodes must not have coordinates")
                if len(node.args) != 2:
                    raise ValueError("binary gates must have exactly two arguments")
                if any(not 0 <= arg < index for arg in node.args):
                    raise ValueError("gate arguments must reference prior nodes")
            else:
                raise ValueError(f"unsupported cone node kind: {node.kind!r}")

        reachable: set[int] = set()
        pending = [self.output]
        while pending:
            node_id = pending.pop()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            pending.extend(self.nodes[node_id].args)
        if len(reachable) != len(self.nodes):
            raise ValueError("cone must contain only nodes reachable from its output")


def simon_cone(rounds: int, target_bit: int) -> SimonCone:
    """Build a symbolic SIMON64/128 output cone for rounds 1 through 16.

    Bit coordinates use ciphertext/plaintext MSB order: the high state word is
    0..31 and the low state word is 32..63. Symbolic key leaves identify the
    round-key bit consumed by the recurrence; no key material is accepted.
    """
    if type(rounds) is not int:
        raise TypeError("rounds must be a built-in int")
    if not 1 <= rounds <= _MAX_ROUNDS:
        raise ValueError("rounds must be between 1 and 6")
    if type(target_bit) is not int:
        raise TypeError("target_bit must be a built-in int")
    if not 0 <= target_bit < 64:
        raise ValueError("target_bit must be between 0 and 63")

    nodes: list[ConeNode] = []

    @cache
    def plaintext_term(index: int) -> int:
        nodes.append(ConeNode("plaintext", coordinate=(index,)))
        return len(nodes) - 1

    @cache
    def key_symbol(round_index: int, bit_index: int) -> int:
        nodes.append(ConeNode("round_key", coordinate=(round_index, bit_index)))
        return len(nodes) - 1

    def gate(kind: str, left: int, right: int) -> int:
        nodes.append(ConeNode(kind, args=(left, right)))
        return len(nodes) - 1

    @cache
    def state_bit(round_index: int, half: int, bit_index: int) -> int:
        if round_index == 0:
            return plaintext_term(bit_index + (32 if half else 0))
        if half == 1:
            return state_bit(round_index - 1, 0, bit_index)

        low_word_bit = state_bit(round_index - 1, 1, bit_index)
        and_term = gate(
            "and",
            state_bit(round_index - 1, 0, (bit_index + 1) % 32),
            state_bit(round_index - 1, 0, (bit_index + 8) % 32),
        )
        first_xor = gate("xor", low_word_bit, and_term)
        second_xor = gate(
            "xor", first_xor, state_bit(round_index - 1, 0, (bit_index + 2) % 32)
        )
        return gate("xor", second_xor, key_symbol(round_index - 1, bit_index))

    output = state_bit(rounds, 0 if target_bit < 32 else 1, target_bit % 32)
    return SimonCone(tuple(nodes), output)

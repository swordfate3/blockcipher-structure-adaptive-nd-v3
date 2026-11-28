"""Public dependency graph for the third-round high-word MSB0 bit."""
from __future__ import annotations

from hashlib import sha256
from types import MappingProxyType

_Node = tuple[int, int]
_Edges = tuple[_Node, _Node, _Node, _Node]


def _parents(node: _Node) -> _Edges:
    rnd, bit = node
    # A left rotation reads the input at a positive offset in MSB0.
    # The preceding low word is a copy of the high word one round earlier.
    return (
        (rnd - 1, (bit + 1) % 16),
        (rnd - 1, (bit + 8) % 16),
        (rnd - 1, (bit + 2) % 16),
        (rnd - 2, bit) if rnd > 1 else (0, 16 + bit),
    )


def _graph() -> tuple[tuple[int, ...], tuple[tuple[tuple[_Node, _Edges], ...], ...]]:
    pending = {(3, 0)}
    levels = []
    for rnd in (3, 2, 1):
        level = tuple((node, _parents(node)) for node in sorted(pending) if node[0] == rnd)
        pending.difference_update(node for node, _ in level)
        pending.update(parent for _, parents in level for parent in parents)
        levels.append(level)
    return tuple(bit for _, bit in sorted(pending)), tuple(reversed(levels))


SUPPORT, LEVELS = _graph()


def _shuffle() -> dict[int, int]:
    binding: dict[int, int] = {}
    for word in (0, 1):
        ordered = sorted(
            (bit for bit in SUPPORT if bit // 16 == word),
            key=lambda bit: sha256(f'simon32-r3-msb0-bit0-control-v1/{bit}'.encode()).digest(),
        )
        # Each source leaf reads the next position in digest order, wrapping at the end.
        binding.update(zip(ordered, ordered[1:] + ordered[:1], strict=True))
    return binding


SHUFFLE = MappingProxyType(_shuffle())

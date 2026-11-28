"""Public dependency graph for the SIMON32 high-word MSB0 bit."""
from __future__ import annotations

from collections.abc import Mapping
from functools import cache
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


@cache
def graph(rounds: int) -> tuple[tuple[int, ...], tuple[tuple[tuple[_Node, _Edges], ...], ...]]:
    """Return the sorted input support and operation levels in evaluation order."""
    if type(rounds) is not int or not 3 <= rounds <= 8:
        raise ValueError('rounds must be an integer from 3 through 8')
    pending = {(rounds, 0)}
    levels = []
    for rnd in range(rounds, 0, -1):
        level = tuple((node, _parents(node)) for node in sorted(pending) if node[0] == rnd)
        pending.difference_update(node for node, _ in level)
        pending.update(parent for _, parents in level for parent in parents)
        levels.append(level)
    return tuple(bit for _, bit in sorted(pending)), tuple(reversed(levels))


@cache
def shuffled_binding(rounds: int) -> Mapping[int, int]:
    """Derange each word's input support with a round-specific digest order."""
    support, _ = graph(rounds)
    def digest(bit: int) -> bytes:
        return sha256(f'simon32-r{rounds}-msb0-bit0-control-v1/{bit}'.encode()).digest()

    binding: dict[int, int] = {}
    for word in (0, 1):
        ordered = sorted((bit for bit in support if bit // 16 == word), key=digest)
        # Each source leaf reads the next position in digest order, wrapping at the end.
        binding.update(zip(ordered, ordered[1:] + ordered[:1], strict=True))
    return MappingProxyType(binding)


SUPPORT, LEVELS = graph(3)
SHUFFLE = shuffled_binding(3)

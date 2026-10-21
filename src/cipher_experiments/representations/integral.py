"""Exact integral (cube) balance over the public SIMON output-bit cone.

For a cube ``S`` (those plaintext bits range over all ``2**|S|`` values while the
rest stay fixed), writing ``f`` in ANF the cube sum equals the *superpoly*

    sum_{cube} f = sum over monomials m that contain every variable of S
                   of (coefficient of m) * (product of m's variables outside S)

which is a polynomial in the fixed bits and the round keys. It vanishes for every
assignment exactly when no monomial of ``f`` contains all of ``S``.

This module therefore tracks whether a monomial with a given ``S``-projection can
exist at all: XOR unions the two projection sets and AND unions pairwise. That
over-approximates, so ``balanced=True`` is a proof (superpoly is identically zero)
while ``balanced=False`` only means "not proven here". Exact parity cancellation
would need the full ANF, which is what makes this cheap.

Nothing here needs a key: round-key leaves are constants, so balance is a
property of the Boolean function. This module never samples key material.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .simon_cone import SimonCone


@dataclass(frozen=True)
class CubeBalance:
    """Outcome of one exact balance question plus its cost."""

    balanced: bool
    active_bits: int
    support_size: int
    monomials_tracked: int


def _pairwise_union(left: set[int], right: set[int]) -> set[int]:
    """Pairwise unions: which S-projections the product can reach (over-approx.)."""
    return {a | b for a in left for b in right}


def cube_balance(cone: SimonCone, active: frozenset[int] | set[int]) -> CubeBalance:
    """Decide whether the cone's output bit is balanced over the given cube.

    ``active`` holds MSB-first plaintext bit coordinates. Balance means the XOR of
    the output bit over all ``2**|active|`` cube assignments is zero, for every
    fixed value of the remaining plaintext bits and every key.
    """
    if not isinstance(cone, SimonCone):
        raise TypeError("cone must be a SimonCone")
    positions = sorted(active)
    if any(type(position) is not int or not 0 <= position < 64 for position in positions):
        raise ValueError("active bits must be integer plaintext coordinates in [0, 64)")
    if len(set(positions)) != len(positions):
        raise ValueError("active bits must be distinct")
    index = {position: order for order, position in enumerate(positions)}

    values: list[set[int]] = []
    tracked = 0
    for node in cone.nodes:
        if node.kind == "plaintext":
            (coordinate,) = node.coordinate
            # An active bit is its own monomial; an inactive one is a constant.
            values.append({1 << index[coordinate]} if coordinate in index else {0})
        elif node.kind == "round_key":
            values.append({0})
        else:
            left, right = (values[arg] for arg in node.args)
            values.append(left | right if node.kind == "xor"
                          else _pairwise_union(left, right))
        tracked = max(tracked, len(values[-1]))

    full = (1 << len(positions)) - 1
    support = values[cone.output]
    # ``full in support`` means a monomial may contain the whole cube, so balance
    # is not proven; absence is a proof of balance for every key and every fixing.
    return CubeBalance(balanced=full not in support, active_bits=len(positions),
                       support_size=len(support), monomials_tracked=tracked)


def brute_force_cube_parity(cone: SimonCone, active: tuple[int, ...], *,
                            fixed: int, encrypt_bit: Callable[[int], int]) -> int:
    """Reference cube-sum parity by direct evaluation over all assignments.

    ``encrypt_bit(plaintext)`` must return the real cipher's output bit for the
    cone's configured rounds/target, so this is an independent oracle.
    """
    positions = tuple(active)
    if len(set(positions)) != len(positions):
        raise ValueError("active bits must be distinct")
    parity = 0
    for assignment in range(1 << len(positions)):
        plaintext = fixed
        for order, position in enumerate(positions):
            if (assignment >> order) & 1:
                plaintext |= 1 << (63 - position)
            else:
                plaintext &= ~(1 << (63 - position))
        parity ^= encrypt_bit(plaintext) & 1
    return parity

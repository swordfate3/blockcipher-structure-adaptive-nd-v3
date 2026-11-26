"""Public PRESENT P-layer metadata only (LSB0 source → destination)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PLayer:
    state_bits: int
    block_count: int
    block_bits: int
    bit_order: str
    source_to_destination: tuple[int, ...]


# The public PRESENT permutation is P(i) = 16i mod 63, with bit 63 fixed.
PRESENT_P_LAYER = PLayer(64, 16, 4, 'LSB0', tuple((16 * i) % 63 for i in range(63)) + (63,))

"""Public PRESENT P-layer coordinates; no block, round-state or key operations."""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from struct import pack

from cipher_lab.present_public import PRESENT_P_LAYER, PLayer

Table = tuple[tuple[int, ...], ...]
SENTINEL = 64
CENTER = 4
DEFAULT_SHA256 = '8e83f4f7e30d817d7f161c67bf85a8320477bd537097d8f241db3add56d38122'
STRUCTURE_SHA256 = 'e3d8578406b90440c74f938c2cb110e12160b8a8342fae18ecb2a5dea0d4f4ae'


def _p(lsb: int) -> int:
    return 63 if lsb == 63 else 16 * lsb % 63


Q = tuple(63 - _p(63 - i) for i in range(64)) + (SENTINEL,)


def _validate(table: Table, expected_sha: str) -> None:
    if (len(table) != 64 or any(len(row) != 9 or row[CENTER] != i
            or any(type(j) is not int or not 0 <= j <= SENTINEL for j in row)
            for i, row in enumerate(table))
            or sum(row.count(SENTINEL) for row in table) != 116
            or sha256(pack('<576q', *(j for row in table for j in row))).hexdigest()
            != expected_sha):
        raise ValueError('NEIGHBORHOOD_TABLE')


def remap(default: Table) -> Table:
    _validate(default, DEFAULT_SHA256)
    if sorted(Q) != list(range(65)) or Q[-1] != SENTINEL or Q[:4] != (0, 16, 32, 48):
        raise ValueError('PRESENT_COORDINATES')
    rows: list[tuple[int, ...]] = [()] * 64
    for i, row in enumerate(default):
        rows[Q[i]] = tuple(Q[j] for j in row)
    structure = tuple(rows)
    _validate(structure, STRUCTURE_SHA256)
    return structure


@dataclass(frozen=True)
class GatherRule:
    input_order: str = 'MSB0'
    block_offsets: tuple[int, ...] = (-1, 0, 1)
    bit_offsets: tuple[int, ...] = (-1, 0, 1)
    slot_order: str = 'block-then-bit'
    center_slot: int = CENTER
    padding: str = 'zero'
    sentinel: int = SENTINEL


NIBBLE_RULE = GatherRule()


def compile_card(layer: PLayer, rule: GatherRule) -> Table:
    """Generate the gather from explicit block/bit offsets and P-layer coordinates."""
    width, blocks, size = layer.state_bits, layer.block_count, layer.block_bits
    p = layer.source_to_destination
    if (any(type(n) is not int for n in (width, blocks, size))
            or (width, blocks, size, layer.bit_order) != (64, 16, 4, 'LSB0')
            or type(p) is not tuple or len(p) != width
            or any(type(j) is not int for j in p) or sorted(p) != list(range(width))):
        raise ValueError('PRESENT_COORDINATES')
    offsets = (rule.block_offsets, rule.bit_offsets)
    if ((rule.input_order, rule.slot_order, rule.padding) != ('MSB0', 'block-then-bit', 'zero')
            or type(rule.sentinel) is not int or rule.sentinel != width
            or type(rule.center_slot) is not int or rule.center_slot != CENTER
            or any(type(axis) is not tuple or len(axis) != 3 or axis[1] != 0
                   or any(type(n) is not int for n in axis) or sorted(axis) != [-1, 0, 1]
                   for axis in offsets)):
        raise ValueError('PRESENT_GATHER_RULE')

    # Convert the source and destination from LSB0 to the model's MSB0; padding
    # is not a state bit. Both the center row and every real gather index move.
    q = tuple(width - 1 - p[width - 1 - i] for i in range(width)) + (rule.sentinel,)
    rows: list[tuple[int, ...]] = [()] * width
    for i in range(width):
        block, bit = divmod(i, size)
        rows[q[i]] = tuple(
            q[b * size + j] if 0 <= b < blocks and 0 <= j < size else rule.sentinel
            for db in rule.block_offsets for dj in rule.bit_offsets
            for b, j in ((block + db, bit + dj),)
        )
    return tuple(rows)


def verified_card(layer: PLayer, rule: GatherRule, *, model_default: Table | None = None) -> Table:
    structure = compile_card(layer, rule)
    _validate(structure, STRUCTURE_SHA256)
    if model_default is None:
        # Static check has no tensor imports. Use the pinned base rule with
        # identity coordinates to feed the original remap and its base-table SHA.
        identity = replace(PRESENT_P_LAYER, source_to_destination=tuple(range(SENTINEL)))
        model_default = compile_card(identity, NIBBLE_RULE)
    if structure != remap(model_default):  # compare all 576 elements, not only the digest
        raise ValueError('NEIGHBORHOOD_TABLE')
    return structure

"""Public coordinates and fixed nibble gathers for the four-task recipe."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from struct import pack
from types import MappingProxyType
from typing import Any

from cipher_lab.present_public import PRESENT_P_LAYER, PLayer

from ..present_neighborhood.layout import (
    DEFAULT_SHA256,
    NIBBLE_RULE,
    STRUCTURE_SHA256,
    GatherRule,
    Table,
    verified_card,
)

LINEAR = tuple(range(16))
COLUMN = tuple(4 * row + col for col in range(4) for row in range(4))
COLUMN_SHA256 = '227afc280e23f090ee6cb1e4a95a83af527bc49f09d852ed783bb8536def2418'


@dataclass(frozen=True)
class StructureCard:
    layout: str
    block_order: tuple[int, ...] = LINEAR
    gather_rule: GatherRule = NIBBLE_RULE
    p_layer: PLayer | None = None

    def to_dict(self) -> dict[str, Any]:
        rule = asdict(self.gather_rule)
        rule.update(block_offsets=list(self.gather_rule.block_offsets),
                    bit_offsets=list(self.gather_rule.bit_offsets))
        value = dict(layout=self.layout, state_bits=64, block_bits=4,
                     block_order=list(self.block_order), gather_rule=rule)
        if self.p_layer is not None:
            layer = asdict(self.p_layer)
            layer['source_to_destination'] = list(self.p_layer.source_to_destination)
            value['p_layer'] = layer
        return value


CARDS: Mapping[str, StructureCard] = MappingProxyType({
    'SIMECK': StructureCard('linear-nibbles'),
    'PRESENT': StructureCard('present-p-layer', p_layer=PRESENT_P_LAYER),
    'SIMON': StructureCard('linear-nibbles'),
    'SKINNY': StructureCard('4x4-column-major', block_order=COLUMN),
})


def _compile(card: StructureCard) -> Table:
    rule = card.gather_rule
    # Require the exact values and types of this recipe's common gather rule.
    if (rule != NIBBLE_RULE or type(rule.sentinel) is not int
            or type(rule.center_slot) is not int
            or any(type(offset) is not int for axis in (rule.block_offsets, rule.bit_offsets)
                   for offset in axis)):
        raise ValueError('STRUCTURE_CARD_RULE')
    scan = card.block_order
    if (type(scan) is not tuple or any(type(block) is not int for block in scan)
            or sorted(scan) != list(LINEAR)):
        raise ValueError('STRUCTURE_CARD_ORDER')
    if card.p_layer is not None:
        if scan != LINEAR:
            raise ValueError('STRUCTURE_CARD_ORDER')
        return verified_card(card.p_layer, rule)

    positions = {block: index for index, block in enumerate(scan)}
    rows = []
    for center in range(64):
        block, bit = divmod(center, 4)
        neighbors = []
        for block_delta in rule.block_offsets:
            position = positions[block] + block_delta
            for bit_delta in rule.bit_offsets:
                within = bit + bit_delta
                index = 4 * scan[position] + within if (
                    0 <= position < 16 and 0 <= within < 4) else rule.sentinel
                neighbors.append(index)
        rows.append(tuple(neighbors))
    return tuple(rows)


def verified_tables(cards: Mapping[str, StructureCard] = CARDS) -> dict[str, Table]:
    hashes = dict(SIMECK=DEFAULT_SHA256, PRESENT=STRUCTURE_SHA256,
                  SIMON=DEFAULT_SHA256, SKINNY=COLUMN_SHA256)
    if set(cards) != set(CARDS):
        raise ValueError('STRUCTURE_CARD_CIPHERS')
    tables = {}
    for cipher, expected in hashes.items():
        card = cards[cipher]
        table = _compile(card)
        if card.layout != CARDS[cipher].layout or (
                sha256(pack('<576q', *(bit for row in table for bit in row))).hexdigest()
                != expected):
            raise ValueError('STRUCTURE_CARD_TABLE')
        tables[cipher] = table
    return tables

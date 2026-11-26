"""Public PRESENT P-layer coordinates; no block, round-state or key operations."""
from __future__ import annotations

from hashlib import sha256
from struct import pack

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

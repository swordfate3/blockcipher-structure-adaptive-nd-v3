"""Private cube sampling. Public output is one full 2080-bit row, never points.

RandomSource(size) returns exactly size fresh private bytes. Production wiring
must inject secrets.token_bytes (no recoverable seed); tests inject finite
fixtures. Key words are four uint32s, least-significant word first, decoded
from a big-endian 16-byte key. Background and null words are big-endian uint64.
No key/background/word is returned in counts or errors. Python objects cannot
provide cryptographic erasure; do not serialize this object's private state.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from itertools import islice
from typing import Protocol

from cipher_lab.config import Cell
from cipher_lab.execution import Kind, Meter, Stopped
from cipher_lab.integral.observables import aggregate_words

Oracle = Callable[[tuple[int, ...], tuple[int, ...], int], Sequence[int]]
Domain = tuple[str, str, int]


class RandomSource(Protocol):
    """Private exact-length byte supplier; exceptions are sanitized by data."""

    def __call__(self, size: int, /) -> bytes: ...


def _points(active: tuple[int, ...], background: int) -> Iterator[int]:
    """MSB0 active bits vary exhaustively; inactive background stays fixed."""
    masks = tuple(1 << (63 - bit) for bit in active)
    base = background
    for mask in masks:
        base &= ~mask
    for assignment in range(1 << len(masks)):
        point = base
        for index, mask in enumerate(masks):
            if assignment & (1 << index):
                point |= mask
        yield point


class CubeCollector:
    """Single-threaded private owner, with no I/O or scientific decisions.

    Domain reuse is rejected even after failure. Group attempts start after
    domain claim (before budget/random/oracle); completion means all chunks
    were validated/aggregated. Block attempts mean reserved oracle input
    blocks, not unique plaintexts across keys. Null units are reserved 64-bit
    output words. Failed chunks retain attempts, earn no completion credit.
    The supplied Meter is the sole block/word ledger, owned by this collector;
    callers must not reserve scientific units again in a runner.
    """

    def __init__(self, meter: Meter, oracle: Oracle, private_random: RandomSource,
                 chunk_points: int) -> None:
        if type(chunk_points) is not int or not 1 <= chunk_points <= 65536:
            raise ValueError("chunk_points must be in 1..65536")
        self._meter = meter
        self._oracle = oracle
        self._random = private_random
        self._chunk = chunk_points
        self._domains: set[tuple[int, tuple[int, ...], str, str, int]] = set()
        self._keys: set[bytes] = set()
        self._groups = {
            "attempted_real_groups": 0, "completed_real_groups": 0,
            "attempted_null_groups": 0, "completed_null_groups": 0,
        }

    def _bytes(self, size: int) -> bytes:
        try:
            value = self._random(size)
            if type(value) is not bytes or len(value) != size:
                raise ValueError
            return value
        except Exception:
            raise Stopped("RANDOM_UNAVAILABLE") from None

    def _key(self) -> tuple[int, ...]:
        # Bounded collision rejection, including adversarial/broken sources.
        for _ in range(128):
            self._meter.check()
            raw = self._bytes(16)
            if raw not in self._keys:
                self._keys.add(raw)
                key = int.from_bytes(raw, "big")
                return tuple((key >> shift) & 0xFFFFFFFF for shift in (0, 32, 64, 96))
        raise Stopped("KEY_REUSE_LIMIT")

    def _encrypt(self, points: tuple[int, ...], key: tuple[int, ...],
                 rounds: int) -> tuple[int, ...]:
        self._meter.check()
        try:
            response = self._oracle(points, key, rounds)
        except Exception:
            raise Stopped("ORACLE_FAILED") from None
        try:
            # Bound conversion to requested count: never drain an unbounded iterator.
            if not isinstance(response, Sequence) or len(response) != len(points):
                raise ValueError
            words = tuple(islice(iter(response), len(points) + 1))
            if len(words) != len(points) or any(
                type(word) is not int or not 0 <= word < (1 << 64) for word in words
            ):
                raise ValueError
            return words
        except Exception:
            raise Stopped("ORACLE_RESPONSE") from None

    def row(self, cell: Cell, domain: Domain) -> int:
        if not isinstance(cell, Cell):
            raise ValueError("expected Cell")
        if type(domain) is not tuple or len(domain) != 3:
            raise ValueError("invalid group domain")
        side, phase, index = domain
        if (side not in ("real", "null") or phase not in ("discovery", "screen", "confirm")
                or type(index) is not int or index < 0):
            raise ValueError("invalid group domain")
        active = tuple(sorted(cell.active))
        identity = (cell.rounds, active, side, phase, index)
        if identity in self._domains:
            raise Stopped("DOMAIN_REUSED")
        self._domains.add(identity)
        self._groups[f"attempted_{side}_groups"] += 1
        kind: Kind = "real_blocks" if side == "real" else "null_words"
        size = 1 << len(active)
        result = 0
        key: tuple[int, ...] = ()
        points: Iterator[int] = iter(())
        for offset in range(0, size, self._chunk):
            amount = min(self._chunk, size - offset)
            # Reserve BEFORE ANY private random or cipher access for this chunk.
            self._meter.reserve(kind, amount)
            if side == "real":
                if offset == 0:
                    key = self._key()
                    background = int.from_bytes(self._bytes(8), "big")
                    points = _points(active, background)
                batch = tuple(islice(points, amount))
                words = self._encrypt(batch, key, cell.rounds)
            else:
                # A random function at distinct cube points has independent uint64 outputs.
                # It does NOT sample a uniform observation vector or generate a real key.
                words = tuple(int.from_bytes(self._bytes(8), "big") for _ in range(amount))
            result ^= aggregate_words(words)
            self._meter.complete(kind, amount)
            self._meter.check()
        self._groups[f"completed_{side}_groups"] += 1
        return result

    def counts(self) -> dict[str, int]:
        """Public cumulative blocks/words (Meter) and group attempts/completions."""
        snapshot = self._meter.snapshot()
        result = self._groups.copy()
        for name in ("attempted_real_blocks", "completed_real_blocks",
                     "attempted_null_words", "completed_null_words"):
            value = snapshot[name]
            assert type(value) is int
            result[name] = value
        return result

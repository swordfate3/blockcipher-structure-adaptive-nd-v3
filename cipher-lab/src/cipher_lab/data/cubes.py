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
from typing import TYPE_CHECKING, Protocol

from cipher_lab.config import Cell
from cipher_lab.execution import Kind, Meter, Stopped
from cipher_lab.integral.observables import aggregate_batch, aggregate_words

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

BatchOracle = Callable[["NDArray[np.uint64]", tuple[int, ...], int], "NDArray[np.uint64]"]
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


def _point_batches(active: tuple[int, ...], background: int,
                   chunk_points: int) -> Iterator[NDArray[np.uint64]]:
    """Vector enumeration in scalar _points order, with O(chunk) storage.

    Caller supplies canonical sorted active coordinates; each assignment bit
    controls the corresponding MSB0 mask, clearing active background first.
    Never construct a points-by-active-bits matrix.
    """
    import numpy as np

    masks = tuple(1 << (63 - bit) for bit in active)
    base = background
    for mask in masks:
        base &= ~mask
    for offset in range(0, 1 << len(active), chunk_points):
        assignments = np.arange(offset, min(offset + chunk_points, 1 << len(active)),
                                dtype=np.uint64)
        points = np.full(len(assignments), base, dtype=np.uint64)
        for index, mask in enumerate(masks):
            points |= ((assignments >> np.uint64(index)) & np.uint64(1)) * np.uint64(mask)
        yield points


class CubeCollector:
    """Single-threaded private owner, with no I/O or scientific decisions.

    Domain reuse is rejected even after failure. Group attempts start after
    domain claim (before budget/random/oracle); completion means all chunks
    were validated/aggregated. Block attempts mean reserved oracle input
    blocks, not unique plaintexts across keys. Null units are reserved 64-bit
    output words. Failed chunks retain attempts, earn no completion credit.
    The supplied Meter is the sole block/word ledger, owned by this collector;
    callers must not reserve scientific units again in a runner.

    batch_oracle explicitly selects the NumPy path; oracle may then be None.
    Without it the original four-position scalar engineering mode remains.
    null_random optionally separates byte-size bulk null requests from private
    key/background requests (e.g. demo counters); production defaults to the
    same exact-size entropy supplier. Both suppliers are private and sanitized.
    """

    def __init__(self, meter: Meter, oracle: Oracle | None, private_random: RandomSource,
                 chunk_points: int, *, batch_oracle: BatchOracle | None = None,
                 null_random: RandomSource | None = None) -> None:
        if type(chunk_points) is not int or not 1 <= chunk_points <= 65536:
            raise ValueError("chunk_points must be in 1..65536")
        if oracle is None and batch_oracle is None:
            raise ValueError("an explicit scalar or batch oracle is required")
        self._meter = meter
        self._oracle = oracle
        self._random = private_random
        self._null_random = private_random if null_random is None else null_random
        self._batch_oracle = batch_oracle
        self._chunk = chunk_points
        self._domains: set[tuple[int, tuple[int, ...], str, str, int]] = set()
        self._keys: set[bytes] = set()
        self._groups = {
            "attempted_real_groups": 0, "completed_real_groups": 0,
            "attempted_null_groups": 0, "completed_null_groups": 0,
        }

    def _bytes(self, size: int, *, null: bool = False) -> bytes:
        try:
            source = self._null_random if null else self._random
            value = source(size)
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
        assert self._oracle is not None
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

    def _encrypt_batch(self, points: NDArray[np.uint64], key: tuple[int, ...],
                       rounds: int) -> NDArray[np.uint64]:
        import numpy as np

        # Retain the immediate pre-oracle resource check after private preparation.
        self._meter.check()
        assert self._batch_oracle is not None
        requested_shape = points.shape
        try:
            response = self._batch_oracle(points, key, rounds)
        except Exception:
            raise Stopped("ORACLE_FAILED") from None
        try:
            # Subclasses may override methods later used outside this private boundary.
            if (type(response) is not np.ndarray or response.dtype != np.dtype(np.uint64)
                    or response.ndim != 1 or response.shape != requested_shape
                    or not response.flags.c_contiguous):
                raise ValueError
            return response
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
        batches: Iterator[NDArray[np.uint64]] = iter(())
        for offset in range(0, size, self._chunk):
            amount = min(self._chunk, size - offset)
            # Reserve BEFORE ANY private random or cipher access for this chunk.
            self._meter.reserve(kind, amount)
            if side == "real":
                if offset == 0:
                    key = self._key()
                    background = int.from_bytes(self._bytes(8), "big")
                    if self._batch_oracle is None:
                        points = _points(active, background)
                    else:
                        batches = _point_batches(active, background, self._chunk)
            if self._batch_oracle is not None:
                if side == "real":
                    batch_words = self._encrypt_batch(next(batches), key, cell.rounds)
                else:
                    import numpy as np

                    # One bulk read, big-endian decode, independent native writable copy.
                    raw = self._bytes(8 * amount, null=True)
                    batch_words = np.frombuffer(raw, dtype=">u8").astype(np.uint64, copy=True)
                result ^= aggregate_batch(batch_words, check=self._meter.check)
            else:
                if side == "real":
                    batch = tuple(islice(points, amount))
                    words = self._encrypt(batch, key, cell.rounds)
                else:
                    # Independent output per point, not a uniform observation vector.
                    words = tuple(int.from_bytes(self._bytes(8, null=True), "big")
                                  for _ in range(amount))
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

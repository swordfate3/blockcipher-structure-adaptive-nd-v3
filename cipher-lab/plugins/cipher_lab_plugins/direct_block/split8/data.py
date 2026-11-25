"""One private data-provider closure; shared, aligned public-input rows and high8 labels."""
from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import cast

import numpy as np
from numpy.typing import NDArray

from cipher_lab.execution import Meter, Stopped

from . import config

Words = NDArray[np.uint32]
Bytes = NDArray[np.uint8]
Floats = NDArray[np.float32]
Key = tuple[int, int, int, int]
BatchOracle = Callable[[Words, Key, int], Words]
MAX_BATCH = 65536
_INPUT_SHIFTS = np.arange(31, -1, -1, dtype=np.intp).astype(np.uint32)
_HIGH_SHIFTS = np.arange(7, -1, -1, dtype=np.intp).astype(np.uint8)


def sample_points(rows: int, meter: Meter) -> Words:
    """A seeded public draw from the entire uint32 population, without replacement."""
    meter.check()
    points = np.asarray(random.Random(config.DATA_SEED).sample(range(2**32), rows), dtype=np.uint32)
    meter.check()
    return points


def _words(value: object, rows: int) -> bool:
    return (type(value) is np.ndarray and value.dtype == np.dtype(np.uint32)
            and value.shape == (rows,) and value.flags.c_contiguous)


def _bytes(value: object, rows: int) -> bool:
    return type(value) is np.ndarray and value.dtype == np.uint8 and value.shape == (rows,)


@dataclass(frozen=True, slots=True)
class Panels:
    points: Callable[[config.Split], Words]
    label: Callable[[config.Split, int], Bytes]


def _draw(draw: Callable[[int], int]) -> Key:
    try:
        k = draw(64)
        if type(k) is not int or not 0 <= k < 2**64:
            raise ValueError
        return (k & 0xffff, (k >> 16) & 0xffff, (k >> 32) & 0xffff, k >> 48)
    except Exception:
        raise Stopped('RANDOM_UNAVAILABLE') from None


def make_panels(meter: Meter, draw: Callable[[int], int], encrypt: BatchOracle, *,
                sample: Callable[[int, Meter], Words] = sample_points) -> Panels:
    """Draw exactly once after validating the shared plaintext domain. No label retries."""
    sizes = config.SIZES
    meter.check()
    sampled = sample(sizes.total, meter)
    if not _words(sampled, sizes.total) or len(np.unique(sampled)) != sizes.total:
        raise ValueError('invalid or repeated plaintext')
    shared = sampled.copy()
    shared.flags.writeable = False
    by_split = {s: shared[at] for s, at in sizes.slices().items()}
    attempted: set[tuple[config.Split, int]] = set()
    meter.check()
    key = _draw(draw)

    def label(split: config.Split, rounds: int) -> Bytes:
        if type(rounds) is not int or rounds not in config.ROUNDS or split not in by_split:
            raise ValueError('invalid label panel')
        identity = (split, rounds)
        if identity in attempted:
            raise Stopped('DOMAIN_REUSED')
        attempted.add(identity)
        rows = by_split[split]
        high8 = np.empty(len(rows), dtype=np.uint8)
        for start in range(0, len(rows), MAX_BATCH):
            section = slice(start, start + MAX_BATCH)
            chunk = rows[section]
            meter.reserve('real_blocks', len(chunk))
            try:
                result = encrypt(chunk, key, rounds)
            except Exception:
                raise Stopped('ORACLE_FAILED') from None
            if not _words(result, len(chunk)) or np.shares_memory(chunk, result):
                raise Stopped('ORACLE_RESPONSE')
            high8[section] = (result >> np.uint32(24)).astype(np.uint8)
            meter.complete('real_blocks', len(chunk))
        meter.check()
        high8.flags.writeable = False
        return high8

    return Panels(by_split.__getitem__, label)


def production_panels(meter: Meter) -> Panels:
    import secrets

    from cipher_lab.ciphers import simon32

    return make_panels(meter, secrets.randbits, simon32.encrypt_batch)


def input_bits(points: Words) -> Floats:
    return cast(Floats, ((points[:, None] >> _INPUT_SHIFTS) & np.uint32(1)).astype(np.float32))


def high_bits(labels: Bytes) -> Bytes:
    return cast(Bytes, (labels[:, None] >> _HIGH_SHIFTS) & np.uint8(1))


class Panel:
    """One two-column view over shared packed rows; no per-net label copies on disk."""

    def __init__(self, points: Words, labels: Bytes, chunk: int) -> None:
        if type(chunk) is not int or chunk not in config.CHUNKS:
            raise ValueError('invalid split8 chunk')
        if not _words(points, len(points)) or not _bytes(labels, len(points)) or not len(points):
            raise ValueError('invalid panel shape')
        self._points, self._labels = points, labels
        self._shifts = _HIGH_SHIFTS[list(config.COLS[chunk])]

    def __len__(self) -> int:
        return len(self._points)

    def batch(self, indices: NDArray[np.intp]) -> tuple[Floats, Bytes]:
        x = input_bits(self._points[indices])
        y = (self._labels[indices, None] >> self._shifts) & np.uint8(1)
        return x, cast(Bytes, y)


def dataset(points: Words, labels: Mapping[int, Bytes]) -> bytes:
    if (not _words(points, len(points)) or not len(points)
            or set(labels) != set(config.ROUNDS)
            or any(not _bytes(labels[r], len(points)) for r in config.ROUNDS)):
        raise ValueError('incomplete dataset')
    with BytesIO() as archive:
        np.savez(archive, plaintext=points, r1_high8=labels[1], r2_high8=labels[2],
                 r3_high8=labels[3])
        return archive.getvalue()

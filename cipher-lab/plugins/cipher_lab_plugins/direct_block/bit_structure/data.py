"""Shared public plaintext panels and on-demand private single-bit labels."""
from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cipher_lab.execution import Meter, Stopped

from . import config

Words = NDArray[np.uint32]
Bytes = NDArray[np.uint8]
Floats = NDArray[np.float32]
Key = tuple[int, int, int, int]
BatchOracle = Callable[[Words, Key, int], Words]
_SHIFTS = np.arange(31, -1, -1, dtype=np.intp).astype(np.uint32)


def sample_points(rows: int, meter: Meter) -> Words:
    meter.check()
    points = np.asarray(random.Random(config.DATA_SEED).sample(range(2**32), rows), dtype=np.uint32)
    meter.check()
    return points


@dataclass(frozen=True, slots=True)
class Panels:
    points: Callable[[int, str], Words]
    label: Callable[[int, str], Bytes]


def _draw_keys(draw: Callable[[int], int]) -> tuple[Key, Key]:
    def one() -> Key:
        value = draw(64)
        if type(value) is not int or not 0 <= value < 2**64:
            raise ValueError
        return value & 0xffff, (value >> 16) & 0xffff, (value >> 32) & 0xffff, value >> 48

    try:
        first, second = one(), one()
    except Stopped:
        raise
    except Exception:
        raise Stopped('RANDOM_UNAVAILABLE') from None
    if first == second:
        raise Stopped('RANDOM_UNAVAILABLE')
    return first, second


def _words(value: object, rows: int) -> bool:
    return (type(value) is np.ndarray and value.dtype == np.uint32
            and value.shape == (rows,) and value.flags.c_contiguous)


def make_panels(meter: Meter, draw: Callable[[int], int], encrypt: BatchOracle, *,
                sample: Callable[[int, Meter], Words] = sample_points) -> Panels:
    sizes = {'train': config.TRAIN, 'dev': config.DEV, 'test': config.TEST}
    total = 2 * sum(sizes.values())
    meter.check()
    sampled = sample(total, meter)
    if not _words(sampled, total) or len(np.unique(sampled)) != total:
        raise ValueError('invalid or repeated plaintext')
    # Immutable backing prevents a caller from restoring write access through a view.
    shared = np.frombuffer(sampled.tobytes(), dtype=np.uint32)
    parts: dict[tuple[int, str], Words] = {}
    offset = 0
    for repeat in range(2):
        for split, rows in sizes.items():
            parts[repeat, split] = shared[offset:offset + rows]
            offset += rows
    meter.check()
    keys = _draw_keys(draw)
    attempted: set[tuple[int, str]] = set()

    def points(repeat: int, split: str) -> Words:
        if type(repeat) is not int or (repeat, split) not in parts:
            raise ValueError('invalid panel identity')
        return parts[repeat, split]

    def label(repeat: int, split: str) -> Bytes:
        rows = points(repeat, split)
        identity = repeat, split
        if identity in attempted:
            raise Stopped('DOMAIN_REUSED')
        attempted.add(identity)
        meter.reserve('real_blocks', len(rows))
        try:
            result = encrypt(rows, keys[repeat], config.ROUNDS)
        except Stopped:
            raise
        except Exception:
            raise Stopped('ORACLE_FAILED') from None
        if not _words(result, len(rows)) or np.shares_memory(rows, result):
            raise Stopped('ORACLE_RESPONSE')
        meter.complete('real_blocks', len(rows))
        meter.check()
        bits = (result >> np.uint32(31 - config.TARGET_MSB0)).astype(np.uint8) & np.uint8(1)
        return np.frombuffer(bits.tobytes(), dtype=np.uint8).reshape(-1, 1)

    return Panels(points, label)


def production_panels(meter: Meter) -> Panels:
    import secrets

    from cipher_lab.ciphers import simon32

    return make_panels(meter, secrets.randbits, simon32.encrypt_batch)


def _check_labels(labels: Bytes, rows: int) -> None:
    if not rows or labels.dtype != np.uint8 or labels.shape != (rows, 1) or (labels > 1).any():
        raise ValueError('invalid binary labels')


class Panel:
    def __init__(self, points: Words, labels: Bytes) -> None:
        if not _words(points, len(points)):
            raise ValueError('invalid panel points')
        _check_labels(labels, len(points))
        self._points, self._labels = points, labels

    def __len__(self) -> int:
        return len(self._points)

    def batch(self, at: NDArray[np.intp]) -> tuple[Floats, Bytes]:
        points = self._points[at]
        bits = ((points[:, None] >> _SHIFTS) & np.uint32(1)).astype(np.float32)
        return bits, self._labels[at]


def panel_digest(points: Words, labels: Bytes) -> str:
    """SHA-256 of packed little-endian words followed by the aligned label bytes."""
    if points.ndim != 1 or points.dtype.kind != 'u' or points.dtype.itemsize != 4:
        raise ValueError('invalid digest points')
    _check_labels(labels, len(points))
    digest = hashlib.sha256(points.astype('<u4', copy=False).tobytes())
    digest.update(labels.tobytes())
    return digest.hexdigest()

"""Owned full-word panels with a fixed key per repeat."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from random import SystemRandom

import numpy as np
from numpy.typing import NDArray

from cipher_lab.execution import Meter, Stopped

from ..bit_structure.data import BatchOracle, Panels, _draw_keys, _words
from ..bit_structure.data import Bytes as Bytes
from ..bit_structure.data import Floats as Floats
from ..bit_structure.data import Key as Key
from ..bit_structure.data import Words as Words
from . import config

_SHIFTS = np.arange(31, -1, -1, dtype=np.uint32)


def sample_points(rows: int, meter: Meter) -> Words:
    if type(rows) is not int or not 0 < rows <= 2**32:
        raise ValueError('invalid plaintext count')
    meter.check()
    result = np.asarray(SystemRandom().sample(range(2**32), rows), dtype=np.uint32)
    meter.check()
    return result


@dataclass(frozen=True, slots=True)
class CyclePanels:
    _round_panels: dict[int, Panels]

    def for_round(self, rounds: int) -> Panels:
        if type(rounds) is not int or rounds not in self._round_panels:
            raise ValueError('invalid round')
        return self._round_panels[rounds]


def make_panels(meter: Meter, draw: Callable[[int], int], encrypt: BatchOracle, *,
                sample: Callable[[int, Meter], Words] = sample_points) -> CyclePanels:
    splits = [('train', 0, config.TRAIN), ('dev', 0, config.DEV)]
    splits.extend(('test', depth, config.TEST) for depth in config.ROUNDS)
    total = config.REPEATS * sum(count for _, _, count in splits)
    meter.check()
    sampled = sample(total, meter)
    if not _words(sampled, total) or len(np.unique(sampled)) != total:
        raise ValueError('invalid or repeated plaintext')
    shared = np.frombuffer(sampled.tobytes(), dtype=np.uint32)
    parts: dict[tuple[int, str, int], Words] = {}
    offset = 0
    for repeat in range(config.REPEATS):
        for split, depth, count in splits:
            parts[repeat, split, depth] = shared[offset:offset + count]
            offset += count
    meter.check()
    keys = _draw_keys(draw)

    def round_panels(depth: int) -> Panels:
        attempted: set[tuple[int, str]] = set()

        def points(repeat: int, split: str) -> Words:
            if type(repeat) is not int or type(split) is not str:
                raise ValueError('invalid panel identity')
            identity = repeat, split, depth if split == 'test' else 0
            if identity not in parts:
                raise ValueError('invalid panel identity')
            return parts[identity]

        def label(repeat: int, split: str) -> Bytes:
            rows = points(repeat, split)
            identity = repeat, split
            if identity in attempted:
                raise Stopped('DOMAIN_REUSED')
            attempted.add(identity)
            meter.reserve('real_blocks', len(rows))
            try:
                ciphertext = encrypt(rows, keys[repeat], depth)
            except Stopped:
                raise
            except Exception:
                raise Stopped('ORACLE_FAILED') from None
            if not _words(ciphertext, len(rows)) or np.shares_memory(shared, ciphertext):
                raise Stopped('ORACLE_RESPONSE')
            meter.complete('real_blocks', len(rows))
            meter.check()
            bits = ((ciphertext[:, None] >> _SHIFTS) & np.uint32(1)).astype(np.uint8)
            return np.frombuffer(bits.tobytes(), dtype=np.uint8).reshape(-1, 32)

        return Panels(points, label)

    return CyclePanels({depth: round_panels(depth) for depth in config.ROUNDS})


def production_panels(meter: Meter) -> CyclePanels:
    import secrets

    from cipher_lab.ciphers import simon32

    return make_panels(meter, secrets.randbits, simon32.encrypt_batch)


def check_labels(labels: Bytes, rows: int) -> None:
    if not rows or labels.dtype != np.uint8 or labels.shape != (rows, 32) or (labels > 1).any():
        raise ValueError('invalid full-word labels')


class Panel:
    def __init__(self, points: Words, labels: Bytes) -> None:
        if not _words(points, len(points)):
            raise ValueError('invalid panel points')
        check_labels(labels, len(points))
        self._points, self._labels = points, labels

    def __len__(self) -> int:
        return len(self._points)

    def batch(self, at: NDArray[np.intp]) -> tuple[Floats, Bytes]:
        words = self._points[at, None]
        bits = ((words >> _SHIFTS) & np.uint32(1)).astype(np.float32)
        return bits, self._labels[at]


def panel_digest(points: Words, labels: Bytes) -> str:
    if points.ndim != 1 or points.dtype.kind != 'u' or points.dtype.itemsize != 4:
        raise ValueError('invalid digest points')
    check_labels(labels, len(points))
    digest = sha256(points.astype('<u4', copy=False).tobytes())
    digest.update(labels.tobytes())
    return digest.hexdigest()

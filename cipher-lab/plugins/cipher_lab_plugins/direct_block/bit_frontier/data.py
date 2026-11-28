"""Cycle-wide plaintext splits and on-demand private SIMON32 bit labels."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cipher_lab.execution import Meter, Stopped

from ..bit_structure import config
from ..bit_structure.data import (
    BatchOracle,
    Bytes,
    Panels,
    Words,
    _draw_keys,
    _words,
    sample_points,
)
from .config import ROUNDS


@dataclass(frozen=True, slots=True)
class CyclePanels:
    _round_panels: dict[int, Panels]

    def for_round(self, rounds: int) -> Panels:
        if type(rounds) is not int or rounds not in self._round_panels:
            raise ValueError('invalid round')
        return self._round_panels[rounds]


def make_panels(meter: Meter, draw: Callable[[int], int], encrypt: BatchOracle, *,
                sample: Callable[[int, Meter], Words] = sample_points) -> CyclePanels:
    rounds = ROUNDS
    splits = [
        (repeat, split, round_count, size)
        for repeat in range(config.REPEATS)
        for split, round_count, size in (
            ('train', None, config.TRAIN),
            ('dev', None, config.DEV),
            *(('test', r, config.TEST) for r in rounds),
        )
    ]
    total = sum(size for _, _, _, size in splits)
    meter.check()
    sampled = sample(total, meter)
    if not _words(sampled, total) or len(np.unique(sampled)) != total:
        raise ValueError('invalid or repeated plaintext')
    shared = np.frombuffer(sampled.tobytes(), dtype=np.uint32)
    parts: dict[tuple[int, str, int | None], Words] = {}
    offset = 0
    for repeat, split, round_count, size in splits:
        parts[repeat, split, round_count] = shared[offset:offset + size]
        offset += size
    meter.check()
    keys = _draw_keys(draw)

    def round_panels(round_count: int) -> Panels:
        attempted: set[tuple[int, str]] = set()

        def points(repeat: int, split: str) -> Words:
            identity = repeat, split, round_count if split == 'test' else None
            if type(repeat) is not int or type(split) is not str or identity not in parts:
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
                result = encrypt(rows, keys[repeat], round_count)
            except Stopped:
                raise
            except Exception:
                raise Stopped('ORACLE_FAILED') from None
            if not _words(result, len(rows)) or np.shares_memory(shared, result):
                raise Stopped('ORACLE_RESPONSE')
            meter.complete('real_blocks', len(rows))
            meter.check()
            bits = (result >> np.uint32(31)).astype(np.uint8)
            return np.frombuffer(bits.tobytes(), dtype=np.uint8).reshape(-1, 1)

        return Panels(points, label)

    return CyclePanels({r: round_panels(r) for r in rounds})


def production_panels(meter: Meter) -> CyclePanels:
    import secrets

    from cipher_lab.ciphers import simon32

    return make_panels(meter, secrets.randbits, simon32.encrypt_batch)

"""Fixed low-round matrix labels; the source contributes only plaintext and IDs."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from cipher_lab.execution import Meter, Stopped

from . import config
from .data import Panel, Words
from .reuse import SavedPanels

_CHUNK = 65536  # blocks, not pairs
Matrix = tuple[Panel, Panel, Panel]  # fixed order r1, r2, r3


class PlainSource(Protocol):
    def build_plaintext(self, split: Literal['train', 'dev']) -> tuple[Words, Words]: ...
    def label_test_plaintext(self) -> tuple[Words, Words]: ...


class LowRounds:
    def __init__(self, source: PlainSource, meter: Meter,
                 encrypt: Callable[[Words, config.LowRound], Words]) -> None:
        self._source, self._meter, self._encrypt = source, meter, encrypt
        self._next = 0
        self._failed = False

    def build(self, split: Literal['train', 'dev']) -> Matrix:
        if split not in ('train', 'dev'):
            raise ValueError('invalid split order; use label_test')
        return self._build(split)

    def label_test(self) -> Matrix:
        """The caller must first strictly restore all three persisted bests."""
        return self._build('test')

    def _build(self, split: Literal['train', 'dev', 'test']) -> Matrix:
        if (self._failed or self._next >= len(config.SPLITS)
                or split != config.SPLITS[self._next]):
            raise ValueError('invalid split order')
        self._failed = True
        self._meter.check()
        points, ids = (self._source.label_test_plaintext() if split == 'test'
                       else self._source.build_plaintext(split))
        panels = (self._label(points, ids, 1), self._label(points, ids, 2),
                  self._label(points, ids, 3))
        self._next += 1
        self._failed = False
        return panels

    def _label(self, points: Words, ids: Words, rounds: config.LowRound) -> Panel:
        plain = points.ravel()
        plain.flags.writeable = False
        labels = np.empty_like(plain)
        for at in range(0, len(plain), _CHUNK):
            batch = plain[at:at + _CHUNK]
            self._meter.reserve('real_blocks', len(batch))
            try:
                result = self._encrypt(batch, rounds)
            except Exception:
                raise Stopped('ORACLE_FAILED') from None
            if (type(result) is not np.ndarray or result.dtype != np.uint64
                    or result.shape != batch.shape or not result.flags.c_contiguous):
                raise Stopped('ORACLE_RESPONSE')
            labels[at:at + len(batch)] = result
            self._meter.complete('real_blocks', len(batch))
        self._meter.check()
        return Panel(points, labels.reshape(-1, 2), ids)


def production_low_rounds(source_root: Path, meter: Meter) -> LowRounds:
    """Bind one fresh private key to all labels, never expose it as provider data."""
    import secrets

    from cipher_lab.ciphers import simon

    source = SavedPanels(source_root, meter)
    meter.check()
    try:
        private = secrets.randbits(config.KEY_BITS)
        if type(private) is not int or not 0 <= private < 1 << config.KEY_BITS:
            raise ValueError
        words = tuple((private >> shift) & 0xffffffff for shift in range(0, config.KEY_BITS, 32))
    except Exception:
        raise Stopped('RANDOM_UNAVAILABLE') from None

    def encrypt(points: Words, rounds: config.LowRound) -> Words:
        return simon.encrypt_batch(points, words, rounds)

    return LowRounds(source, meter, encrypt)

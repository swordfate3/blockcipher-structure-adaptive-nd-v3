"""Compact cube pairs and one sequential, delayed-TEST label provider."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cipher_lab.execution import Meter, Stopped

from .config import (
    ACTIVE_BITS,
    BLOCK_BITS,
    DATA_SEED,
    FAMILIES,
    KEY_BITS,
    MASK,
    PAIRS_PER_FAMILY,
    ROUNDS,
    SPLITS,
    WIDTH,
)

Words = NDArray[np.uint64]
Oracle = Callable[[Words], Words]
Indices = NDArray[np.intp]
Floats = NDArray[np.float32]
Labels = NDArray[np.uint8]


def _bits(words: Words) -> Labels:
    return np.unpackbits(words.astype('>u8', copy=False).view(np.uint8),
                         bitorder='big').reshape(-1, WIDTH)


@dataclass(frozen=True, slots=True)
class Panel:
    plaintext: Words
    ciphertext: Words
    family_ids: Words

    def __post_init__(self) -> None:
        if (self.plaintext.ndim != 2 or self.plaintext.shape[1] != 2 or not len(self.plaintext)
                or self.plaintext.shape != self.ciphertext.shape
                or self.family_ids.shape != (len(self.plaintext),)
                or any(a.dtype != np.uint64 for a in
                       (self.plaintext, self.ciphertext, self.family_ids))):
            raise ValueError('invalid compact panel')
        for array in (self.plaintext, self.ciphertext, self.family_ids):
            array.flags.writeable = False

    def __len__(self) -> int:
        return len(self.plaintext)

    def batch(self, indices: Indices) -> tuple[Floats, Labels]:
        return _bits(self.plaintext[indices]).astype(np.float32), self.label_batch(indices)

    def label_batch(self, indices: Indices) -> Labels:
        return _bits(self.ciphertext[indices])

    def archive(self) -> bytes:
        """Compact portable NPZ with row-aligned identities. RunStore binds its SHA."""
        out = BytesIO()
        np.savez(out, plaintext=self.plaintext.astype('<u8', copy=False),
                 ciphertext=self.ciphertext.astype('<u8', copy=False),
                 family_ids=self.family_ids.astype('<u8', copy=False))
        return out.getvalue()


_OFFSETS = np.array([sum(((tag >> j) & 1) << bit for j, bit in enumerate(ACTIVE_BITS))
                     for tag in range(PAIRS_PER_FAMILY)], dtype=np.uint64)
_LABEL_CHUNK = 65536  # core batch interface bound, measured in blocks


class CubePanels:
    def __init__(self, oracle: Oracle, meter: Meter, *,
                 families: tuple[int, int, int] = FAMILIES,
                 draw: Callable[[], int] | None = None) -> None:
        if len(families) != 3 or any(type(n) is not int or n <= 0 for n in families):
            raise ValueError('invalid family counts')
        self._oracle, self._meter, self._families = oracle, meter, families
        self._rng = np.random.Generator(np.random.PCG64(DATA_SEED))
        self._draw = self._rng.bit_generator.random_raw if draw is None else draw
        self._seen: set[int] = set()
        self._next = 0
        self._failed = False

    def _unique(self) -> int:
        for _ in range(128):
            try:
                value = self._draw()
                if type(value) is not int or not 0 <= value < 1 << BLOCK_BITS:
                    raise ValueError
            except Exception:
                raise Stopped('RANDOM_UNAVAILABLE') from None
            base = value & ~MASK
            if base not in self._seen:
                self._seen.add(base)
                return base
        raise Stopped('DOMAIN_REUSED')

    def build(self, split: Literal['train', 'dev']) -> Panel:
        if split not in ('train', 'dev'):
            raise ValueError('invalid split order; use label_test')
        return self._build(split)

    def label_test(self) -> Panel:
        """Call only through the caller's successful dual persisted-reload gate."""
        return self._build('test')

    def _build(self, split: str) -> Panel:
        if self._failed or self._next >= len(SPLITS) or split != SPLITS[self._next]:
            raise ValueError('invalid split order')
        self._failed = True  # poison the stream if any actual build fails
        self._meter.check()
        n = self._families[self._next]
        bases = np.empty(n, dtype=np.uint64)
        for at in range(0, n, 4096):
            self._meter.check()
            end = min(n, at + 4096)
            bases[at:end] = np.fromiter((self._unique() for _ in range(at, end)),
                                       dtype=np.uint64, count=end - at)
        left = (bases[:, None] | _OFFSETS).ravel()
        points = np.column_stack((left, left ^ np.uint64(MASK)))
        order = self._rng.permutation(len(points))
        points = points[order]
        family_ids = np.repeat(bases, PAIRS_PER_FAMILY)[order]
        plain = points.ravel()
        plain.flags.writeable = False
        labels = np.empty_like(plain)
        for at in range(0, len(plain), _LABEL_CHUNK):
            batch = plain[at:at + _LABEL_CHUNK]
            self._meter.reserve('real_blocks', len(batch))
            try:
                result = self._oracle(batch)
            except Exception:
                raise Stopped('ORACLE_FAILED') from None
            if (type(result) is not np.ndarray or result.dtype != np.uint64
                    or result.shape != batch.shape or not result.flags.c_contiguous):
                raise Stopped('ORACLE_RESPONSE')
            labels[at:at + len(batch)] = result
            self._meter.complete('real_blocks', len(batch))
        self._meter.check()
        panel = Panel(points, labels.reshape(-1, 2), family_ids)
        self._next += 1
        self._failed = False
        return panel


def production_panels(meter: Meter) -> CubePanels:
    """Draw exactly one private key; only this provider closure retains the key."""
    import secrets

    from cipher_lab.ciphers import simon

    meter.check()
    try:
        private = secrets.randbits(KEY_BITS)
        if type(private) is not int or not 0 <= private < 1 << KEY_BITS:
            raise ValueError
        words = tuple((private >> shift) & 0xffffffff for shift in range(0, KEY_BITS, 32))
    except Exception:
        raise Stopped('RANDOM_UNAVAILABLE') from None

    def encrypt(points: Words) -> Words:
        return simon.encrypt_batch(points, words, ROUNDS)

    return CubePanels(encrypt, meter)

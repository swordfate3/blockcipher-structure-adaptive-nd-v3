"""Compact in-memory C7 panels; only the requested batch is expanded into bits."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from cipher_lab.execution import Meter, Stopped

from ..prefix_bridge.data import Draw, Encrypt, Floats, Indices, Labels, Split, Words, _canonical
from ..prefix_bridge.kat import Key
from . import config

type Sample = Callable[[np.random.Generator, int], Words]
PairKeys = NDArray[np.void]
_PAIR_KEY = np.dtype([('lo', np.uint64), ('hi', np.uint64)])
_CHUNK_PAIRS = 32_768
_SPLIT_IDS = {'train': 0, 'dev': 1, 'test': 2}


def _sample(rng: np.random.Generator, rows: int) -> Words:
    return rng.integers(0, 2**64, size=(rows, 2), dtype=np.uint64)


def _keys(pairs: Words, out: PairKeys) -> None:
    np.minimum(pairs[:, 0], pairs[:, 1], out=out['lo'])
    np.maximum(pairs[:, 0], pairs[:, 1], out=out['hi'])


def _require_disjoint(keys: PairKeys, earlier: list[PairKeys]) -> None:
    for sorted_keys in earlier:
        # Compare unsigned fields, not their native byte encoding or a digest.
        # Clipping makes the past-the-end candidate safe (and never equal).
        at = np.searchsorted(sorted_keys, keys)
        np.minimum(at, len(sorted_keys) - 1, out=at)
        if np.any(sorted_keys[at] == keys):
            raise ValueError('SPLIT_PAIR_OVERLAP')


@dataclass(frozen=True, slots=True)
class BatchPanel:
    _words: Words
    _labels: Labels
    digest: str

    def __len__(self) -> int:
        return len(self._labels)

    def batch(self, indices: Indices) -> tuple[Floats, Labels]:
        if (indices.ndim != 1 or indices.dtype != np.intp or not indices.size
                or indices.min() < 0 or indices.max() >= len(self)):
            raise ValueError('INVALID_PANEL_INDICES')
        x = np.zeros((len(indices), 1, 6, 64), dtype=np.float32)
        x[:, 0, 3:] = _canonical(self._words[indices])
        return x, self._labels[indices]


class Panels:
    def __init__(self, generate: Callable[[Split, int], BatchPanel]) -> None:
        self._splits = {'train': generate('train', config.TRAIN_ROWS),
                        'dev': generate('dev', config.DEV_ROWS)}
        self._generate_test: Callable[[Split, int], BatchPanel] | None = generate

    @property
    def hashes(self) -> dict[str, str]:
        return {split: panel.digest for split, panel in self._splits.items()}

    def train(self, slot: str) -> BatchPanel:
        return self.view('train', slot)

    def dev(self, slot: str) -> BatchPanel:
        return self.view('dev', slot)

    def test(self) -> None:
        generate = self._generate_test
        if generate is None:
            raise ValueError('TEST_ALREADY_REQUESTED')
        # Drop the key and the lookup tables even on failure; no second request.
        self._generate_test = None
        self._splits['test'] = generate('test', config.TEST_ROWS)

    def view(self, split: Split, slot: str) -> BatchPanel:
        if slot != config.PANEL:
            raise ValueError('INVALID_SLOT')
        if split == 'test' and split not in self._splits:
            raise ValueError('TEST_NOT_READY')
        return self._splits[split]


def _points(rng: np.random.Generator, n: int, labels: Labels, sample: Sample) -> Words:
    try:
        raw = sample(rng, n)
        if type(raw) is not np.ndarray or raw.dtype != np.uint64 or raw.shape != (n, 2):
            raise ValueError
        pairs = raw.copy(order='C')
    except Stopped:
        raise
    except Exception:
        raise ValueError('PANEL_SAMPLING_FAILED') from None
    pos = labels[:, 0] == 1
    pairs[pos, 1] = pairs[pos, 0] ^ np.uint64(config.DIFFERENCE)
    return pairs


def _cipher(pairs: Words, key: Key, encrypt: Encrypt, meter: Meter) -> Words:
    flat = pairs.reshape(-1)
    points = flat.copy()
    n = len(points)
    meter.reserve('real_blocks', n)
    try:
        out = encrypt(points, key, config.ROUNDS)
    except Stopped:
        raise
    except Exception:
        raise ValueError('PREFIX_ORACLE_FAILED') from None
    if (type(out) is not np.ndarray or out.dtype != np.uint64 or out.shape != (n, 1)
            or not out.flags.c_contiguous or np.shares_memory(points, out)
            or points.shape != (n,) or points.dtype != np.uint64
            or not points.flags.c_contiguous or not np.array_equal(points, flat)):
        raise ValueError('PREFIX_ORACLE_RESPONSE')
    return out.reshape(-1, 2)


def make_panels(meter: Meter, draw: Draw, encrypt: Encrypt, *, sample: Sample = _sample) -> Panels:
    meter.check()
    try:
        secret = draw(80)
        if type(secret) is not int or not 0 <= secret < 2**80:
            raise ValueError
        key: Key = (secret & 0xFFFFFFFF, (secret >> 32) & 0xFFFFFFFF, secret >> 64)
    except Stopped:
        raise
    except Exception:
        raise ValueError('PRIVATE_SOURCE_FAILED') from None
    del secret
    earlier_plain: list[PairKeys] = []
    earlier_c7: list[PairKeys] = []

    def generate(split: Split, rows: int) -> BatchPanel:
        meter.check()
        seed = [config.DATA_SEED, _SPLIT_IDS[split]]
        lrng, rng = (np.random.default_rng(np.random.SeedSequence([*seed, i])) for i in (0, 1))
        labels = (lrng.permutation(rows) < rows // 2).astype(np.uint8).reshape(rows, 1)
        words = np.empty((rows, 2), dtype=np.uint64)
        # A bijection preserves unordered-pair identity. Keep the exact P keys
        # to reject overlap before charging the chunk, and the C7 keys to also
        # enforce the public split invariant against malformed oracle output.
        key_rows = min(rows, _CHUNK_PAIRS) if split == 'test' else rows
        plain, c7 = np.empty(key_rows, _PAIR_KEY), np.empty(key_rows, _PAIR_KEY)
        digest = sha256(f'{config.METHOD}/{split}/{rows}/N,2/<u8/MSB0/N,1/u8/'.encode('ascii'))
        digest.update(memoryview(labels))
        for start in range(0, rows, _CHUNK_PAIRS):
            meter.check()
            end = min(start + _CHUNK_PAIRS, rows)
            at = slice(0, end-start) if split == 'test' else slice(start, end)
            pairs = _points(rng, end-start, labels[start:end], sample)
            _keys(pairs, plain[at])
            _require_disjoint(plain[at], earlier_plain)
            result = _cipher(pairs, key, encrypt, meter)
            _keys(result, c7[at])
            _require_disjoint(c7[at], earlier_c7)
            words[start:end] = result
            meter.complete('real_blocks', 2 * (end-start))
            digest.update(memoryview(words[start:end].astype('<u8', copy=False)))
        if split != 'test':
            for keys, earlier in ((plain, earlier_plain), (c7, earlier_c7)):
                meter.check()
                keys.sort(kind='quicksort')
                earlier.append(keys)
        meter.check()
        words.flags.writeable = labels.flags.writeable = False
        return BatchPanel(words, labels, digest.hexdigest())

    return Panels(generate)


def production_panels(meter: Meter) -> Panels:
    from secrets import randbits

    from cipher_lab.ciphers.present_prefix import encrypt_prefixes

    return make_panels(meter, randbits, encrypt_prefixes)

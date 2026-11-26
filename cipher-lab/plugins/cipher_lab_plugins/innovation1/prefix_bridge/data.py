"""One private draw; shared immutable in-memory PRESENT prefix panels."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cipher_lab.execution import Meter

from . import config
from .kat import Key

Words = NDArray[np.uint64]
Indices = NDArray[np.intp]
Floats = NDArray[np.float32]
Labels = NDArray[np.uint8]
Split = Literal['train', 'dev', 'test']
Draw = Callable[[int], int]
Encrypt = Callable[[Words, Key, tuple[int, ...]], Words]
Sample = Callable[[np.random.Generator, int], Words]
_SHIFTS = np.arange(63, -1, -1).astype(np.uint64)
_MAX_POINTS = 65536  # bound of the existing pure prefix API


def _sample(rng: np.random.Generator, rows: int) -> Words:
    return rng.integers(0, 2**64, size=(rows, config.PAIRS_PER_GROUP, 2), dtype=np.uint64)


def _canonical(words: Words) -> Floats:
    left, right = words[..., 0], words[..., 1]
    channels = np.stack((left, right, left ^ right), axis=-1)
    return ((channels[..., None] >> _SHIFTS) & 1).astype(np.float32)


@dataclass(frozen=True, slots=True)
class BatchPanel:
    # S2-only holds exactly a view of column 0, never the final view or round.
    _s2: Words | None
    _final: Words | None
    _labels: Labels

    def __len__(self) -> int:
        return len(self._labels)

    def batch(self, indices: Indices) -> tuple[Floats, Labels]:
        if (indices.ndim != 1 or indices.dtype != np.intp or not indices.size
                or indices.min() < 0 or indices.max() >= len(self)):
            raise ValueError('INVALID_PANEL_INDICES')
        x = np.zeros((len(indices), config.PAIRS_PER_GROUP, 6, 64), dtype=np.float32)
        if self._s2 is not None:
            x[:, :, :3] = _canonical(self._s2[indices])
        if self._final is not None:
            x[:, :, 3:] = _canonical(self._final[indices])
        return x, self._labels[indices]


@dataclass(frozen=True, slots=True)
class _Split:
    states: Words
    labels: Labels
    digest: str
    rounds: tuple[int, ...] = config.ROUNDS

    def view(self, slot: str) -> BatchPanel:
        r, mask = config.slot_input(slot)
        if self.rounds == (7,) and slot != 'r7-cipher':
            raise ValueError('INVALID_SLOT')
        if r is None:
            return BatchPanel(self.states[..., 0], None, self.labels)
        final = self.states[..., self.rounds.index(r)]
        return BatchPanel(self.states[..., 0] if mask == 63 else None, final, self.labels)


class Panels:
    def __init__(self, generate: Callable[[Split, int], _Split]) -> None:
        self._splits = {'train': generate('train', config.TRAIN_ROWS),
                        'dev': generate('dev', config.DEV_ROWS)}
        self._generate_test: Callable[[Split, int], _Split] | None = generate

    @property
    def hashes(self) -> dict[str, str]:
        return {name: panel.digest for name, panel in self._splits.items()}

    def train(self, slot: str) -> BatchPanel:
        return self.view('train', slot)

    def dev(self, slot: str) -> BatchPanel:
        return self.view('dev', slot)

    def test(self) -> None:
        generate = self._generate_test
        if generate is None:
            raise ValueError('TEST_ALREADY_REQUESTED')
        # Consume the closure even on failure. There is no retry and no live key
        # in Panels once the sole TEST request leaves this method.
        self._generate_test = None
        self._splits['test'] = generate('test', config.TEST_ROWS)

    def view(self, split: Split, slot: str) -> BatchPanel:
        if split == 'test' and split not in self._splits:
            raise ValueError('TEST_NOT_READY')
        return self._splits[split].view(slot)


def make_panels(meter: Meter, draw: Draw, encrypt_prefixes: Encrypt, *,
                sample: Sample = _sample, rounds: tuple[int, ...] = config.ROUNDS) -> Panels:
    if (type(rounds) is not tuple or any(type(r) is not int for r in rounds)
            or rounds not in (config.ROUNDS, (7,))):
        raise ValueError('INVALID_ROUNDS')
    meter.check()
    try:
        key_int = draw(80)
        if type(key_int) is not int or not 0 <= key_int < 2**80:
            raise ValueError
        key: Key = (key_int & 0xFFFFFFFF, (key_int >> 32) & 0xFFFFFFFF, key_int >> 64)
    except Exception:
        raise ValueError('PRIVATE_SOURCE_FAILED') from None
    del key_int
    rng = np.random.default_rng(config.DATA_SEED)
    earlier_pairs: set[bytes] = set()

    def generate(split: Split, rows: int) -> _Split:
        meter.check()
        labels = (rng.permutation(rows) < rows // 2).astype(np.uint8).reshape(rows, 1)
        shape = (rows, config.PAIRS_PER_GROUP, 2)
        try:
            sampled = sample(rng, rows)
            if (type(sampled) is not np.ndarray or sampled.dtype != np.uint64
                    or sampled.shape != shape):
                raise ValueError
            raw = sampled.copy(order='C')
        except Exception:
            raise ValueError('PANEL_SAMPLING_FAILED') from None
        del sampled
        positive = labels[:, 0] == 1
        raw[positive, :, 1] = raw[positive, :, 0] ^ np.uint64(config.DIFFERENCE)
        # Retain digests, not raw plains, across splits. A reversed full pair
        # is also overlap; a repeated single endpoint is not overlap.
        pairs = {sha256(pair).digest() for pair in
                 np.sort(raw.reshape(-1, 2), axis=1).astype('<u8', copy=False)}
        if not earlier_pairs.isdisjoint(pairs):
            raise ValueError('SPLIT_PAIR_OVERLAP')
        earlier_pairs.update(pairs)
        del pairs
        flat = raw.reshape(-1)
        states = np.empty((flat.size, len(rounds)), dtype=np.uint64)
        for start in range(0, len(flat), _MAX_POINTS):
            points = flat[start:start + _MAX_POINTS].copy()
            n = len(points)
            charged = n * len(rounds)
            meter.reserve('real_blocks', charged)
            try:
                result = encrypt_prefixes(points, key, rounds)
            except Exception:
                raise ValueError('PREFIX_ORACLE_FAILED') from None
            if (type(result) is not np.ndarray or result.dtype != np.uint64
                    or result.shape != (n, len(rounds)) or not result.flags.c_contiguous
                    or np.shares_memory(points, result) or points.shape != (n,)
                    or points.dtype != np.uint64
                    or not np.array_equal(points, flat[start:start + n])):
                raise ValueError('PREFIX_ORACLE_RESPONSE')
            states[start:start + n] = result
            meter.complete('real_blocks', charged)
        meter.check()
        domain = config.METHOD if rounds == config.ROUNDS else f'{config.METHOD}/r7-cipher-only'
        digest = sha256(f'{domain}/{split}/{rows}/'.encode('ascii'))
        digest.update(labels.tobytes())
        digest.update(states.astype('<u8', copy=False).tobytes())
        states.flags.writeable = labels.flags.writeable = False
        return _Split(states.reshape((*shape, len(rounds))), labels, digest.hexdigest(), rounds)

    return Panels(generate)


def production_panels(meter: Meter) -> Panels:
    from secrets import randbits

    from cipher_lab.ciphers.present_prefix import encrypt_prefixes

    return make_panels(meter, randbits, encrypt_prefixes)

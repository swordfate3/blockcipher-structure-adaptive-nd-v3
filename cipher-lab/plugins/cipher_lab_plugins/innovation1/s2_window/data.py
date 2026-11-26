"""In-memory ideal S2 selection and same-key, same-trajectory C7 panels."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from cipher_lab.execution import Meter

from ..prefix_bridge.data import BatchPanel, Draw, Encrypt, Sample, Split, Words
from ..prefix_bridge.kat import Key
from . import config

Inverse = Callable[[Words, Key], Words]
_MAX_POINTS = 65536


def _sample(rng: np.random.Generator, rows: int) -> Words:
    return rng.integers(0, 2**64, size=(rows, config.PAIRS_PER_GROUP, 2), dtype=np.uint64)


def _words(a: Words, shape: tuple[int, ...]) -> bool:
    return (type(a) is np.ndarray and a.dtype == np.dtype(np.uint64) and a.shape == shape
            and a.flags.c_contiguous)


def _inverse_points(meter: Meter, s2: Words, key: Key, inverse: Inverse, *,
                    expected_plain: int | None = None) -> Words:
    supplied = s2.copy()
    n = len(s2)
    meter.reserve('real_blocks', n)
    try:
        plain = inverse(supplied, key)
    except Exception:
        raise ValueError('INVERSE_ORACLE_FAILED') from None
    if (not _words(supplied, (n,)) or not _words(plain, (n,))
            or np.shares_memory(supplied, plain) or np.shares_memory(s2, plain)
            or not supplied.flags.writeable or not np.array_equal(supplied, s2)
            or expected_plain is not None and int(plain[0]) != expected_plain):
        raise ValueError('INVERSE_ORACLE_RESPONSE')
    meter.complete('real_blocks', n)
    return plain


def _full_forward(meter: Meter, plain: Words, s2: Words, key: Key, prefixes: Encrypt, *,
                  expected_c7: int | None = None) -> Words:
    n = len(s2)
    supplied = plain.copy()
    meter.reserve('real_blocks', 2 * n)
    try:
        states = prefixes(supplied, key, config.ROUNDS)
    except Exception:
        raise ValueError('PREFIX_ORACLE_FAILED') from None
    if (not _words(supplied, (n,)) or not _words(states, (n, 2))
            or np.shares_memory(supplied, states) or np.shares_memory(plain, states)
            or np.shares_memory(s2, states) or not supplied.flags.writeable
            or not np.array_equal(supplied, plain)
            or not np.array_equal(states[:, 0], s2)
            or expected_c7 is not None and int(states[0, 1]) != expected_c7):
        raise ValueError('PREFIX_ORACLE_RESPONSE')
    meter.complete('real_blocks', 2 * n)
    return states[:, 1]


@dataclass(frozen=True, slots=True)
class _Split:
    panel: BatchPanel
    digest: str


class Panels:
    def __init__(self, generate: Callable[[Split, int], _Split]) -> None:
        self._splits = {'train': generate('train', config.TRAIN_ROWS),
                        'dev': generate('dev', config.DEV_ROWS)}
        self._generate_test: Callable[[Split, int], _Split] | None = generate

    @property
    def hashes(self) -> dict[str, str]:
        return {name: value.digest for name, value in self._splits.items()}

    def view(self, split: Split) -> BatchPanel:
        if split == 'test' and split not in self._splits:
            raise ValueError('TEST_NOT_READY')
        return self._splits[split].panel

    def test(self) -> None:
        generate = self._generate_test
        if generate is None:
            raise ValueError('TEST_ALREADY_REQUESTED')
        # Consume even on failure; no retry and no retained key after request.
        self._generate_test = None
        self._splits['test'] = generate('test', config.TEST_ROWS)


def make_panels(meter: Meter, draw: Draw, inverse: Inverse, prefixes: Encrypt, *,
                sample: Sample = _sample) -> Panels:
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
            s2 = sampled.copy(order='C')
        except Exception:
            raise ValueError('PANEL_SAMPLING_FAILED') from None
        del sampled
        positive = labels[:, 0] == 1
        s2[positive, :, 1] = s2[positive, :, 0] ^ np.uint64(config.DIFFERENCE)
        # One bijection under this key: complete unordered pair equality at S2
        # coincides with pair equality at P or C7. Retain only hashes across splits.
        pairs = {sha256(pair).digest() for pair in
                 np.sort(s2.reshape(-1, 2), axis=1).astype('<u8', copy=False)}
        if not earlier_pairs.isdisjoint(pairs):
            raise ValueError('SPLIT_PAIR_OVERLAP')
        if split != 'test':
            earlier_pairs.update(pairs)
        del pairs
        flat = s2.reshape(-1)
        final = np.empty(shape, dtype=np.uint64)
        c7 = final.reshape(-1)
        for start in range(0, len(flat), _MAX_POINTS):
            chunk = flat[start:start + _MAX_POINTS]
            plain = _inverse_points(meter, chunk, key, inverse)
            c7[start:start + len(chunk)] = _full_forward(meter, plain, chunk, key, prefixes)
            del plain
        meter.check()
        digest = sha256(f'{config.METHOD}/{split}/{rows}/'.encode('ascii'))
        digest.update(labels.tobytes())
        digest.update(final.astype('<u8', copy=False).tobytes())
        final.flags.writeable = labels.flags.writeable = False
        return _Split(BatchPanel(None, final, labels), digest.hexdigest())

    return Panels(generate)


def production_panels(meter: Meter) -> Panels:
    from secrets import randbits

    from cipher_lab.ciphers.present_prefix import encrypt_prefixes
    from cipher_lab.ciphers.present_s2 import invert_second_output

    return make_panels(meter, randbits, invert_second_output, encrypt_prefixes)

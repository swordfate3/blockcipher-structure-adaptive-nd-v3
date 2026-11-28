"""Own synthetic plaintext, labels, and private keys for the two training arms."""
from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from cipher_lab.execution import Meter, Stopped

from ..mask_aux_single.data import (
    _PAIR_KEY,
    _SPLIT_IDS,
    BatchPanel,
    PairKeys,
    Sample,
    _keys,
    _points,
    _require_disjoint,
    _sample,
)
from ..prefix_bridge.data import Draw, Encrypt, Split, Words
from ..prefix_bridge.kat import Key
from . import config
from .cipher import MAX_PAIRS, encrypt_pairs

type DrawBytes = Callable[[int], bytes]
type EncryptPairs = Callable[[Words, NDArray[np.uint64], NDArray[np.uint16]], Words]


class Panels:
    def __init__(self, trains: dict[str, BatchPanel], dev: BatchPanel,
                 generate_test: Callable[[], BatchPanel]) -> None:
        self._trains = trains
        self._shared = {'dev': dev}
        self._pending = {'test': generate_test}

    @property
    def hashes(self) -> dict[str, str]:
        result = {f'train-{slot}': panel.digest for slot, panel in self._trains.items()}
        result.update((split, panel.digest) for split, panel in self._shared.items())
        return result

    def train(self, slot: str) -> BatchPanel:
        return self.view('train', slot)

    def dev(self, slot: str) -> BatchPanel:
        return self.view('dev', slot)

    def test(self) -> None:
        generate = self._pending.pop('test', None)
        if generate is None:
            raise ValueError('TEST_ALREADY_REQUESTED')
        self._shared['test'] = generate()

    def view(self, split: Split, slot: str) -> BatchPanel:
        if slot not in self._trains:
            raise ValueError('INVALID_SLOT')
        if split == 'train':
            return self._trains[slot]
        if split == 'test' and split not in self._shared:
            raise ValueError('TEST_NOT_READY')
        return self._shared[split]


def _call(meter: Meter, pairs: Words, oracle: Callable[[], Words],
          inputs: tuple[np.ndarray, ...], shape: tuple[int, int]) -> Words:
    before = tuple(value.copy() for value in inputs)
    meter.reserve('real_blocks', pairs.size)
    try:
        out = oracle()
    except Stopped:
        raise
    except Exception:
        raise ValueError('PREFIX_ORACLE_FAILED') from None
    if (type(out) is not np.ndarray or out.dtype != np.uint64 or out.shape != shape
            or not out.flags.c_contiguous or np.shares_memory(out, pairs)
            or any(np.shares_memory(out, value) or value.dtype != saved.dtype
                   or not value.flags.c_contiguous or not np.array_equal(value, saved)
                   for value, saved in zip(inputs, before, strict=True))):
        raise ValueError('PREFIX_ORACLE_RESPONSE')
    result = out.copy().reshape(-1, 2)
    meter.complete('real_blocks', pairs.size)
    return result


def make_panels(meter: Meter, draw: Draw, draw_bytes: DrawBytes, encrypt: Encrypt,
                encrypt_pairs: EncryptPairs, *, sample: Sample = _sample) -> Panels:
    meter.check()
    train_rows, dev_rows, test_rows = config.TRAIN_ROWS, config.DEV_ROWS, config.TEST_ROWS
    if any(type(rows) is not int or rows < 2 or rows % 2
           for rows in (train_rows, dev_rows, test_rows)):
        raise ValueError('UNBALANCED_PANEL_ROWS')
    try:
        secret = draw(80)
        if type(secret) is not int or not 0 <= secret < 1 << 80:
            raise ValueError
    except Stopped:
        raise
    except Exception:
        raise ValueError('PRIVATE_SOURCE_FAILED') from None
    used = {secret.to_bytes(10, 'little')}
    fixed_key = (secret & 0xFFFFFFFF, (secret >> 32) & 0xFFFFFFFF, secret >> 64)
    del secret
    earlier_plain: list[PairKeys] = []
    earlier_cipher: list[PairKeys] = []

    def paired(pairs: Words) -> Words:
        n = len(pairs)
        try:
            raw = draw_bytes(10 * n)
            if type(raw) is not bytes or len(raw) != 10 * n:
                raise ValueError
        except Stopped:
            raise
        except Exception:
            raise ValueError('PRIVATE_SOURCE_FAILED') from None
        # Compare exact keys, including the fixed TRAIN key, before calling the oracle.
        for i in range(0, len(raw), 10):
            key = raw[i:i + 10]
            if key in used:
                raise ValueError('DUPLICATE_PRIVATE_KEY')
            used.add(key)
        records = np.frombuffer(raw, dtype=[('low', '<u8'), ('high', '<u2')])
        low = records['low'].astype(np.uint64, order='C', copy=True)
        high = records['high'].astype(np.uint16, order='C', copy=True)
        points = pairs.copy()
        return _call(meter, pairs, lambda: encrypt_pairs(points, low, high),
                     (points, low, high), (n, 2))

    def generate(split: Split, rows: int, fixed: Callable[[Words], Words] | None = None
                 ) -> dict[str, BatchPanel]:
        meter.check()
        seed = [config.DATA_SEED, _SPLIT_IDS[split]]
        lrng, rng = (np.random.default_rng(np.random.SeedSequence([*seed, i])) for i in (0, 1))
        labels = (lrng.permutation(rows) < rows // 2).astype(np.uint8).reshape(rows, 1)
        arms: dict[str, Callable[[Words], Words]] = {config.PER_PAIR: paired}
        if fixed is not None:
            arms = {config.FIXED: fixed, **arms}
        words = {slot: np.empty((rows, 2), dtype=np.uint64) for slot in arms}
        stored_keys = rows if split != 'test' else min(rows, MAX_PAIRS)
        plain = np.empty(stored_keys, _PAIR_KEY)
        cipher = {slot: np.empty(stored_keys, _PAIR_KEY) for slot in arms}
        for start in range(0, rows, MAX_PAIRS):
            meter.check()
            end = min(start + MAX_PAIRS, rows)
            pairs = _points(rng, end - start, labels[start:end], sample, config.DIFFERENCE)
            key_slice = slice(0, end - start) if split == 'test' else slice(start, end)
            _keys(pairs, plain[key_slice])
            _require_disjoint(plain[key_slice], earlier_plain)
            for slot, oracle in arms.items():
                result = oracle(pairs)
                cipher_keys = cipher[slot][key_slice]
                _keys(result, cipher_keys)
                _require_disjoint(cipher_keys, earlier_cipher)
                words[slot][start:end] = result
        if split != 'test':
            meter.check()
            plain.sort(kind='quicksort')
            earlier_plain.append(plain)
            for keys in cipher.values():
                meter.check()
                keys.sort(kind='quicksort')
                earlier_cipher.append(keys)
        labels.flags.writeable = False
        panels = {}
        for slot, result in words.items():
            meter.check()
            digest = sha256(f'{config.METHOD}/{split}/{slot}/{rows}/N,2/<u8/MSB0/N,1/u8/'
                            .encode('ascii'))
            digest.update(memoryview(labels))
            digest.update(memoryview(result.astype('<u8', copy=False)))
            result.flags.writeable = False
            panels[slot] = BatchPanel(result, labels, digest.hexdigest(), True)
        return panels

    def fixed(pairs: Words, key: Key = fixed_key) -> Words:
        points = pairs.flatten()
        return _call(meter, pairs, lambda: encrypt(points, key, config.ROUNDS),
                     (points,), (points.size, 1))

    trains = generate('train', train_rows, fixed)
    del fixed, fixed_key
    dev = generate('dev', dev_rows)[config.PER_PAIR]

    def generate_test() -> BatchPanel:
        try:
            return generate('test', test_rows)[config.PER_PAIR]
        finally:
            used.clear()
            earlier_plain.clear()
            earlier_cipher.clear()

    return Panels(trains, dev, generate_test)


def production_panels(meter: Meter) -> Panels:
    from secrets import randbits, token_bytes

    from cipher_lab.ciphers.present_prefix import encrypt_prefixes

    return make_panels(meter, randbits, token_bytes, encrypt_prefixes, encrypt_pairs)

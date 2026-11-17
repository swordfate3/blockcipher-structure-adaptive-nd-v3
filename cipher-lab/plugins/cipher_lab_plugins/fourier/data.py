"""One private-key owner for paired Fourier queries and explicitly gated panels.

Only encrypt_batch's ndarray/key/rounds signature is accepted. Sampling is public,
with replacement; no label cache, global deduplication, reference or cipher check.
The owner never exports a key identity, complete ciphertext or private exception.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from cipher_lab.execution import Meter, Stopped
from cipher_lab_plugins.fourier.config import FourierRecipe

type Points = NDArray[np.uint64]
type Signs = NDArray[np.int8]
type Panel = tuple[Points, Signs, Signs | None]
type Paired = tuple[Points, Points, Signs, Signs | None, Signs, Signs | None,
                    dict[str, object]]
type Oracle = Callable[[Points, tuple[int, ...], int], object]
_NULL_SALT = b'cipher-lab/fourier-null-v1/20261111'
_SHA = re.compile(r'[0-9a-f]{64}\Z', re.ASCII)
_MAX_BATCH = 65536


def _readonly[DType: np.generic](array: NDArray[DType]) -> NDArray[DType]:
    # Backing bytes, not just setflags(False): callers cannot re-enable writing.
    return np.frombuffer(array.tobytes(), dtype=array.dtype)


class SpectralData:
    """Serial, one-shot domains. Gates grant permission, never acquire labels.

    Inject private_random(16) only after claim. It is called lazily after the first
    real reservation, at most once, including on failure. Synthetic recipes use
    only an injected artificial oracle; rounds=0 must never reach a real cipher.
    """

    def __init__(self, recipe: FourierRecipe, oracle: Oracle,
                 private_random: Callable[[int], bytes], meter: Meter) -> None:
        if (type(recipe) is not FourierRecipe or not callable(oracle)
                or not callable(private_random)):
            raise ValueError('expected Fourier recipe and injected callables')
        self._recipe = recipe
        self._oracle = oracle
        self._random = private_random
        self._meter = meter
        self._key: tuple[int, ...] | None = None
        self._key_attempted = False
        self._claimed: set[str] = set()
        self._finished: set[str] = set()
        self._null_enabled = True
        self._dev_sha: str | None = None
        self._test_sha: str | None = None

    def disable_null(self) -> None:
        """Irreversibly stop null work only; never disable real labels."""
        self._null_enabled = False

    def _claim(self, domain: str) -> None:
        if domain in self._claimed:
            raise Stopped('DOMAIN_REUSED')
        self._claimed.add(domain)

    def _key_once(self) -> tuple[int, ...]:
        if self._key is None:
            if self._key_attempted:
                raise Stopped('KEY_REUSE_LIMIT')
            self._key_attempted = True
            try:
                raw = self._random(16)
                if type(raw) is not bytes or len(raw) != 16:
                    raise ValueError
                packed = int.from_bytes(raw, 'big')
                self._key = tuple((packed >> shift) & 0xffffffff for shift in (0, 32, 64, 96))
            except Exception:
                raise Stopped('RANDOM_UNAVAILABLE') from None
        return self._key

    def _generator(self, seed: int, domain: str) -> np.random.Generator:
        self._meter.check()
        raw = b'cipher-lab/fourier-sampling-v1/' + str(seed).encode('ascii') + b'/'
        derived = int.from_bytes(sha256(raw + domain.encode('ascii')).digest(), 'big')
        return np.random.Generator(np.random.PCG64(derived))

    def _sample_pair(self, depth: int, count: int) -> tuple[Points, Points]:
        """Bounded public sampling primitive; k64 has empty suffix, no shift by64."""
        if type(depth) is not int or not 1 <= depth <= self._recipe.input_bits:
            raise ValueError('depth outside input domain')
        if type(count) is not int or not 1 <= count <= _MAX_BATCH:
            raise ValueError('invalid pair count')
        rng = self._generator(self._recipe.search_seed, f'search/{depth}')
        # random_raw covers the full uint64 domain, including UINT64_MAX.
        y, yy, z = (rng.bit_generator.random_raw(count) for _ in range(3))
        prefix = np.uint64(((1 << depth) - 1) << (64 - depth))
        domain = np.uint64(((1 << self._recipe.input_bits) - 1)
                           << (64 - self._recipe.input_bits))
        suffix = domain ^ prefix
        return (y & prefix) | (z & suffix), (yy & prefix) | (z & suffix)

    def _real(self, points: Points) -> Signs:
        signs = np.empty(len(points), dtype=np.int8)
        for start in range(0, len(points), _MAX_BATCH):
            batch = _readonly(points[start:start + _MAX_BATCH])
            self._meter.reserve('real_blocks', len(batch))
            key = self._key_once()
            try:
                response = self._oracle(batch, key, self._recipe.rounds)
            except Exception:
                raise Stopped('ORACLE_FAILED') from None
            try:
                if (type(response) is not np.ndarray or response.dtype != np.dtype(np.uint64)
                        or response.ndim != 1 or response.shape != batch.shape):
                    raise ValueError
                target = ((response >> np.uint64(63)) & np.uint64(1)).astype(np.int8)
                signs[start:start + len(batch)] = 1 - 2 * target
            except Exception:
                raise Stopped('ORACLE_RESPONSE') from None
            self._meter.complete('real_blocks', len(batch))
            self._meter.check()
        return _readonly(signs)

    def _null(self, points: Points) -> Signs | None:
        if not self._null_enabled:
            return None
        signs = np.empty(len(points), dtype=np.int8)
        for start in range(0, len(points), _MAX_BATCH):
            batch = points[start:start + _MAX_BATCH]
            self._meter.reserve('null_words', len(batch))
            for offset in range(0, len(batch), 4096):
                self._meter.check()
                part = batch[offset:offset + 4096]
                values = [1 - 2 * (sha256(_NULL_SALT + int(p).to_bytes(8, 'big'))
                          .digest()[0] & 1) for p in part]
                signs[start + offset:start + offset + len(part)] = values
                # Preserve validated partial work even if a later chunk stops.
                self._meter.complete('null_words', len(part))
            self._meter.check()
        return _readonly(signs)

    def paired(self, depth: int) -> Paired:
        if type(depth) is not int or not 1 <= depth <= self._recipe.input_bits:
            raise ValueError('depth outside input domain')
        domain = f'search/{depth}'
        self._claim(domain)
        p, q = self._sample_pair(depth, self._recipe.pairs_per_layer)
        p, q = _readonly(p), _readonly(q)
        g, gg = self._real(p), self._real(q)
        h, hh = self._null(p), self._null(q)
        self._finished.add(domain)
        return p, q, g, h, gg, hh, {
            'depth': depth, 'count': len(p),
            'pair_collisions': int(np.count_nonzero(p == q)),
            'sampling': 'with_replacement', 'split_exact_intersection': 'UNKNOWN',
            'null_label_mode': 'pseudorandom_function',
        }

    @staticmethod
    def _sha(model_sha: str) -> None:
        if type(model_sha) is not str or _SHA.fullmatch(model_sha) is None:
            raise ValueError('expected model SHA256')

    def release_dev(self, model_sha: str) -> None:
        self._sha(model_sha)
        if 'coef' not in self._finished or self._dev_sha is not None:
            raise ValueError('DEV release out of order')
        self._dev_sha = model_sha

    def release_test(self, model_sha: str) -> None:
        self._sha(model_sha)
        if ('dev' not in self._finished or self._test_sha is not None
                or model_sha != self._dev_sha):
            raise ValueError('TEST release out of order or changed model')
        self._test_sha = model_sha

    def panel(self, split: str) -> Panel:
        if split not in ('coef', 'dev', 'test'):
            raise ValueError('expected coef/dev/test panel')
        if split == 'dev' and self._dev_sha is None:
            raise ValueError('DEV not released')
        if split == 'test' and self._test_sha is None:
            raise ValueError('TEST not released')
        self._claim(split)
        count = getattr(self._recipe, f'{split}_size')
        seed = getattr(self._recipe, f'{split}_seed')
        rng = self._generator(seed, split)
        raw = rng.bit_generator.random_raw(count)
        suffix = np.uint64((1 << (64 - self._recipe.input_bits)) - 1)
        points = _readonly(raw & ~suffix)
        real, null = self._real(points), self._null(points)
        self._finished.add(split)
        return points, real, null

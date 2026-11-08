"""Fixed-private-key SIMON bit data, with an explicit deferred TEST label gate.

Construction prebuilds globally distinct public plaintexts for ALL splits, then
collects TRAIN/DEV labels. release_test grants permission only; labels(test) or
test_bundle subsequently collects TEST once. Callers must release only after
strict best-checkpoint loading (this owner cannot verify model selection).

Public sampling/shuffling/init seeds never choose keys or null labels. Inject
secrets.token_bytes in production; finite byte fixtures in engineering tests.
No file I/O, training, production cipher imports or public private-state export.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from itertools import islice

import numpy as np
import torch
from torch import Tensor

from cipher_lab.data.bundles import DatasetBundle, PredictionTask
from cipher_lab.data.cubes import Oracle, RandomSource
from cipher_lab.execution import Kind, Meter, Stopped
from cipher_lab.training_config import PredictionRecipe

_SPLITS = ("train", "dev", "test")
_CHUNK = 8192


class PredictionData:
    """Single-run/single-thread owner; all returned tensors are defensive copies.

    oracle(points, key_words, rounds) returns exactly len(points) uint64s. Key
    words are uint32s, least-significant first, from a big-endian 16-byte key,
    exactly matching ciphers.simon.encrypt_many. Inputs and targets are MSB0.
    This owner alone charges real_blocks/null_words. Failed label domains stay
    claimed; no retry can silently reuse a partial private acquisition.
    """

    def __init__(self, recipe: PredictionRecipe, oracle: Oracle,
                 private_random: RandomSource, meter: Meter) -> None:
        if not isinstance(recipe, PredictionRecipe) or not callable(oracle):
            raise ValueError("expected prediction recipe/oracle")
        self._recipe = recipe
        self._oracle = oracle
        self._random = private_random
        self._meter = meter
        self._released = False
        self._claimed: set[str] = set()
        self._labels: dict[str, Tensor] = {}
        self._key: tuple[int, ...] | None = None
        self._points: dict[str, tuple[int, ...]] = {}
        self._inputs: dict[str, Tensor] = {}
        public = random.Random(recipe.training.sample_seed)
        seen: set[int] = set()
        for split in _SPLITS:
            points: list[int] = []
            for _ in range(getattr(recipe.training, split)):
                self._meter.check()
                for _attempt in range(128):
                    value = public.getrandbits(64)
                    if value not in seen:
                        seen.add(value)
                        points.append(value)
                        break
                else:
                    raise Stopped("RANDOM_UNAVAILABLE")
            self._points[split] = tuple(points)
            raw = b"".join(point.to_bytes(8, "big") for point in points)
            bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(-1, 64)
            self._inputs[split] = torch.from_numpy(bits).to(torch.float32)
        self.labels("train")
        self.labels("dev")

    def _bytes(self, size: int) -> bytes:
        try:
            value = self._random(size)
            if type(value) is not bytes or len(value) != size:
                raise ValueError
            return value
        except Exception:
            raise Stopped("RANDOM_UNAVAILABLE") from None

    def _encrypt(self, points: tuple[int, ...]) -> tuple[int, ...]:
        self._meter.check()
        assert self._key is not None
        try:
            response = self._oracle(points, self._key, self._recipe.rounds)
        except Exception:
            raise Stopped("ORACLE_FAILED") from None
        try:
            if not isinstance(response, Sequence) or len(response) != len(points):
                raise ValueError
            words = tuple(islice(iter(response), len(points) + 1))
            if len(words) != len(points) or any(
                type(word) is not int or not 0 <= word < (1 << 64) for word in words
            ):
                raise ValueError
            return words
        except Exception:
            raise Stopped("ORACLE_RESPONSE") from None

    @staticmethod
    def _split(split: str) -> None:
        if split not in _SPLITS:
            raise ValueError("expected train/dev/test split")

    def inputs(self, split: str) -> Tensor:
        """Public, prebuilt inputs only; never triggers a private operation."""
        self._split(split)
        return self._inputs[split].clone()

    def labels(self, split: str) -> Tensor:
        self._split(split)
        if split == "test" and not self._released:
            raise ValueError("TEST labels not released")
        if split in self._labels:
            return self._labels[split].clone()
        if split in self._claimed:
            raise Stopped("DOMAIN_REUSED")
        self._claimed.add(split)
        points = self._points[split]
        labels = torch.empty((len(points), 1), dtype=torch.float32)
        kind: Kind = "real_blocks" if self._recipe.label_mode == "cipher" else "null_words"
        for start in range(0, len(points), _CHUNK):
            batch = points[start:start + _CHUNK]
            # Reserve before first private key/null draw or oracle invocation.
            self._meter.reserve(kind, len(batch))
            if kind == "real_blocks":
                if self._key is None:
                    packed = int.from_bytes(self._bytes(16), "big")
                    self._key = tuple((packed >> shift) & 0xFFFFFFFF
                                      for shift in (0, 32, 64, 96))
                words = self._encrypt(batch)
                values = [(word >> (63 - self._recipe.target_bit)) & 1 for word in words]
            else:
                values = []
                for _ in batch:
                    self._meter.check()
                    values.append(self._bytes(1)[0] & 1)
                    # Each validated null draw is an independently completed
                    # unit, even if a later draw in this chunk fails.
                    self._meter.complete(kind, 1)
            labels[start:start + len(batch), 0] = torch.tensor(values, dtype=torch.float32)
            if kind == "real_blocks":
                self._meter.complete(kind, len(batch))
            self._meter.check()
        self._labels[split] = labels
        return labels.clone()

    def release_test(self) -> None:
        """Idempotently grant permission, without accessing any TEST label."""
        self._released = True

    def test_bundle(self) -> DatasetBundle:
        labels = self.labels("test")
        declaration = PredictionTask("SIMON", self._recipe.rounds, self._recipe.target_bit,
                                     self._recipe.label_mode)
        return DatasetBundle(self._recipe.method, (declaration,),
                             {"SIMON": (self.inputs("test"), labels)})

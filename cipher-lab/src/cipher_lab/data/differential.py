"""Prebuilt four-pair differential panels; secrets confined to collection.

Unlike prediction, construction acquires ALL splits including TEST. Consumers
must not read/score TEST during DEV selection; this provider's panel API is not
an isolation boundary against a malicious trainer. Public plaintext sampling and
balanced label ordering use separate sample_seed/shuffle_seed streams. Private
keys come only from the injected byte source, never any public seed.

No I/O, model, selection, public13 representation or eager cipher imports.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from itertools import islice

import numpy as np
import torch
from torch import Tensor

from cipher_lab.data.bundles import DatasetBundle
from cipher_lab.data.cubes import Oracle, RandomSource
from cipher_lab.execution import Meter, Stopped
from cipher_lab.training_config import TASK_FAMILIES, SharedMRecipe, TaskSpec

_SPLITS = ("train", "dev", "test")


class DifferentialData:
    """One prebuilt run, one meter owner, exactly eight real blocks per group.

    Every oracle has the SIMON batch signature (points, key_words, rounds).
    uint32 key words are least-significant first from big-endian private bytes:
    SIMECK/SIMON 4 words; PRESENT 3 words (high word <=65535); SKINNY 2 words.
    The pure family encrypt_many adapters consume exactly this convention.
    Ciphertexts are paired [P0,Q0,...,P3,Q3], packed to CPU FP32 MSB0
    [groups,4,2,64]. Labels are [groups,1]. No key material is returned.
    """

    def __init__(self, recipe: SharedMRecipe, oracles: Mapping[str, Oracle],
                 private_random: RandomSource, meter: Meter) -> None:
        if (not isinstance(recipe, SharedMRecipe)
                or set(oracles) != {task.cipher for task in recipe.tasks}
                or any(not callable(oracle) for oracle in oracles.values())):
            raise ValueError("expected shared-M recipe and four family oracles")
        self._recipe = recipe
        self._meter = meter
        self._random = private_random
        self._keys: set[tuple[str, bytes]] = set()
        self._panels: dict[tuple[str, str], tuple[Tensor, Tensor]] = {}
        public = random.Random(recipe.training.sample_seed)
        shuffle = random.Random(recipe.training.shuffle_seed)
        for task, family in zip(recipe.tasks, TASK_FAMILIES, strict=True):
            for split in _SPLITS:
                size = getattr(recipe.training, split)
                labels = [0] * (size // 2) + [1] * (size // 2)
                shuffle.shuffle(labels)
                inputs = torch.empty((size, 4, 2, 64), dtype=torch.float32)
                for group, label in enumerate(labels):
                    # Sole charge, before private random or oracle operation.
                    self._meter.reserve("real_blocks", 8)
                    key = self._key(task.cipher, family.key_bits // 8)
                    points: list[int] = []
                    for _ in range(4):
                        first = public.getrandbits(64)
                        second = first ^ task.difference if label else public.getrandbits(64)
                        points.extend((first, second))
                    words = self._encrypt(oracles[task.cipher], tuple(points), key, task.rounds)
                    raw = b"".join(word.to_bytes(8, "big") for word in words)
                    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(4, 2, 64)
                    inputs[group] = torch.from_numpy(bits).to(torch.float32)
                    self._meter.complete("real_blocks", 8)
                    self._meter.check()
                self._panels[(split, task.cipher)] = (
                    inputs, torch.tensor(labels, dtype=torch.float32).reshape(-1, 1))
        # Private raw key set is useful only during construction collision checks.
        self._keys.clear()

    def _key(self, cipher: str, size: int) -> tuple[int, ...]:
        for _ in range(128):
            self._meter.check()
            try:
                raw = self._random(size)
                if type(raw) is not bytes or len(raw) != size:
                    raise ValueError
            except Exception:
                raise Stopped("RANDOM_UNAVAILABLE") from None
            identity = (cipher, raw)
            if identity not in self._keys:
                self._keys.add(identity)
                packed = int.from_bytes(raw, "big")
                return tuple((packed >> shift) & 0xFFFFFFFF
                             for shift in range(0, size * 8, 32))
        raise Stopped("KEY_REUSE_LIMIT")

    def _encrypt(self, oracle: Oracle, points: tuple[int, ...], key: tuple[int, ...],
                 rounds: int) -> tuple[int, ...]:
        self._meter.check()
        try:
            response = oracle(points, key, rounds)
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

    def panel(self, split: str, task: TaskSpec) -> tuple[Tensor, Tensor]:
        """Read a prebuilt panel (including TEST); no collection side effects."""
        if split not in _SPLITS or task not in self._recipe.tasks:
            raise ValueError("unknown split/task identity")
        x, y = self._panels[(split, task.cipher)]
        return x.clone(), y.clone()

    def test_bundle(self) -> DatasetBundle:
        return DatasetBundle(self._recipe.method, self._recipe.tasks,
                             {task.cipher: self.panel("test", task) for task in self._recipe.tasks})

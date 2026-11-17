"""Online four-pair sampling. Only this module receives private bytes and keys.

TRAIN batches are not retained or replayable. DEV is explicitly prepared; TEST
is constructed only on test(task), which the training owner must call after all
five persisted best checkpoints have been frozen and strictly reloaded. This is
an access-timing contract, not a security boundary against malicious callers.
No cipher/RNG/model/resource implementation is selected implicitly here.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import numpy as np

from .batch import Batch as Batch
from .config import Recipe, Task, validate

Oracle = Callable[[tuple[int, ...], tuple[int, ...], int], Sequence[int]]
PrivateSource = Callable[[int], bytes]


class BlockMeter(Protocol):
    def reserve(self, kind: str, amount: int) -> None: ...
    def complete(self, kind: str, amount: int) -> None: ...
    def check(self) -> None: ...


class DataError(RuntimeError):
    """Public fixed-code failure. Never includes private source/oracle exceptions."""


def derive_seed(recipe: Recipe, task: Task, split: str, index: int, purpose: str) -> int:
    """Stable public SHA256 stream, never a source of actual key material."""
    if (task not in recipe.tasks or split not in ('train', 'dev', 'test')
            or type(index) is not int or index < 0
            or purpose not in ('sample', 'shuffle', 'bootstrap')):
        raise ValueError('invalid public substream identity')
    seed = getattr(recipe.seeds, purpose)
    identity = [recipe.namespace, seed, task.cipher, task.rounds, task.difference,
                split, index, purpose]
    return int.from_bytes(hashlib.sha256(json.dumps(
        identity, separators=(',', ':')).encode('ascii')).digest(), 'big')




class Provider:
    """Injected oracle(points,key_words,rounds) encrypts all eight blocks per group.

    One private draw per group; big-endian bytes -> least-significant uint32
    word first, PRESENT's third/high word is 16 bits. Natural key collisions are
    allowed, not rejected/reseeded. No key/source state is serializable output.
    A failed collection permanently stops this provider (no hidden retry).
    """

    def __init__(self, recipe: Recipe, oracles: Mapping[str, Oracle],
                 private_random: PrivateSource, meter: BlockMeter) -> None:
        validate(recipe)
        if (set(oracles) != {t.cipher for t in recipe.tasks}
                or any(not callable(o) for o in oracles.values())
                or not callable(private_random)):
            raise ValueError('four explicit oracles and private source required')
        self._recipe = recipe
        self._oracles = dict(oracles)
        self._private = private_random
        self._meter = meter
        self._next = {task.cipher: 0 for task in recipe.tasks}
        self._panels: dict[tuple[str, str], Batch] = {}
        self._stopped = False

    def _task(self, task: Task) -> None:
        if self._stopped:
            raise DataError('PROVIDER_STOPPED')
        if not isinstance(task, Task) or task not in self._recipe.tasks:
            raise ValueError('unknown public task')

    def train(self, task: Task, cycle: int) -> Batch:
        self._task(task)
        cycles = self._recipe.training.epochs * self._recipe.training.batches_per_epoch
        if (type(cycle) is not int or not 0 <= cycle < cycles
                or cycle != self._next[task.cipher]):
            raise ValueError('TRAIN must be requested once per task in cycle order')
        batch = self._collect(task, 'train', cycle, self._recipe.training.batch_groups)
        self._next[task.cipher] += 1
        return batch

    def prepare_dev(self) -> Mapping[str, Batch]:
        """Call before updates. Constructs four DEV panels only, never TEST."""
        return {task.cipher: self.dev(task) for task in self._recipe.tasks}

    def dev(self, task: Task) -> Batch:
        return self._panel(task, 'dev', self._recipe.training.dev_groups)

    def test(self, task: Task) -> Batch:
        """Only the training owner, after five strict best reloads, may call this."""
        return self._panel(task, 'test', self._recipe.training.test_groups)

    def _panel(self, task: Task, split: str, size: int) -> Batch:
        self._task(task)
        identity = (split, task.cipher)
        if identity not in self._panels:
            # Collect bounded chunks, then store only immutable packed panel bytes.
            chunk = self._recipe.training.eval_batch_groups
            parts = [self._collect(task, split, i, chunk) for i in range(size // chunk)]
            self._panels[identity] = Batch(
                np.concatenate([part.ciphertext for part in parts]),
                np.concatenate([part.labels for part in parts]), task, split, 0)
        return self._panels[identity]

    def _collect(self, task: Task, split: str, index: int, size: int) -> Batch:
        try:
            return self._generate(task, split, index, size)
        except Exception:
            self._stopped = True
            raise

    def _generate(self, task: Task, split: str, index: int, size: int) -> Batch:
        public = random.Random(derive_seed(self._recipe, task, split, index, 'sample'))
        shuffle = random.Random(derive_seed(self._recipe, task, split, index, 'shuffle'))
        labels = [0] * (size // 2) + [1] * (size // 2)
        shuffle.shuffle(labels)
        packed = np.empty((size, 4, 2, 8), dtype=np.uint8)
        for group, label in enumerate(labels):
            self._meter.check()
            self._meter.reserve('real_blocks', 8)
            try:
                raw = self._private(task.key_bytes)
                if type(raw) is not bytes or len(raw) != task.key_bytes:
                    raise ValueError
                integer = int.from_bytes(raw, 'big')
                key = tuple((integer >> shift) & 0xFFFFFFFF
                            for shift in range(0, task.key_bytes * 8, 32))
            except Exception:
                raise DataError('PRIVATE_SOURCE_FAILED') from None
            points: list[int] = []
            for _ in range(4):
                first = public.getrandbits(64)
                second = first ^ task.difference if label else public.getrandbits(64)
                points.extend((first, second))
            try:
                response = self._oracles[task.cipher](tuple(points), key, task.rounds)
            except Exception:
                raise DataError('ORACLE_FAILED') from None
            try:
                if not isinstance(response, Sequence) or len(response) != 8:
                    raise ValueError
                words = tuple(response)
                if len(words) != 8 or any(type(w) is not int or not 0 <= w < 2**64
                                          for w in words):
                    raise ValueError
                block_bytes = b''.join(word.to_bytes(8, 'big') for word in words)
                packed[group] = np.frombuffer(block_bytes, dtype=np.uint8).reshape(4, 2, 8)
            except Exception:
                raise DataError('ORACLE_RESPONSE') from None
            self._meter.complete('real_blocks', 8)
            self._meter.check()
        return Batch(packed, np.asarray(labels, dtype=np.uint8).reshape(-1, 1),
                     task, split, index)

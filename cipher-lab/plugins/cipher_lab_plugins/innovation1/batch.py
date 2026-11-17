"""Immutable public ciphertext/label batches; no generation or private source."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .config import Task

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, eq=False)
class Batch:
    """Packed ciphertext [G,4,2,8] uint8/MSB0, labels [G,1] uint8.

    Arrays are backed by immutable bytes, not merely a reversible write flag.
    Metadata contains only public task/split/index. No plaintext/key identities.
    tensors() creates independent CPU FP32 [G,4,2,64], [G,1] for both arms.
    """

    ciphertext: NDArray[np.uint8]
    labels: NDArray[np.uint8]
    task: Task
    split: str
    index: int

    def __post_init__(self) -> None:
        c, y = self.ciphertext, self.labels
        if (c.dtype != np.uint8 or c.ndim != 4 or c.shape[1:] != (4, 2, 8)
                or c.shape[0] < 1 or y.dtype != np.uint8 or y.shape != (c.shape[0], 1)
                or not np.isin(y, [0, 1]).all() or self.split not in ('train', 'dev', 'test')
                or type(self.index) is not int or self.index < 0):
            raise ValueError('invalid packed public batch')
        for name, a in (('ciphertext', c), ('labels', y)):
            object.__setattr__(self, name, np.frombuffer(a.tobytes(), dtype=np.uint8).reshape(
                a.shape))

    @property
    def groups(self) -> int:
        return int(self.labels.shape[0])

    @property
    def digest(self) -> str:
        identity = [self.task.cipher, self.task.rounds, self.task.difference,
                    self.split, self.index, self.groups]
        h = hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode('ascii'))
        h.update(self.ciphertext.tobytes())
        h.update(self.labels.tobytes())
        return h.hexdigest()

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        import torch
        bits = np.unpackbits(self.ciphertext, axis=-1, bitorder='big').astype(np.float32)
        return torch.from_numpy(bits), torch.from_numpy(self.labels.astype(np.float32))

    def slice(self, start: int, stop: int) -> Batch:
        """Evaluation view copied to immutable packed bytes; retains panel identity."""
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= self.groups:
            raise ValueError('invalid group slice')
        return Batch(self.ciphertext[start:stop], self.labels[start:stop],
                     self.task, self.split, self.index)


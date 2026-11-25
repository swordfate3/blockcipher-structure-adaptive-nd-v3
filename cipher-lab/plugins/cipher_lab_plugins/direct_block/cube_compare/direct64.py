"""Read pinned absolute labels; expose only single-word rows to the fit."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np

from cipher_lab.artifacts import ArtifactError
from cipher_lab.execution import Meter

from . import config
from .data import Floats, Indices, Labels, Words
from .reuse import SavedPanel, _check_disjoint, _PinnedSource


def _word_bits(words: Words) -> Labels:
    return np.unpackbits(words.astype('>u8', copy=False).view(np.uint8),
                         bitorder='big').reshape(-1, 64)


class DirectPanel:
    """C-order absolute P/C view of an already validated source round/split."""

    def __init__(self, paired: SavedPanel) -> None:
        self._plaintext = paired.plaintext.reshape(-1)
        self._ciphertext = paired.ciphertext.reshape(-1)
        self.family_ids = np.repeat(paired.family_ids, 2)
        self.family_ids.flags.writeable = False
        self._raw = paired.raw

    def __len__(self) -> int:
        return len(self._plaintext)

    def batch(self, indices: Indices) -> tuple[Floats, Labels]:
        return _word_bits(self._plaintext[indices]).astype(np.float32), self.label_batch(indices)

    def label_batch(self, indices: Indices) -> Labels:
        return _word_bits(self._ciphertext[indices])

    def archive(self) -> bytes:
        return self._raw


DirectMatrix = tuple[DirectPanel, DirectPanel, DirectPanel]


class DirectPanels:
    def __init__(self, root: Path, meter: Meter, *, run_id: str, run_sha256: str,
                 artifacts: Mapping[str, tuple[int, str]], families: tuple[int, int, int]) -> None:
        if set(artifacts) != {f'dataset/{name}-{split}.npz'
                              for name in config.LOW_NAMES for split in config.SPLITS}:
            raise ArtifactError('INVALID_SOURCE_RUN')
        self._source = _PinnedSource(root, meter, run_id=run_id, run_sha256=run_sha256,
                                     pins=artifacts)
        self._meter, self._families = meter, families
        self._next = 0
        self._failed = False
        self._bases: list[Words] = []

    def build(self, split: Literal['train', 'dev']) -> DirectMatrix:
        if split not in ('train', 'dev'):
            raise ValueError('invalid split order; use label_test')
        return self._build(split)

    def label_test(self) -> DirectMatrix:
        """Called by training only after strict disk restore of all three bests."""
        return self._build('test')

    def _build(self, split: str) -> DirectMatrix:
        i = self._next
        if self._failed or i >= len(config.SPLITS) or split != config.SPLITS[i]:
            raise ValueError('invalid split order')
        self._failed = True
        r1, bases = self._source.panel(f'dataset/r1-{split}.npz', self._families[i])
        _check_disjoint(bases, self._bases, self._meter)

        def aligned(name: str) -> DirectPanel:
            panel, _ = self._source.panel(f'dataset/{name}-{split}.npz', self._families[i])
            if not (np.array_equal(panel.plaintext, r1.plaintext)
                    and np.array_equal(panel.family_ids, r1.family_ids)):
                raise ValueError('SOURCE_ROUND_ALIGNMENT')
            self._meter.check()
            return DirectPanel(panel)

        result = (DirectPanel(r1), aligned('r2'), aligned('r3'))
        self._meter.check()
        self._bases.append(bases)
        self._next += 1
        self._failed = False
        return result


def production_direct_panels(root: Path, meter: Meter) -> DirectPanels:
    return DirectPanels(root, meter, run_id=config.DIRECT_SOURCE_RUN_ID,
                        run_sha256=config.DIRECT_SOURCE_RUN_SHA256,
                        artifacts=config.DIRECT_SOURCE_ARTIFACTS, families=config.FAMILIES)

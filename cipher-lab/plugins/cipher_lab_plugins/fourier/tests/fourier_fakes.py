"""Task2-only tiny fixtures; guards block scientific imports and private RNG/RSS.

Not conftest: never changes unrelated test collection or production behavior.
The child probe calls install_guards itself; no inherited monkeypatch assumption.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import random
import secrets
import sys
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from types import ModuleType
from typing import NoReturn

import numpy as np
import pytest
from cipher_lab_plugins.fourier.config import FourierRecipe
from numpy.typing import NDArray

from cipher_lab import execution
from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter


class ScientificImportTrap(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None,
                  target: object = None) -> NoReturn | None:
        if fullname.split('.')[0] in ('torch', 'psutil') or fullname.startswith(
            ('cipher_lab.ciphers', 'cipher_lab.models', 'cipher_lab.training')
        ):
            raise AssertionError('scientific import forbidden')
        return None


def forbidden(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError('real scientific call forbidden')


def install_guards(patch: pytest.MonkeyPatch) -> None:
    patch.setattr(secrets, 'token_bytes', forbidden)
    patch.setattr(secrets, 'randbits', forbidden)
    patch.setattr(secrets, 'randbelow', forbidden)
    patch.setattr(secrets, 'choice', forbidden)
    patch.setattr(random, 'SystemRandom', forbidden)
    patch.setattr(random, '_urandom', forbidden)
    # RunStore's technical temporary filenames use uuid4, not private science.
    # Give that dependency a deterministic fake before trapping os.urandom.
    counter = iter(range(1, 100000))
    patch.setattr(uuid, 'uuid4', lambda: uuid.UUID(int=next(counter)))
    patch.setattr('cipher_lab.artifacts.uuid4', lambda: uuid.UUID(int=next(counter)))
    # NumPy lazily initializes an unused legacy global RNG at import. Feed that
    # import deterministic fake bytes, then trap all subsequent entropy requests.
    patch.setattr(os, 'urandom', lambda size: b'\0' * size)
    patch.setattr(secrets, 'randbits', lambda bits: 0)
    import numpy.random  # noqa: F401
    patch.setattr(secrets, 'randbits', forbidden)
    patch.setattr(os, 'urandom', forbidden)
    patch.setattr(execution, 'production_rss_bytes', forbidden)
    # If a forbidden dependency was pre-imported, trap its existing entrypoints.
    for module_name, module in tuple(sys.modules.items()):
        if module_name.startswith('cipher_lab.ciphers.'):
            for attr in ('encrypt_batch', 'encrypt_many', 'encrypt'):
                if hasattr(module, attr):
                    patch.setattr(module, attr, forbidden)
    if 'torch' in sys.modules:
        torch = sys.modules['torch']
        patch.setattr(torch.nn.Module, '__call__', forbidden)
        patch.setattr(torch.nn.Linear, '__init__', forbidden)
        patch.setattr(torch.Tensor, 'backward', forbidden)
        for optimizer in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
            patch.setattr(optimizer, '__init__', forbidden)
            patch.setattr(optimizer, 'step', forbidden)
    trap = ScientificImportTrap()
    original_import_module = importlib.import_module

    def guarded_import_module(name: str, package: str | None = None) -> ModuleType:
        # meta_path cannot see sys.modules hits. Apply precisely the same policy
        # before delegating; resolve relative names without importing their package.
        fullname = importlib.util.resolve_name(name, package) if name.startswith('.') else name
        trap.find_spec(fullname)
        return original_import_module(name, package)

    patch.setattr(importlib, 'import_module', guarded_import_module)
    patch.setattr(sys, 'meta_path', [trap, *sys.meta_path])


@pytest.fixture(autouse=True)
def science_guard(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    install_guards(monkeypatch)
    yield


def recipe(**changes: object) -> FourierRecipe:
    values: dict[str, object] = dict(synthetic=True, rounds=0, input_bits=2,
        pairs_per_layer=128, coef_size=128, dev_size=128, test_size=128,
        tau=0.5, beam_width=4)
    values.update(changes)
    return FourierRecipe(**values)  # type: ignore[arg-type]


def meter(*, real: int = 10000, null: int = 10000) -> Meter:
    now = datetime(2026, 11, 11, tzinfo=UTC)
    policy = Policy(schema=2, wall_seconds=900., cpu_seconds=900.,
        deadline_utc='2026-11-12T00:00:00+00:00', real_blocks=real, null_words=null,
        rss_stop_bytes=1000, output_bytes=4_000_000, optimizer_updates=0, forward_rows=0)
    return Meter(policy, Clocks(wall=lambda: 0., cpu=lambda: 0., utc=lambda: now), lambda: 1)


class PrivateBytes:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, size: int) -> bytes:
        self.calls += 1
        assert size == 16 and self.calls == 1
        return bytes(range(16))


class ToyOracle:
    def __init__(self) -> None:
        self.batches: list[int] = []

    def __call__(self, points: NDArray[np.uint64], key: tuple[int, ...],
                 rounds: int) -> NDArray[np.uint64]:
        assert type(points) is np.ndarray and points.dtype == np.uint64
        assert points.flags.c_contiguous and 0 < len(points) <= 65536
        assert len(key) == 4 and rounds in (0, 6)
        self.batches.append(len(points))
        # Literal MSB parity, placed in output bit0; not a real cipher.
        return np.array([((int(p) >> 63) & 1) << 63 for p in points], dtype=np.uint64)

"""Finite artificial aggregates: no cipher, RNG, RSS probes or optimizer calls."""
from __future__ import annotations

import copy
import sys
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pytest

import cipher_lab.integral.stages as stages
from cipher_lab.config import Cell, StageSizes
from cipher_lab_plugins.integral_search import config


def source_identity() -> dict[str, str]:
    return SOURCE.copy()


SOURCE = {"cipher_lab/artificial.py": "0" * 64}


def recipe(rounds: tuple[int, ...] = (11, 12, 13, 14), *,
           discovery: int = 1, confirm: int = 1) -> config.SearchRecipe:
    assert hasattr(config, "SearchRecipe"), "missing typed search recipe"
    return config.SearchRecipe(
        1, "integral-search-v1", rounds,
        (tuple(range(0, 32, 2)), tuple(range(32, 64, 2))),
        10, 16, StageSizes(discovery, 1, confirm), 65536,
    )


class FakeMeter:
    def __init__(self) -> None:
        self.hook: Callable[[], None] = lambda: None
        self.checks = 0

    def check(self) -> None:
        self.checks += 1
        self.hook()

    def snapshot(self) -> dict[str, object]:
        return {"checks": self.checks}


class FakeStore:
    def __init__(self) -> None:
        self.payloads: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self.final: dict[str, Any] = {}
        self.finalizations = 0
        self.hook: Callable[[Mapping[str, Any]], None] = lambda value: None

    def current_source_identity(self) -> dict[str, str]:
        return source_identity()

    def source_snapshot(self) -> dict[str, str]:
        return SOURCE.copy()

    def write_snapshot(self, value: Mapping[str, object]) -> None:
        self.snapshots.append(copy.deepcopy(dict(value)))
        self.hook(value)

    def append_event(self, value: Mapping[str, object]) -> None:
        self.events.append(copy.deepcopy(dict(value)))

    def write_payload(self, name: str, value: object) -> str:
        assert name not in self.payloads, "payload indices must be globally unique"
        self.payloads[name] = copy.deepcopy(value)
        return name

    def finalize(self, value: Mapping[str, object]) -> None:
        self.finalizations += 1
        assert self.finalizations == 1
        self.final = copy.deepcopy(dict(value))


class FakeCollector:
    def __init__(self) -> None:
        self.calls: list[tuple[Cell, tuple[str, str, int]]] = []
        self.value: Callable[[Cell, tuple[str, str, int]], int] = (
            lambda cell, domain: 9 if domain[0] == "null" else 0)
        self.hook: Callable[[Cell, tuple[str, str, int]], None] = lambda cell, domain: None
        self.completed = 0

    def row(self, cell: Cell, domain: tuple[str, str, int]) -> int:
        assert (cell, domain) not in self.calls, "canonical cell/domain reused"
        self.calls.append((cell, domain))
        self.hook(cell, domain)
        row = self.value(cell, domain)
        self.completed += 1
        return row

    def counts(self) -> dict[str, int]:
        return {"attempted_groups": len(self.calls), "completed_groups": self.completed}


@pytest.fixture(autouse=True)
def narrow_math(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.modules[__name__], "source_identity", lambda: SOURCE.copy())

    def kernel(rows: Iterable[int], *, check: Callable[[], None]) -> tuple[int, ...]:
        check()
        first = tuple(rows)[0]
        result = () if first == 9 else ((1, 2) if first == 8 else (1,))
        check()
        return result

    def survivor(basis: tuple[int, ...], rows: Iterable[int], *,
                 check: Callable[[], None]) -> tuple[int, ...]:
        check()
        result = () if tuple(rows)[0] == 9 else basis
        check()
        return result

    def confirms(basis: tuple[int, ...], rows: Iterable[int], *,
                 check: Callable[[], None]) -> bool:
        check()
        result = all((vector & row).bit_count() % 2 == 0 for row in rows for vector in basis)
        check()
        return result

    monkeypatch.setattr(stages, "kernel", kernel)
    monkeypatch.setattr(stages, "survivor_basis", survivor)
    monkeypatch.setattr(stages, "confirms", confirms)



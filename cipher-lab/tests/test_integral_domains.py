"""Ordinary domains and empty-D screen semantics remain core contracts."""
from typing import Any

from integral_helpers.fakes import (
    FakeCollector,
    FakeMeter,
    FakeStore,
)
from integral_helpers.fakes import (
    narrow_math as narrow_math,
)

from cipher_lab.config import Cell, Recipe, StageSizes
from cipher_lab.integral import method


def execute(r: Recipe, collector: FakeCollector) -> tuple[dict[str, Any], FakeCollector, FakeStore]:
    store = FakeStore()
    result = method.run_integral(r, collector, FakeMeter(), store)
    assert result == store.final and store.finalizations == 1
    return result, collector, store

def test_plain_recipe_empty_d_still_collects_screen() -> None:
    r = Recipe(1, "integral-empirical-v1", (Cell(11, (0, 1)),), StageSizes(1, 1, 1), 4)
    collector = FakeCollector()
    collector.value = lambda cell, domain: 9
    result, collector, _ = execute(r, collector)
    assert result["status"] == "SUCCEEDED"
    assert [d[1] for _, d in collector.calls] == ["discovery", "screen"] * 2
    assert "selections" not in result["summary"]


def test_plain_recipe_keeps_local_domains_and_original_journal_schema() -> None:
    r = Recipe(1, "integral-empirical-v1", (Cell(11, (0, 1)), Cell(12, (0, 1))),
               StageSizes(2, 3, 2), 4)
    collector = FakeCollector()
    collector.value = lambda cell, domain: (
        1 if domain[0] == "null" and domain[1] == "confirm" else 0)
    result, collector, store = execute(r, collector)
    assert result["status"] == "SUCCEEDED"
    expected = [(side, phase, index) for side in ("real", "null")
                for phase, size in (("discovery", 2), ("screen", 3), ("confirm", 2))
                for index in range(size)]
    assert [domain for _, domain in collector.calls] == expected * 2
    for event, (_, domain) in zip(store.events, collector.calls, strict=True):
        group = event["summary"]
        assert set(group) == {"cell_index", "side", "phase", "index", "row"}
        assert group["index"] == domain[2]



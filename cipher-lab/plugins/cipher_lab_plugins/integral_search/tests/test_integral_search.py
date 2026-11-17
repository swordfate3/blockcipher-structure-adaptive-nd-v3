"""Finite artificial aggregates: no cipher, RNG, RSS probes or optimizer calls."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

import cipher_lab.integral.stages as stages
import cipher_lab_plugins.integral_search.method as method
from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.config import Cell, StageSizes
from cipher_lab.execution import Stopped
from cipher_lab_plugins.integral_search import config
from cipher_lab_plugins.integral_search.tests import fakes
from cipher_lab_plugins.integral_search.tests.fakes import (
    SOURCE,
    FakeCollector,
    FakeMeter,
    FakeStore,
    recipe,
)
from cipher_lab_plugins.integral_search.tests.fakes import (
    narrow_math as narrow_math,
)


def execute(r: config.SearchRecipe, collector: FakeCollector | None = None,
            meter: FakeMeter | None = None, store: FakeStore | None = None,
            ) -> tuple[dict[str, Any], FakeCollector, FakeStore]:
    collector = collector or FakeCollector()
    meter = meter or FakeMeter()
    store = store or FakeStore()
    result = method.run_integral(r, collector, meter, store)
    assert result == store.final
    assert store.finalizations == 1
    return result, collector, store


@pytest.mark.parametrize("dimensions,ranks,expected", [
    ((1, 2), (1, 2080), 1), ((2, 1), (2080, 1), 0),
    ((1, 1), (2080, 1), 0), ((0, 1), (1, 2080), 1),
    ((0, 0), (2080, 2079), 1), ((0, 0), (2079, 2080), 0),
    ((0, 0), (2080, 2080), 0),
])
def test_select_pattern_only_public_dimensions_and_ranks(
    dimensions: tuple[int, int], ranks: tuple[int, int], expected: int,
) -> None:
    assert hasattr(method, "select_pattern"), "missing pure selection API"
    assert method.select_pattern(dimensions, ranks) == expected


def test_empty_discovery_skips_sc_but_saves_full_empty_spaces() -> None:
    collector = FakeCollector()
    collector.value = lambda cell, domain: 9
    result, collector, store = execute(recipe(), collector)
    assert result["status"] == "SUCCEEDED"
    cells = result["summary"]["cells"]
    visited = [entry for entry in cells if entry["outcome"] != "NOT_STARTED_GATE"]
    assert len(visited) == 9  # all eight k10, then r11 k16 despite all negatives
    assert [(c.rounds, len(c.active), c.active[0], d[0]) for c, d in collector.calls] == [
        (r, 10, p, side) for r in (11, 12, 13, 14) for p in (0, 32)
        for side in ("real", "null")
    ] + [(11, 16, 0, side) for side in ("real", "null")]
    for entry in visited:
        for side in ("real", "null"):
            state = entry[side]
            assert state["discovery_rank"] == 2080
            assert state["discovery_dimension"] == state["frozen_dimension"] == 0
            assert state["confirmation"] == "NOT_APPLICABLE"
            for phase in ("screen", "confirm"):
                assert state["stages"][phase] == {
                    "requested": 1, "collected": 0, "status": "NOT_APPLICABLE"}
            assert store.payloads[state["discovery_basis_ref"]] == {"discovery_basis": ()}
            assert store.payloads[state["basis_ref"]] == {
                "discovery_basis": (), "frozen_basis": ()}
    for entry in cells[9:]:
        assert entry["outcome"] == "NOT_STARTED_GATE"
        assert "real" not in entry and "null" not in entry
    assert result["summary"]["tiers"]["expanded"]["status"] == "GATED"
    assert len(result["summary"]["selections"]) == 4
    assert len(store.events) == len(collector.calls)
    assert len({name.split("-")[0] for name in store.payloads}) == 9




def test_empty_screen_skips_c_and_selection_uses_d_rank_if_ds_both_zero() -> None:
    collector = FakeCollector()
    collector.value = lambda cell, domain: (
        9 if domain[0] == "null" or domain[1] == "screen" else
        (8 if cell.active[0] == 32 else 0))
    result, collector, store = execute(recipe((11, 12)), collector)
    assert not any(d[1] == "confirm" for _, d in collector.calls)
    selection = result["summary"]["selections"][0]
    assert selection == {
        "rounds": 11, "tier": "discovery", "cell_indices": [0, 1],
        "dimensions": [0, 0], "ranks": [2079, 2078], "pattern_index": 1,
        "rule": "MAX_DS_DIM_THEN_ZERO_DS_MIN_D_RANK_THEN_PATTERN_0",
    }
    expanded = result["summary"]["cells"][4]
    assert expanded["tier"] == "expanded" and expanded["pattern_index"] == 1
    real = expanded["real"]
    assert real["discovery_dimension"] == 2 and real["frozen_dimension"] == 0
    assert store.payloads[real["basis_ref"]] == {
        "discovery_basis": (1, 2), "frozen_basis": ()}


@pytest.mark.parametrize("real_confirm,null_confirm", [(0, 1), (1, 1)])
def test_confirmation_and_null_cannot_change_selection(
    real_confirm: int, null_confirm: int,
) -> None:
    collector = FakeCollector()
    collector.value = lambda cell, domain: (
        (real_confirm if domain[0] == "real" else null_confirm)
        if domain[1] == "confirm" else 0)
    result, _, _ = execute(recipe((11,)), collector)
    assert result["summary"]["selections"][0]["pattern_index"] == 0
    assert result["summary"]["selections"][0]["dimensions"] == [1, 1]


def test_full_frozen_space_saved_before_c_failure_no_rescue_and_complete_null() -> None:
    collector = FakeCollector()
    store = FakeStore()
    collector.value = lambda cell, domain: (
        9 if domain[0] == "null" else (2 if domain[1] == "confirm" else 8))

    def inspect(cell: Cell, domain: tuple[str, str, int]) -> None:
        if domain[0:2] == ("real", "confirm"):
            entry = store.snapshots[-1]["summary"]["cells"]
            current = next(e for e in entry if e["cell"] == cell.to_dict())
            assert store.payloads[current["real"]["basis_ref"]]["frozen_basis"] == (1, 2)
    collector.hook = inspect
    result, collector, _ = execute(recipe((11, 12), confirm=2), collector, store=store)
    assert result["status"] == "SUCCEEDED"
    assert len(collector.calls) == 25  # 5 visited cells * (real D/S/2C + null D)
    assert [e["outcome"] for e in result["summary"]["cells"]] == ["REJECTED"] * 5 + [
        "NOT_STARTED_GATE"]


def test_null_alarm_stops_globally_before_other_discovery_cells() -> None:
    collector = FakeCollector()
    collector.value = lambda cell, domain: 0
    result, collector, _ = execute(recipe(), collector)
    assert result["status"] == "ALARM" and len(collector.calls) == 6
    cells = result["summary"]["cells"]
    assert cells[0]["outcome"] == "ALARM"
    assert all(entry["outcome"] == "NOT_STARTED" for entry in cells[1:])
    assert result["summary"]["selections"] == []


@pytest.mark.parametrize("phase", ["discovery", "screen", "confirm"])
@pytest.mark.parametrize("failure", ["resource", "identity", "error"])
def test_partial_side_never_becomes_negative(
    monkeypatch: pytest.MonkeyPatch, phase: str, failure: str,
) -> None:
    collector = FakeCollector()

    def stop(cell: Cell, domain: tuple[str, str, int]) -> None:
        if domain == ("real", phase, 0):
            if failure == "resource":
                raise Stopped("CPU_LIMIT")
            if failure == "identity":
                monkeypatch.setattr(fakes, "source_identity", lambda: {"changed.py": "1" * 64})
            else:
                raise RuntimeError("PRIVATE must not escape")
    collector.hook = stop
    result, collector, store = execute(recipe(), collector)
    assert result["status"] == ("FAILED" if failure == "error" else "STOPPED")
    entry = result["summary"]["cells"][0]
    assert entry["outcome"] == "INCOMPLETE"
    assert entry["real"]["confirmation"] == "PENDING"
    assert all(d[0] == "real" for _, d in collector.calls)
    assert result["summary"]["selections"] == []
    assert "PRIVATE" not in str(result)
    assert store.payloads[entry["real"]["row_refs"][phase]]["complete"] is False


@pytest.mark.parametrize("boundary", ["selection", "expanded", "higher_round"])
def test_fresh_guard_after_selection_and_at_tier_round_boundaries(boundary: str) -> None:
    collector, meter, store = FakeCollector(), FakeMeter(), FakeStore()
    triggered = False

    def persist(value: Mapping[str, Any]) -> None:
        nonlocal triggered
        summary = value["summary"]
        completed = [e for e in summary["cells"] if e["outcome"] == "CONFIRMED"]
        ready = ((boundary == "selection" and bool(summary.get("selections"))) or
                 (boundary == "expanded" and len(completed) == 8) or
                 (boundary == "higher_round" and len(completed) == 9))
        if ready:
            triggered = True

            def stop() -> None:
                raise Stopped("WALL_LIMIT")
            meter.hook = stop
    store.hook = persist
    result, collector, _ = execute(recipe(), collector, meter, store)
    assert triggered and result["status"] == "STOPPED"
    expected_visited = {"selection": 2, "expanded": 8, "higher_round": 9}[boundary]
    assert len({cell for cell, _ in collector.calls}) == expected_visited
    assert all(e["outcome"] not in ("REJECTED", "NOT_APPLICABLE", "NOT_STARTED_GATE")
               for e in result["summary"]["cells"])


def test_partial_null_does_not_gate_or_confirm_real() -> None:
    collector = FakeCollector()

    def stop(cell: Cell, domain: tuple[str, str, int]) -> None:
        if domain[0] == "null":
            raise Stopped("NULL_WORDS_LIMIT")
    collector.hook = stop
    result, _, _ = execute(recipe(), collector)
    entry = result["summary"]["cells"][0]
    assert result["status"] == "STOPPED"
    assert entry["real"]["confirmation"] == "PASSED"
    assert entry["outcome"] == "INCOMPLETE"
    assert result["summary"]["selections"] == []


def test_math_stop_not_empty_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    def stop(rows: Iterable[int], *, check: Callable[[], None]) -> tuple[int, ...]:
        check()
        raise Stopped("CPU_LIMIT")
    monkeypatch.setattr(stages, "kernel", stop)
    result, collector, _ = execute(recipe())
    entry = result["summary"]["cells"][0]
    assert result["status"] == "STOPPED" and len(collector.calls) == 1
    assert entry["outcome"] == "INCOMPLETE"
    assert "discovery_dimension" not in entry["real"]


def test_saved_journal_and_source_identity_with_static_uuid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uuid

    import cipher_lab.artifacts as artifacts

    monkeypatch.setattr(artifacts, "source_identity", lambda: SOURCE.copy())
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=1))
    store = RunStore.claim(tmp_path, "search", output_bytes=20000000)
    collector = FakeCollector()
    collector.value = lambda cell, domain: 9
    result = method.run_integral(recipe((11, 12)), collector, FakeMeter(), store)
    saved = read_saved(tmp_path, "search")
    assert result["status"] == saved["status"] == "SUCCEEDED"
    assert saved["source"] == SOURCE
    assert saved["summary"] == result["summary"]
    events = (tmp_path / "data/search/events.jsonl").read_text()
    assert events.count('"kind":"group"') == len(collector.calls)
    assert events.count('"row":9') == len(collector.calls)


def test_not_started_reasons_distinguish_gate_alarm_and_resource_stop() -> None:
    negative = FakeCollector()
    negative.value = lambda cell, domain: 9
    gated, _, _ = execute(recipe((11, 12)), negative)
    last = gated["summary"]["cells"][-1]
    assert last["outcome"] == "NOT_STARTED_GATE"
    assert last["not_started_reason"] == "REAL_NOT_CONFIRMED"
    alarm = FakeCollector()
    alarm.value = lambda cell, domain: 0
    alarmed, _, _ = execute(recipe((11, 12)), alarm)
    assert alarmed["summary"]["cells"][1]["not_started_reason"] == "NULL_ALARM"
    meter = FakeMeter()

    def stop() -> None:
        raise Stopped("CPU_LIMIT")
    meter.hook = stop
    stopped, collector, _ = execute(recipe((11, 12)), meter=meter)
    assert collector.calls == []
    assert all(entry["not_started_reason"] == "CPU_LIMIT"
               for entry in stopped["summary"]["cells"])


@pytest.mark.parametrize("discovery_active,expanded_active,patterns", [
    (2, 3, ((0, 1, 2), (0, 1, 3))),
    (10, 16, (tuple(range(16)), tuple(range(10)) + tuple(range(11, 17)))),
])
def test_shared_discovery_prefix_collects_independent_candidates_and_expands(
    discovery_active: int, expanded_active: int,
    patterns: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    r = config.SearchRecipe(1, "integral-search-v1", (11,), patterns,
                            discovery_active, expanded_active, StageSizes(2, 3, 2), 4)
    collector = FakeCollector()
    # Give the second logical candidate a larger real D/DS space. All null
    # candidates reach C and fail it, so all six side/phase domains are exercised.
    collector.value = lambda cell, domain: (
        (1 if domain[1] == "confirm" else 0) if domain[0] == "null" else
        (8 if domain[1] == "discovery" and domain[2] >= 2 else 0))
    result, collector, store = execute(r, collector)
    assert result["status"] == "SUCCEEDED"
    summary = result["summary"]
    assert len(collector.calls) == 42  # three logical cells * two sides * (2+3+2)
    assert len(set(collector.calls)) == 42
    assert [entry["outcome"] for entry in summary["cells"]] == ["CONFIRMED"] * 3
    assert summary["selections"][0]["dimensions"] == [1, 2]
    assert summary["selections"][0]["pattern_index"] == 1
    assert summary["cells"][2]["cell"]["active"] == list(patterns[1])
    assert len(summary["cells"][2]["cell"]["active"]) == expanded_active
    assert summary["tiers"]["expanded"]["status"] == "COMPLETED"
    for cell_index in range(3):
        for side in ("real", "null"):
            for phase, size in (("discovery", 2), ("screen", 3), ("confirm", 2)):
                events = [event["summary"] for event in store.events
                          if event["summary"]["cell_index"] == cell_index
                          and event["summary"]["side"] == side
                          and event["summary"]["phase"] == phase]
                assert [event["index"] for event in events] == list(range(size))
                assert [event["domain_index"] for event in events] == list(
                    range(cell_index * size, (cell_index + 1) * size))
                assert store.payloads[f"c{cell_index}-{side}-{phase}"]["complete"] is True
    assert collector.calls[0][0] == collector.calls[14][0]
    assert collector.calls[0][1] == ("real", "discovery", 0)
    assert collector.calls[14][1] == ("real", "discovery", 2)




def test_expanded_alarm_stops_all_higher_rounds_without_marking_gate() -> None:
    collector = FakeCollector()
    collector.value = lambda cell, domain: (
        9 if domain[0] == "null" and len(cell.active) == 10 else 0)
    result, collector, _ = execute(recipe(), collector)
    assert result["status"] == "ALARM"
    assert len(collector.calls) == 38  # eight (real D/S/C + null D), one full pair
    assert len(result["summary"]["selections"]) == 4
    assert result["summary"]["cells"][8]["outcome"] == "ALARM"
    assert all(e["outcome"] == "NOT_STARTED" and e["not_started_reason"] == "NULL_ALARM"
               for e in result["summary"]["cells"][9:])


def test_all_confirmed_expanded_cells_follow_per_round_selected_pattern_only() -> None:
    collector = FakeCollector()
    collector.value = lambda cell, domain: (
        9 if domain[0] == "null" else
        (8 if domain[1] == "discovery" and
         ((cell.rounds % 2 == 1) == (cell.active[0] == 32)) else 0))
    result, collector, _ = execute(recipe(), collector)
    assert result["status"] == "SUCCEEDED"
    summary = result["summary"]
    assert [s["pattern_index"] for s in summary["selections"]] == [1, 0, 1, 0]
    assert [e["pattern_index"] for e in summary["cells"][8:]] == [1, 0, 1, 0]
    assert all(e["outcome"] == "CONFIRMED" for e in summary["cells"])
    assert summary["tiers"] == {
        "discovery": {"status": "COMPLETED"}, "expanded": {"status": "COMPLETED"}}
    assert len({cell for cell, _ in collector.calls}) == 12


def test_high_empty_18_still_starts_only_selected_20_with_same_domains_and_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities: list[dict[str, str]] = []
    def identity() -> dict[str, str]:
        identities.append(SOURCE.copy())
        return SOURCE.copy()
    monkeypatch.setattr(fakes, "source_identity", identity)
    r = config.SearchRecipe(1, "integral-search-v1", (11,),
                            (tuple(range(20)), tuple(range(32, 52))),
                            18, 20, StageSizes(1, 1, 1), 65536)
    collector = FakeCollector()
    collector.value = lambda cell, domain: 9
    store = FakeStore()
    result, collector, used_store = execute(r, collector, store=store)
    assert used_store is store and store.finalizations == 1
    assert result["status"] == "SUCCEEDED"
    assert [(len(cell.active), cell.active[0], domain) for cell, domain in collector.calls] == [
        (18, 0, ("real", "discovery", 0)), (18, 0, ("null", "discovery", 0)),
        (18, 32, ("real", "discovery", 1)), (18, 32, ("null", "discovery", 1)),
        (20, 0, ("real", "discovery", 2)), (20, 0, ("null", "discovery", 2)),
    ]
    summary = result["summary"]
    assert len(summary["cells"]) == 3
    assert summary["selections"][0]["pattern_index"] == 0
    assert summary["selections"][0]["dimensions"] == [0, 0]
    assert summary["selections"][0]["ranks"] == [2080, 2080]
    assert all(entry["outcome"] == "NOT_APPLICABLE" for entry in summary["cells"])
    assert len(store.events) == 6 and len(set(collector.calls)) == 6
    assert len(identities) > len(collector.calls)
    assert all(identity == store.source_snapshot() for identity in identities)
    assert all(snapshot["recipe"] == r.to_dict() for snapshot in store.snapshots)

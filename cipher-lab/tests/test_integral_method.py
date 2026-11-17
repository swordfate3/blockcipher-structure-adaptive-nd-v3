"""Fixed-flow tests: artificial packed rows only; no cipher or scientific RNG.

Most flow cases substitute a small *relation* basis at the narrow math boundary
(the real 2080-bit kernel/intersection is independently tested in math tests).
One full-rank artificial-row case exercises the actual mathematics end to end.
"""
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

import cipher_lab.artifacts as artifacts
import cipher_lab.integral.method as method
import cipher_lab.integral.stages as stages
from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.config import Cell, Policy, Recipe, StageSizes
from cipher_lab.execution import Clocks, Meter, Stopped


def recipe(cells: int = 1, discovery: int = 1, confirm: int = 1) -> Recipe:
    return Recipe(1, "integral-empirical-v1", tuple(Cell(i + 1, (0, 1)) for i in range(cells)),
                  StageSizes(discovery, 1, confirm), 4)


def policy() -> Policy:
    return Policy(2, 100.0, 100.0, "2099-01-01T00:00:00Z", 100000, 100000,
                  1000000, 20000000, 0, 0)


class FakeCollector:
    def __init__(self, values: Mapping[tuple[str, str], tuple[int, ...]] | None = None) -> None:
        self.values = values or {}
        self.calls: list[tuple[Cell, tuple[str, str, int]]] = []
        self.completed = 0
        self.hook: Callable[[Cell, tuple[str, str, int]], None] = lambda c, d: None

    def row(self, cell: Cell, domain: tuple[str, str, int]) -> int:
        self.calls.append((cell, domain))
        self.hook(cell, domain)
        value = self.values.get((domain[0], domain[1]), (0,))[domain[2]]
        self.completed += 1
        return value

    def counts(self) -> dict[str, int]:
        return {"attempted_groups": len(self.calls), "completed_groups": self.completed}


class FakeMeter:
    def __init__(self) -> None:
        self.checks = 0
        self.hook: Callable[[], None] = lambda: None

    def check(self) -> None:
        self.checks += 1
        self.hook()

    def snapshot(self) -> dict[str, object]:
        return {"wall_elapsed_seconds": float(self.checks), "cpu_elapsed_seconds": 0.0,
                "rss_bytes": 1}


def mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def cells(saved: Mapping[str, object]) -> list[dict[str, object]]:
    value = mapping(saved["summary"])["cells"]
    assert isinstance(value, list)
    return [mapping(v) for v in value]


def side(saved: Mapping[str, object], name: str, index: int = 0) -> dict[str, object]:
    return mapping(cells(saved)[index][name])


@pytest.fixture
def small_math(monkeypatch: pytest.MonkeyPatch) -> None:
    # An artificial one-dimensional relation; null is full rank by default.
    monkeypatch.setattr(stages, "kernel",
                        lambda rows, *, check=None: () if tuple(rows) == (9,) else (1,))
    monkeypatch.setattr(stages, "survivor_basis",
                        lambda discovery, screen, *, check=None: tuple(discovery))


def run(tmp_path: Path, r: Recipe, collector: FakeCollector, meter: FakeMeter | Meter
        ) -> tuple[dict[str, object], dict[str, object]]:
    store = RunStore.claim(tmp_path, "case", output_bytes=policy().output_bytes)
    store.write_snapshot({"policy": policy().to_dict()})
    result = method.run_integral(r, collector, meter, store)
    return result, read_saved(tmp_path, "case")


def quiet_null() -> dict[tuple[str, str], tuple[int, ...]]:
    return {("null", "discovery"): (9,)}


def test_freeze_precedes_c_and_screen_receives_relation_basis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_math: None,
) -> None:
    collector = FakeCollector(quiet_null())
    observed: list[tuple[int, ...]] = []

    def survivor(discovery: tuple[int, ...], screen: tuple[int, ...], *,
                 check: Callable[[], None] | None = None) -> tuple[int, ...]:
        observed.append(discovery)
        return discovery

    monkeypatch.setattr(stages, "survivor_basis", survivor)

    def inspect(cell: Cell, domain: tuple[str, str, int]) -> None:
        if domain == ("real", "confirm", 0):
            saved = read_saved(tmp_path, "case")
            ref = side(saved, "real")["basis_ref"]
            assert mapping(saved["payloads"])[str(ref)] == {
                "discovery_basis": [1], "frozen_basis": [1]}

    collector.hook = inspect
    result, saved = run(tmp_path, recipe(), collector, FakeMeter())
    assert result["status"] == saved["status"] == "SUCCEEDED"
    assert observed == [(1,), ()]
    assert [d for _, d in collector.calls] == [
        ("real", "discovery", 0), ("real", "screen", 0), ("real", "confirm", 0),
        ("null", "discovery", 0), ("null", "screen", 0)]
    assert side(saved, "real")["confirmation"] == "PASSED"
    assert side(saved, "null")["confirmation"] == "NOT_APPLICABLE"
    assert saved["policy"] == policy().to_dict()
    summary = mapping(saved["summary"])
    assert summary["backend_identity"] == "UNVERIFIED"
    assert summary["scientific_validation"] is False


def test_confirm_failure_is_whole_cell_no_rescue_and_next_cell_runs(
    tmp_path: Path, small_math: None,
) -> None:
    collector = FakeCollector({**quiet_null(), ("real", "confirm"): (1, 0)})
    _, saved = run(tmp_path, recipe(cells=2, confirm=2), collector, FakeMeter())
    assert saved["status"] == "SUCCEEDED"  # Scientific rejection is not execution failure.
    assert len(cells(saved)) == 2
    assert all(cell["outcome"] == "REJECTED" for cell in cells(saved))
    assert side(saved, "real")["confirmation"] == "FAILED"
    assert sum(d[1] == "confirm" for _, d in collector.calls) == 4
    payload = mapping(saved["payloads"])[str(side(saved, "real")["basis_ref"])]
    assert mapping(payload)["frozen_basis"] == [1]


def test_nonempty_null_pass_is_alarm_stops_new_cells(tmp_path: Path, small_math: None) -> None:
    collector = FakeCollector()
    result, saved = run(tmp_path, recipe(cells=2), collector, FakeMeter())
    assert result["status"] == saved["status"] == "ALARM"
    assert len(cells(saved)) == 1
    assert side(saved, "null")["confirmation"] == "PASSED"
    assert len(collector.calls) == 6


def test_empty_real_and_null_spaces_are_not_applicable(
    tmp_path: Path, small_math: None,
) -> None:
    collector = FakeCollector({("real", "discovery"): (9,), **quiet_null()})
    _, saved = run(tmp_path, recipe(), collector, FakeMeter())
    assert all(side(saved, s)["confirmation"] == "NOT_APPLICABLE" for s in ("real", "null"))
    assert cells(saved)[0]["outcome"] == "NOT_APPLICABLE"
    assert not any(d[1] == "confirm" for _, d in collector.calls)
    for s in ("real", "null"):
        stage = mapping(mapping(side(saved, s)["stages"])["confirm"])
        assert stage == {"requested": 1, "collected": 0, "status": "NOT_APPLICABLE"}


@pytest.mark.parametrize("failure", ["stop", "private_exception"])
def test_partial_stage_rows_and_attempts_survive_failure(
    tmp_path: Path, failure: str, small_math: None,
) -> None:
    collector = FakeCollector({("real", "discovery"): (0, 0, 0)})

    def fail(cell: Cell, domain: tuple[str, str, int]) -> None:
        if domain[2] == 1:
            if failure == "stop":
                raise Stopped("ORACLE_FAILED")
            raise RuntimeError("PRIVATE canary must not be saved")

    collector.hook = fail
    _, saved = run(tmp_path, recipe(discovery=3), collector, FakeMeter())
    assert saved["status"] == ("STOPPED" if failure == "stop" else "FAILED")
    assert saved["counts"] == {"attempted_groups": 2, "completed_groups": 1}
    ref = mapping(side(saved, "real")["row_refs"])["discovery"]
    assert mapping(saved["payloads"])[str(ref)] == {"rows": [0], "complete": False}
    events = (tmp_path / "data/case/events.jsonl").read_text()
    assert '"row":0' in events
    assert "PRIVATE" not in str(saved) + events
    assert len(collector.calls) == 2


@pytest.mark.parametrize("when", ["row", "freeze"])
@pytest.mark.parametrize("identity", ["recipe", "source"])
def test_identity_change_stops_before_more_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_math: None,
    when: str, identity: str,
) -> None:
    r = recipe(cells=2)
    collector = FakeCollector(quiet_null())
    identity_value = {"cipher_lab/fake.py": "0" * 64}
    monkeypatch.setattr(artifacts, "source_identity", lambda: identity_value.copy())

    def mutate() -> None:
        if identity == "recipe":
            object.__setattr__(r, "chunk_points", 8)
        else:
            identity_value["cipher_lab/fake.py"] = "1" * 64

    if when == "row":
        collector.hook = lambda c, d: mutate()
    store = RunStore.claim(tmp_path, "case", output_bytes=policy().output_bytes)
    if when == "freeze":
        original = store.write_payload

        def write(name: str, value: object) -> str:
            ref = original(name, value)
            if name.endswith("basis"):
                mutate()
            return ref

        monkeypatch.setattr(store, "write_payload", write)
    method.run_integral(r, collector, FakeMeter(), store)
    saved = read_saved(tmp_path, "case")
    assert saved["status"] == "STOPPED"
    assert mapping(saved["summary"])["reason"] == "IDENTITY_CHANGED"
    assert len(collector.calls) == (1 if when == "row" else 2)
    if when == "freeze":
        assert side(saved, "real")["basis_ref"] in mapping(saved["payloads"])


def test_claim_source_change_before_method_stops_without_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_math: None,
) -> None:
    source_a = {"cipher_lab/fake.py": "0" * 64}
    source_b = {"cipher_lab/fake.py": "1" * 64}
    monkeypatch.setattr(artifacts, "source_identity", lambda: source_a.copy())
    store = RunStore.claim(tmp_path, "case", output_bytes=policy().output_bytes)
    monkeypatch.setattr(store, "current_source_identity", lambda: source_b.copy())
    collector = FakeCollector(quiet_null())
    result = method.run_integral(recipe(), collector, FakeMeter(), store)
    assert collector.calls == []
    saved = read_saved(tmp_path, "case")
    assert result["status"] == saved["status"] == "STOPPED"
    assert mapping(saved["summary"])["reason"] == "IDENTITY_CHANGED"
    assert saved["source"] == source_a
    assert cells(saved) == []
    assert "SUCCEEDED" not in (tmp_path / "data/case/run.json").read_text()


def test_fake_store_claim_source_is_method_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_a = {"cipher_lab/fake.py": "0" * 64}

    class FakeStore:
        def __init__(self) -> None:
            self.final: dict[str, object] = {}

        def current_source_identity(self) -> dict[str, str]:
            return {"cipher_lab/fake.py": "1" * 64}

        def source_snapshot(self) -> dict[str, str]:
            return source_a.copy()

        def write_snapshot(self, value: Mapping[str, object]) -> None:
            pass

        def append_event(self, value: Mapping[str, object]) -> None:
            pytest.fail("identity mismatch must prevent group events")

        def write_payload(self, name: str, value: object) -> str:
            pytest.fail("identity mismatch must prevent stage payloads")

        def finalize(self, value: Mapping[str, object]) -> None:
            self.final = dict(value)

    store = FakeStore()
    collector = FakeCollector()
    result = method.run_integral(recipe(), collector, FakeMeter(), store)
    assert collector.calls == []
    assert result == store.final
    assert result["status"] == "STOPPED"
    assert mapping(result["summary"])["reason"] == "IDENTITY_CHANGED"


def test_fresh_deadline_check_after_persistence_prevents_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_math: None,
) -> None:
    clock = [0.0]
    meter = Meter(policy(), Clocks(lambda: clock[0], lambda: 0.0,
                                  lambda: datetime(2098, 1, 1, tzinfo=UTC)), lambda: 1)
    collector = FakeCollector(quiet_null())
    store = RunStore.claim(tmp_path, "case", output_bytes=policy().output_bytes)
    original = store.write_snapshot

    def write(value: Mapping[str, object]) -> None:
        original(value)
        if len(collector.calls) == 5:
            clock[0] = 101.0

    monkeypatch.setattr(store, "write_snapshot", write)
    method.run_integral(recipe(cells=2), collector, meter, store)
    saved = read_saved(tmp_path, "case")
    assert saved["status"] == "STOPPED"
    assert mapping(saved["summary"])["reason"] == "WALL_LIMIT"
    assert len(collector.calls) == 5
    assert cells(saved)[0]["outcome"] != "CONFIRMED"


def test_actual_2080_math_with_artificial_full_rank_rows(tmp_path: Path) -> None:
    units = tuple(1 << bit for bit in range(2080))
    collector = FakeCollector({("real", "discovery"): units,
                               ("null", "discovery"): units})
    _, saved = run(tmp_path, recipe(discovery=2080), collector, FakeMeter())
    assert saved["status"] == "SUCCEEDED"
    assert side(saved, "real")["discovery_dimension"] == 0
    assert side(saved, "null")["confirmation"] == "NOT_APPLICABLE"
    assert len(list((tmp_path / "data/case/payload").iterdir())) == 8
    assert len(collector.calls) == 4162


def test_discovery_basis_retained_when_screen_collection_fails(
    tmp_path: Path, small_math: None,
) -> None:
    collector = FakeCollector(quiet_null())

    def fail(cell: Cell, domain: tuple[str, str, int]) -> None:
        if domain == ("real", "screen", 0):
            raise Stopped("ORACLE_FAILED")

    collector.hook = fail
    _, saved = run(tmp_path, recipe(), collector, FakeMeter())
    assert saved["status"] == "STOPPED"
    state = side(saved, "real")
    assert mapping(saved["payloads"])[str(state["discovery_basis_ref"])] == {
        "discovery_basis": [1]}


def test_event_storage_failure_never_calls_sealed_owner_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_math: None,
) -> None:
    from cipher_lab.artifacts import ArtifactError

    collector = FakeCollector(quiet_null())
    store = RunStore.claim(tmp_path, "case", output_bytes=policy().output_bytes)
    writes: list[str] = []
    original = store.write_payload

    def fail(value: Mapping[str, object]) -> None:
        raise ArtifactError("OUTPUT_LIMIT")

    def payload(name: str, value: object) -> str:
        writes.append(name)
        return original(name, value)

    monkeypatch.setattr(store, "append_event", fail)
    monkeypatch.setattr(store, "write_payload", payload)
    with pytest.raises(ArtifactError, match="^OUTPUT_LIMIT$"):
        method.run_integral(recipe(), collector, FakeMeter(), store)
    assert writes == []
    assert len(collector.calls) == 1


def test_source_scan_failure_finalizes_without_more_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_math: None,
) -> None:
    from cipher_lab.artifacts import ArtifactError

    collector = FakeCollector(quiet_null())

    def unavailable() -> dict[str, str]:
        raise ArtifactError("SOURCE_UNAVAILABLE")

    def fail_scan(cell: Cell, domain: tuple[str, str, int]) -> None:
        monkeypatch.setattr(artifacts, "source_identity", unavailable)

    collector.hook = fail_scan
    _, saved = run(tmp_path, recipe(), collector, FakeMeter())
    assert saved["status"] == "STOPPED"
    assert len(collector.calls) == 1
    assert mapping(saved["summary"])["reason"] == "IDENTITY_CHANGED"


def test_engineering_control_failure_never_requests_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = FakeCollector()

    def broken() -> None:
        raise RuntimeError("PRIVATE control error")

    monkeypatch.setattr(method, "_engineering_control", broken)
    _, saved = run(tmp_path, recipe(), collector, FakeMeter())
    assert saved["status"] == "FAILED"
    assert collector.calls == []
    assert "PRIVATE" not in str(saved)


def test_budget_stop_before_first_group(tmp_path: Path) -> None:
    meter = FakeMeter()

    def stop() -> None:
        raise Stopped("RSS_LIMIT")

    meter.hook = stop
    collector = FakeCollector()
    _, saved = run(tmp_path, recipe(), collector, meter)
    assert saved["status"] == "STOPPED"
    assert collector.calls == []


def test_all_frozen_vectors_checked_even_nonrepresentative_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = tuple(1 << bit for bit in range(8))
    monkeypatch.setattr(stages, "kernel",
                        lambda rows, *, check=None: frozen if tuple(rows) == (0,) else ())
    monkeypatch.setattr(stages, "survivor_basis",
                        lambda discovery, screen, *, check=None: tuple(discovery))
    collector = FakeCollector({**quiet_null(), ("real", "confirm"): (128,)})
    _, saved = run(tmp_path, recipe(), collector, FakeMeter())
    assert cells(saved)[0]["outcome"] == "REJECTED"
    state = side(saved, "real")
    payload = mapping(mapping(saved["payloads"])[str(state["basis_ref"])])
    assert payload["frozen_basis"] == list(frozen)
    assert state["frozen_dimension"] == 8


def test_stopped_during_c_retains_complete_frozen_space_and_partial_c(
    tmp_path: Path, small_math: None,
) -> None:
    collector = FakeCollector({**quiet_null(), ("real", "confirm"): (0, 0)})

    def stop(cell: Cell, domain: tuple[str, str, int]) -> None:
        if domain == ("real", "confirm", 1):
            raise Stopped("CPU_LIMIT")

    collector.hook = stop
    _, saved = run(tmp_path, recipe(confirm=2), collector, FakeMeter())
    assert saved["status"] == "STOPPED"
    state = side(saved, "real")
    assert mapping(mapping(saved["payloads"])[str(state["basis_ref"])])["frozen_basis"] == [1]
    ref = mapping(state["row_refs"])["confirm"]
    assert mapping(saved["payloads"])[str(ref)] == {"rows": [0], "complete": False}
    assert not any(d[0] == "null" for _, d in collector.calls)


def test_stop_before_new_cell_preserves_previously_completed_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_math: None,
) -> None:
    collector = FakeCollector(quiet_null())
    meter = FakeMeter()
    store = RunStore.claim(tmp_path, "case", output_bytes=policy().output_bytes)
    original = store.write_snapshot
    observed: list[object] = []

    def observe(value: Mapping[str, object]) -> None:
        # Retain the caller's public summary reference only, never store internals.
        observed[:] = [mapping(value["summary"])["cells"]]
        original(value)

    def stop_before_next_cell() -> None:
        if observed and isinstance(observed[0], list) and observed[0]:
            if mapping(observed[0][0]).get("outcome") == "CONFIRMED":
                raise Stopped("CPU_LIMIT")

    monkeypatch.setattr(store, "write_snapshot", observe)
    meter.hook = stop_before_next_cell
    method.run_integral(recipe(cells=2), collector, meter, store)
    saved = read_saved(tmp_path, "case")
    assert saved["status"] == "STOPPED"
    assert cells(saved)[0]["outcome"] == "CONFIRMED"
    assert len(collector.calls) == 5


@pytest.mark.parametrize("backend", ["demo", "simon", "UNVERIFIED"])
def test_explicit_backend_identity_is_saved_before_collection_and_terminal(
    tmp_path: Path, small_math: None, backend: str,
) -> None:
    collector = FakeCollector(quiet_null())
    store = RunStore.claim(tmp_path, "backend", output_bytes=policy().output_bytes)

    def inspect(cell: Cell, domain: tuple[str, str, int]) -> None:
        saved = read_saved(tmp_path, "backend")
        summary = mapping(saved["summary"])
        assert summary["backend_identity"] == backend
        assert summary["scientific_validation"] is False
    collector.hook = inspect
    method.run_integral(recipe(), collector, FakeMeter(), store,
                        backend_identity=cast("method.BackendIdentity", backend))
    saved = read_saved(tmp_path, "backend")
    assert mapping(saved["summary"])["backend_identity"] == backend
    assert mapping(saved["summary"])["scientific_validation"] is False


@pytest.mark.parametrize("invalid", [None, True, 1, [], {}, "engineering", "SIMON", ""])
def test_backend_identity_rejects_invalid_input_before_scientific_or_store_calls(
    invalid: object,
) -> None:
    collector = FakeCollector()
    meter = FakeMeter()
    class NoStore:
        def __getattr__(self, name: str) -> object:
            pytest.fail("invalid backend must not access store")
    with pytest.raises(ValueError, match="backend_identity"):
        method.run_integral(recipe(), collector, meter, cast(stages.Store, NoStore()),
                            backend_identity=cast("method.BackendIdentity", invalid))
    assert collector.calls == [] and meter.checks == 0

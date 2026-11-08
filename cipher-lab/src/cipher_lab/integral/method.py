"""Fixed empirical method orchestration; no private samples or second unit ledger.

The caller prewrites public policy to the claimed store. This narrow entry
point records an explicit caller backend identity (default UNVERIFIED).
That metadata is never a certificate of a real cipher experiment.
Only collector.counts owns group/unit accounting; Meter.snapshot adds resources.
Rows are journaled immediately, then packed into one payload per stage (including
partial stages). K_D is saved before S; the freeze payload saves K_D and K_DS.
Storage failure is owned by RunStore and propagated, not retried on a sealed owner.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol

from cipher_lab.artifacts import ArtifactError, source_identity
from cipher_lab.config import Cell, Recipe
from cipher_lab.execution import Stopped
from cipher_lab.integral import linear
from cipher_lab.integral.linear import confirms, kernel, survivor_basis

BackendIdentity = Literal["demo", "simon", "UNVERIFIED"]


class Collector(Protocol):
    """Public aggregate-only boundary; also satisfied by CubeCollector."""

    def row(self, cell: Cell, domain: tuple[str, str, int]) -> int: ...
    def counts(self) -> dict[str, int]: ...


class Resources(Protocol):
    def check(self) -> None: ...
    def snapshot(self) -> dict[str, object]: ...


class Store(Protocol):
    def source_snapshot(self) -> dict[str, str]: ...
    def write_snapshot(self, value: Mapping[str, object]) -> None: ...
    def append_event(self, value: Mapping[str, object]) -> None: ...
    def write_payload(self, name: str, value: object) -> str: ...
    def finalize(self, value: Mapping[str, object]) -> None: ...


class _IdentityChanged(Exception):
    pass


def _engineering_control() -> None:
    """Deterministic small mathematics only: never a scientific positive control."""
    basis = linear.kernel((1,), width=2)
    if (basis != (2,) or not linear.confirms(basis, (1, 0))
            or linear.confirms(basis, (2,))
            or linear.kernel((2, 1), width=2) != ()
            or not linear.confirms((), (3,))):
        raise RuntimeError  # fixed public failure is selected by the method boundary


def run_integral(recipe: Recipe, collector: Collector, meter: Resources,
                 store: Store, *,
                 backend_identity: BackendIdentity = "UNVERIFIED") -> dict[str, object]:
    """Run real D/S/freeze/C then independent null D/S/freeze/C in cell order.

    SUCCEEDED means execution completed, including scientifically rejected cells.
    Empty K_DS is NOT_APPLICABLE even though mathematical confirms((), C) is true.
    Any nonempty null passing *all* C is ALARM. Real C failure rejects the whole
    frozen space, never selects surviving representatives. Resource and identity
    checks are cooperative; an atomic deadline guarantee during OS writes is not
    possible. A fresh check precedes each scientific call and terminal decision.
    """
    if type(backend_identity) is not str or backend_identity not in ("demo", "simon", "UNVERIFIED"):
        raise ValueError("backend_identity must be demo, simon or UNVERIFIED")
    cells: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "method": "integral-empirical-v1", "backend_identity": backend_identity,
        "scientific_validation": False, "engineering_control": "PENDING",
        "cells": cells,
    }
    frozen_recipe = recipe.to_dict()
    frozen_source: dict[str, str] | None = None
    current: dict[str, object] | None = None

    def snapshot() -> None:
        summary["meter"] = meter.snapshot()
        store.write_snapshot({"recipe": frozen_recipe, "counts": collector.counts(),
                              "summary": summary})

    def identity() -> dict[str, str]:
        try:
            return source_identity()
        except ArtifactError:
            # Source inspection failure is not a failed/sealed storage owner.
            raise _IdentityChanged from None

    def guard() -> None:
        meter.check()
        if recipe.to_dict() != frozen_recipe or identity() != frozen_source:
            raise _IdentityChanged
        meter.check()  # source inspection itself uses wall/CPU time, too

    def collect(cell: Cell, cell_index: int, side: str, phase: str, size: int,
                stages: dict[str, object], refs: dict[str, object]) -> tuple[int, ...]:
        rows: list[int] = []
        stage: dict[str, object] = {"requested": size, "collected": 0, "status": "RUNNING"}
        stages[phase] = stage
        complete = False
        storage_failed = False
        try:
            for index in range(size):
                guard()
                row = collector.row(cell, (side, phase, index))
                # The collector contract is a public packed aggregate, never points.
                if type(row) is not int or not 0 <= row < (1 << 2080):
                    raise ValueError
                rows.append(row)
                stage["collected"] = len(rows)
                store.append_event({"kind": "group", "counts": collector.counts(), "summary": {
                    "cell_index": cell_index, "side": side, "phase": phase,
                    "index": index, "row": row}})
                guard()  # completed aggregate is already durable if resources/identity changed
            complete = True
            return tuple(rows)
        except ArtifactError:
            storage_failed = True
            raise
        finally:
            # A storage error may seal the sole owner: never retry it in cleanup.
            if not storage_failed:
                # No per-group files: one stage payload, including completed rows.
                stage["status"] = "COLLECTED" if complete else "INCOMPLETE"
                refs[phase] = store.write_payload(f"c{cell_index}-{side}-{phase}",
                                                  {"rows": rows, "complete": complete})
                snapshot()

    def run_side(cell: Cell, index: int, side: str) -> str:
        assert current is not None
        stages: dict[str, object] = {}
        refs: dict[str, object] = {}
        state: dict[str, object] = {"stages": stages, "row_refs": refs,
                                    "confirmation": "PENDING"}
        current[side] = state
        d_rows = collect(cell, index, side, "discovery", recipe.groups.discovery, stages, refs)
        guard()
        discovery = kernel(d_rows)
        state["discovery_dimension"] = len(discovery)
        del d_rows
        guard()
        state["discovery_basis_ref"] = store.write_payload(f"c{index}-{side}-kd", {
            "discovery_basis": discovery})
        snapshot()
        guard()
        s_rows = collect(cell, index, side, "screen", recipe.groups.screen, stages, refs)
        guard()
        # First argument is the RELATION basis, never discovery constraint rows.
        frozen = survivor_basis(discovery, s_rows)
        del s_rows
        state["frozen_dimension"] = len(frozen)
        guard()
        state["basis_ref"] = store.write_payload(f"c{index}-{side}-basis", {
            "discovery_basis": discovery, "frozen_basis": frozen})
        snapshot()  # full canonical space frozen and referenced BEFORE any C request
        guard()
        if not frozen:
            stages["confirm"] = {"requested": recipe.groups.confirm, "collected": 0,
                                 "status": "NOT_APPLICABLE"}
            state["confirmation"] = "NOT_APPLICABLE"
        else:
            c_rows = collect(cell, index, side, "confirm", recipe.groups.confirm, stages, refs)
            guard()
            # Check ALL frozen relations and ALL confirmation rows, no filtering.
            state["confirmation"] = "PASSED" if confirms(frozen, c_rows) else "FAILED"
        snapshot()
        guard()  # include preceding payload/snapshot persistence time
        return str(state["confirmation"])

    status = "SUCCEEDED"
    try:
        frozen_source = store.source_snapshot()
        guard()  # compare claim identity before any collector access
        snapshot()
        guard()
        _engineering_control()
        summary["engineering_control"] = "PASSED_SYNTHETIC_ONLY"
        snapshot()
        for index, cell in enumerate(recipe.cells):
            guard()
            current = {"cell": cell.to_dict(), "outcome": "INCOMPLETE"}
            cells.append(current)
            real = run_side(cell, index, "real")
            null = run_side(cell, index, "null")
            # Persist evidence with pending cell outcome, then sample fresh resources.
            snapshot()
            guard()
            if null == "PASSED":
                current["outcome"] = "ALARM"
                status = "ALARM"
                break
            current["outcome"] = {"PASSED": "CONFIRMED", "FAILED": "REJECTED",
                                  "NOT_APPLICABLE": "NOT_APPLICABLE"}[real]
            current = None  # a later stop must not erase a completed prior cell
        # Persist all evidence and sample fresh clocks before a scientific terminal.
        snapshot()
        guard()
    except ArtifactError:
        # A failed storage owner is sealed; it alone attempts a failure record.
        raise
    except _IdentityChanged:
        status = "STOPPED"
        summary["reason"] = "IDENTITY_CHANGED"
    except Stopped as error:
        status = "STOPPED"
        summary["reason"] = error.reason
    except Exception:
        status = "FAILED"
        summary["reason"] = "METHOD_FAILED"
    if status in ("STOPPED", "FAILED") and current is not None:
        current["outcome"] = "INCOMPLETE"
    summary["meter"] = meter.snapshot()
    result: dict[str, object] = {"status": status, "recipe": frozen_recipe,
                                "counts": collector.counts(), "summary": summary}
    store.finalize(result)
    return result

"""Fixed aggregate orchestration; durable D/S/C primitives own stage persistence."""

from __future__ import annotations

from typing import Literal

from cipher_lab.artifacts import ArtifactError
from cipher_lab.config import Cell, Recipe
from cipher_lab.execution import Stopped
from cipher_lab.integral import linear
from cipher_lab.integral.stages import (
    Collector,
    IdentityChanged,
    Resources,
    StageExecutor,
    StageInput,
    Store,
)

BackendIdentity = Literal["demo", "simon", "UNVERIFIED"]


def _engineering_control() -> None:
    """Deterministic small mathematics only: never a scientific positive control."""
    basis = linear.kernel((1,), width=2)
    if (
        basis != (2,)
        or not linear.confirms(basis, (1, 0))
        or linear.confirms(basis, (2,))
        or linear.kernel((2, 1), width=2) != ()
        or not linear.confirms((), (3,))
    ):
        raise RuntimeError  # fixed public failure is selected by the method boundary


def run_integral(
    recipe: Recipe,
    collector: Collector,
    meter: Resources,
    store: Store,
    *,
    backend_identity: BackendIdentity = "UNVERIFIED",
) -> dict[str, object]:
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
        "method": recipe.method,
        "backend_identity": backend_identity,
        "scientific_validation": False,
        "engineering_control": "PENDING",
        "cells": cells,
    }
    frozen_recipe = recipe.to_dict()
    frozen_source: dict[str, str] | None = None
    current: dict[str, object] | None = None

    def snapshot() -> None:
        summary["meter"] = meter.snapshot()
        store.write_snapshot(
            {"recipe": frozen_recipe, "counts": collector.counts(), "summary": summary}
        )

    def identity() -> dict[str, str]:
        try:
            return store.current_source_identity()
        except ArtifactError:
            raise IdentityChanged from None

    def guard() -> None:
        meter.check()
        if recipe.to_dict() != frozen_recipe or identity() != frozen_source:
            raise IdentityChanged
        meter.check()

    executor = StageExecutor(collector, meter, store, snapshot, guard)

    def run_side(cell: Cell, index: int, side: str) -> str:
        assert current is not None
        state: dict[str, object] = {}
        current[side] = state
        discovery = executor.discover(cell, index, side, state, StageInput(recipe.groups.discovery))
        executor.save_discovery(index, side, state, discovery)
        return executor.freeze_confirm(
            cell,
            index,
            side,
            state,
            discovery,
            screen=StageInput(recipe.groups.screen),
            screen_requested=recipe.groups.screen,
            confirm=StageInput(recipe.groups.confirm),
        )

    status = "SUCCEEDED"
    try:
        frozen_source = store.source_snapshot()
        guard()
        snapshot()
        guard()
        _engineering_control()
        summary["engineering_control"] = "PASSED_SYNTHETIC_ONLY"
        snapshot()
        for index, cell in enumerate(recipe.cells):
            guard()
            assert cell is not None
            current = {"cell": cell.to_dict(), "outcome": "INCOMPLETE"}
            cells.append(current)
            assert cell is not None
            real = run_side(cell, index, "real")
            null = run_side(cell, index, "null")
            snapshot()
            guard()
            if null == "PASSED":
                current["outcome"] = "ALARM"
                status = "ALARM"
                break
            current["outcome"] = {
                "PASSED": "CONFIRMED",
                "FAILED": "REJECTED",
                "NOT_APPLICABLE": "NOT_APPLICABLE",
            }[real]
            current = None
        snapshot()
        guard()
    except ArtifactError:
        raise
    except IdentityChanged:
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
    result: dict[str, object] = {
        "status": status,
        "recipe": frozen_recipe,
        "counts": collector.counts(),
        "summary": summary,
    }
    store.finalize(result)
    return result

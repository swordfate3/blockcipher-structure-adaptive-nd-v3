"""Fixed aggregate orchestration; durable D/S/C primitives own stage persistence."""
from __future__ import annotations

from typing import Literal, cast

from cipher_lab.artifacts import ArtifactError
from cipher_lab.config import Cell
from cipher_lab.execution import Stopped
from cipher_lab.integral.method import _engineering_control
from cipher_lab.integral.stages import (
    Collector,
    IdentityChanged,
    Resources,
    StageExecutor,
    StageInput,
    Store,
)
from cipher_lab_plugins.integral_search.config import SearchRecipe

BackendIdentity = Literal["demo", "simon", "UNVERIFIED"]

def select_pattern(dimensions: tuple[int, int], ranks: tuple[int, int]) -> int:
    """Predeclared choice from real D/S only; positive DS ties never use rank."""
    if dimensions == (0, 0):
        return int(ranks[1] < ranks[0])
    return int(dimensions[1] > dimensions[0])


def run_integral(recipe: SearchRecipe, collector: Collector, meter: Resources,
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
    selections: list[dict[str, object]] = []
    tiers: dict[str, dict[str, object]] = {
        "discovery": {"status": "NOT_STARTED"}, "expanded": {"status": "NOT_STARTED"},
    }
    # One global index across tiers: payload names and journal references never reset.
    plan: list[tuple[Cell | None, str, int | None]] = []
    for rounds in recipe.rounds:
        for pattern_index, pattern in enumerate(recipe.patterns):
            cell = Cell(rounds, pattern[:recipe.discovery_active])
            plan.append((cell, "discovery", pattern_index))
            cells.append({"cell": cell.to_dict(), "tier": "discovery",
                          "pattern_index": pattern_index, "outcome": "NOT_STARTED"})
    for rounds in recipe.rounds:
        plan.append((None, "expanded", None))
        cells.append({"rounds": rounds, "tier": "expanded", "pattern_index": None,
                      "outcome": "NOT_STARTED"})
    summary: dict[str, object] = {
        "method": recipe.method, "backend_identity": backend_identity,
        "scientific_validation": False, "engineering_control": "PENDING",
        "cells": cells,
    }
    summary.update(selections=selections, tiers=tiers)
    frozen_recipe = recipe.to_dict()
    frozen_source: dict[str, str] | None = None
    current: dict[str, object] | None = None

    def snapshot() -> None:
        summary["meter"] = meter.snapshot()
        store.write_snapshot({"recipe": frozen_recipe, "counts": collector.counts(),
                              "summary": summary})

    def identity() -> dict[str, str]:
        try:
            return store.current_source_identity()
        except ArtifactError:
            # Source inspection failure is not a failed/sealed storage owner.
            raise IdentityChanged from None

    def guard() -> None:
        meter.check()
        if recipe.to_dict() != frozen_recipe or identity() != frozen_source:
            raise IdentityChanged
        meter.check()  # source inspection itself uses wall/CPU time, too

    executor = StageExecutor(collector, meter, store, snapshot, guard)

    def run_side(cell: Cell, index: int, side: str) -> str:
        assert current is not None
        state: dict[str, object] = {}
        current[side] = state
        discovery = executor.discover(cell, index, side, state, StageInput(
            recipe.groups.discovery, index * recipe.groups.discovery))
        state["discovery_rank"] = 2080 - len(discovery)
        executor.save_discovery(index, side, state, discovery)
        return executor.freeze_confirm(
            cell, index, side, state, discovery,
            screen=(StageInput(recipe.groups.screen, index * recipe.groups.screen)
                    if discovery else None), screen_requested=recipe.groups.screen,
            confirm=StageInput(recipe.groups.confirm, index * recipe.groups.confirm))

    status = "SUCCEEDED"
    try:
        frozen_source = store.source_snapshot()
        guard()  # compare claim identity before any collector access
        snapshot()
        guard()
        _engineering_control()
        summary["engineering_control"] = "PASSED_SYNTHETIC_ONLY"
        snapshot()
        for index, (cell, tier, pattern_index) in enumerate(plan):
            guard()  # also fresh at discovery -> expanded and expanded -> next round
            if tier == "expanded" and tiers[tier]["status"] == "GATED":
                continue  # recorded NOT_STARTED_GATE, never request a collector row
            current = cells[index]
            tiers[tier]["status"] = "RUNNING"
            if tier == "expanded":
                offset = index - 2 * len(recipe.rounds)
                selection = selections[offset]
                pattern_index = cast(int, selection["pattern_index"])
                cell = Cell(recipe.rounds[offset],
                            recipe.patterns[pattern_index][:recipe.expanded_active])
                current.update(cell=cell.to_dict(), pattern_index=pattern_index)
            current["outcome"] = "INCOMPLETE"
            assert cell is not None
            real = run_side(cell, index, "real")
            null = run_side(cell, index, "null")
            # Persist evidence with pending cell outcome, then sample fresh resources.
            snapshot()
            guard()
            if null == "PASSED":
                current["outcome"] = "ALARM"
                status = "ALARM"
                tiers[tier]["status"] = "ALARM"
                break
            current["outcome"] = {"PASSED": "CONFIRMED", "FAILED": "REJECTED",
                                  "NOT_APPLICABLE": "NOT_APPLICABLE"}[real]
            current = None  # a later stop must not erase a completed prior cell
            if tier == "discovery" and pattern_index == 1:
                # Read only the two real public D/S statistics, never C or null.
                left = cast(dict[str, object], cells[index - 1]["real"])
                right = cast(dict[str, object], cells[index]["real"])
                dimensions = (cast(int, left["frozen_dimension"]),
                              cast(int, right["frozen_dimension"]))
                ranks = (cast(int, left["discovery_rank"]),
                         cast(int, right["discovery_rank"]))
                chosen = select_pattern(dimensions, ranks)
                selections.append({
                    "rounds": cell.rounds, "tier": "discovery",
                    "cell_indices": [index - 1, index], "dimensions": list(dimensions),
                    "ranks": list(ranks), "pattern_index": chosen,
                    "rule": "MAX_DS_DIM_THEN_ZERO_DS_MIN_D_RANK_THEN_PATTERN_0",
                })
                expanded = cells[2 * len(recipe.rounds) + index // 2]
                expanded.update(pattern_index=chosen, cell=Cell(
                    cell.rounds, recipe.patterns[chosen][:recipe.expanded_active]).to_dict())
            if tier == "discovery" and index == 2 * len(recipe.rounds) - 1:
                tiers[tier]["status"] = "COMPLETED"
            if tier == "expanded":
                if real != "PASSED":
                    tiers[tier].update(status="GATED", gate_cell_index=index)
                    for pending in cells[index + 1:]:
                        pending.update(outcome="NOT_STARTED_GATE", gate_cell_index=index,
                                       not_started_reason="REAL_NOT_CONFIRMED")
                elif index == len(plan) - 1:
                    tiers[tier]["status"] = "COMPLETED"
            snapshot()  # selection and gate evidence are durable before proceeding
            guard()
        # Persist all evidence and sample fresh clocks before a scientific terminal.
        snapshot()
        guard()
    except ArtifactError:
        # A failed storage owner is sealed; it alone attempts a failure record.
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
    for pending in cells:
        if pending["outcome"] == "NOT_STARTED":
            pending["not_started_reason"] = (
                "NULL_ALARM" if status == "ALARM" else summary.get("reason", "NOT_REACHED"))
    if status in ("STOPPED", "FAILED"):
        for tier_state in tiers.values():
            if tier_state["status"] == "RUNNING":
                tier_state["status"] = status
    summary["meter"] = meter.snapshot()
    result: dict[str, object] = {"status": status, "recipe": frozen_recipe,
                                "counts": collector.counts(), "summary": summary}
    store.finalize(result)
    return result

"""Durable aggregate stages with explicit Cell and domain inputs.

No recipe, tier, pattern selection or method dispatch belongs here. Callers
choose whether S is requested after D. The full basis is frozen before C.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from cipher_lab.artifacts import ArtifactError
from cipher_lab.config import Cell
from cipher_lab.integral.linear import confirms, kernel, survivor_basis


class Collector(Protocol):
    """Public aggregate-only boundary; also satisfied by CubeCollector."""

    def row(self, cell: Cell, domain: tuple[str, str, int]) -> int: ...
    def counts(self) -> dict[str, int]: ...


class Resources(Protocol):
    def check(self) -> None: ...
    def snapshot(self) -> dict[str, object]: ...


class Store(Protocol):
    def current_source_identity(self) -> dict[str, str]: ...
    def source_snapshot(self) -> dict[str, str]: ...
    def write_snapshot(self, value: Mapping[str, object]) -> None: ...
    def append_event(self, value: Mapping[str, object]) -> None: ...
    def write_payload(self, name: str, value: object) -> str: ...
    def finalize(self, value: Mapping[str, object]) -> None: ...


class IdentityChanged(Exception):
    pass


@dataclass(frozen=True)
class StageInput:
    """Requested groups and explicit domain origin (None keeps local journal schema)."""

    size: int
    start: int | None = None


class StageExecutor:
    """Shared D/S/C persistence, not a workflow or experiment runner."""

    def __init__(
        self,
        collector: Collector,
        meter: Resources,
        store: Store,
        snapshot: Callable[[], None],
        guard: Callable[[], None],
    ) -> None:
        self.collector, self.meter, self.store = collector, meter, store
        self.snapshot, self.guard = snapshot, guard

    def collect(
        self,
        cell: Cell,
        cell_index: int,
        side: str,
        phase: str,
        stages: dict[str, object],
        refs: dict[str, object],
        domain: StageInput,
    ) -> tuple[int, ...]:
        size = domain.size
        rows: list[int] = []
        stage: dict[str, object] = {"requested": size, "collected": 0, "status": "RUNNING"}
        stages[phase] = stage
        complete = False
        storage_failed = False
        try:
            for index in range(size):
                self.guard()
                domain_index = index if domain.start is None else domain.start + index
                row = self.collector.row(cell, (side, phase, domain_index))
                # The collector contract is a public packed aggregate, never points.
                if type(row) is not int or not 0 <= row < (1 << 2080):
                    raise ValueError
                rows.append(row)
                stage["collected"] = len(rows)
                group: dict[str, object] = {
                    "cell_index": cell_index,
                    "side": side,
                    "phase": phase,
                    "index": index,
                    "row": row,
                }
                if domain.start is not None:
                    group["domain_index"] = domain_index
                self.store.append_event(
                    {"kind": "group", "counts": self.collector.counts(), "summary": group}
                )
                self.guard()  # completed aggregate is already durable if resources/identity changed
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
                refs[phase] = self.store.write_payload(
                    f"c{cell_index}-{side}-{phase}", {"rows": rows, "complete": complete}
                )
                self.snapshot()

    def discover(
        self, cell: Cell, index: int, side: str, state: dict[str, object], stage: StageInput
    ) -> tuple[int, ...]:
        stages: dict[str, object] = {}
        refs: dict[str, object] = {}
        state.update(stages=stages, row_refs=refs, confirmation="PENDING")
        d_rows = self.collect(cell, index, side, "discovery", stages, refs, stage)
        self.guard()
        discovery = kernel(d_rows, check=self.meter.check)
        state["discovery_dimension"] = len(discovery)
        return discovery

    def save_discovery(
        self, index: int, side: str, state: dict[str, object], discovery: tuple[int, ...]
    ) -> None:
        self.guard()
        state["discovery_basis_ref"] = self.store.write_payload(
            f"c{index}-{side}-kd", {"discovery_basis": discovery}
        )
        self.snapshot()
        self.guard()

    def freeze_confirm(
        self,
        cell: Cell,
        index: int,
        side: str,
        state: dict[str, object],
        discovery: tuple[int, ...],
        *,
        screen: StageInput | None,
        screen_requested: int,
        confirm: StageInput,
    ) -> str:
        stages = cast(dict[str, object], state["stages"])
        refs = cast(dict[str, object], state["row_refs"])
        frozen: tuple[int, ...]
        if screen is None:
            stages["screen"] = {
                "requested": screen_requested,
                "collected": 0,
                "status": "NOT_APPLICABLE",
            }
            frozen = ()
        else:
            s_rows = self.collect(cell, index, side, "screen", stages, refs, screen)
            self.guard()
            frozen = survivor_basis(discovery, s_rows, check=self.meter.check)
            del s_rows
        state["frozen_dimension"] = len(frozen)
        self.guard()
        state["basis_ref"] = self.store.write_payload(
            f"c{index}-{side}-basis", {"discovery_basis": discovery, "frozen_basis": frozen}
        )
        self.snapshot()
        self.guard()
        if not frozen:
            stages["confirm"] = {
                "requested": confirm.size,
                "collected": 0,
                "status": "NOT_APPLICABLE",
            }
            state["confirmation"] = "NOT_APPLICABLE"
        else:
            c_rows = self.collect(cell, index, side, "confirm", stages, refs, confirm)
            self.guard()
            state["confirmation"] = (
                "PASSED" if confirms(frozen, c_rows, check=self.meter.check) else "FAILED"
            )
        self.snapshot()
        self.guard()
        return str(state["confirmation"])

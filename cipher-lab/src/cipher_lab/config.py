"""Strict, public settings only; no clocks, collection, execution, or approval."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

from cipher_lab.speck_study_config import METHOD, SpeckStudyRecipe
from cipher_lab.training_config import (
    PredictionRecipe,
    SharedMRecipe,
    parse_prediction,
    parse_shared_m,
)

_MAX_INPUT_BYTES = 65536
_METHOD = "integral-empirical-v1"
_STAGES = {"discovery", "screen", "confirm"}


def _integer(value: object, name: str, low: int, high: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{name}: expected integer, not boolean")
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name}: outside supported range")
    return value


def _positive_seconds(value: object, name: str) -> None:
    if type(value) not in (int, float):
        raise ValueError(f"{name}: expected positive finite seconds")
    try:
        valid = math.isfinite(cast("int | float", value)) and cast("int | float", value) > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name}: expected positive finite seconds")


def _deadline(value: object) -> None:
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)", value
    ) is None:
        raise ValueError("deadline_utc: expected UTC timestamp with Z or +00:00")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        raise ValueError("deadline_utc: invalid UTC timestamp") from None
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("deadline_utc: expected UTC")


def _object(value: object, keys: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name}: missing or unknown fields")
    return cast("dict[str, object]", value)


@dataclass(frozen=True, slots=True)
class Cell:
    """MSB0 active positions in one 32-bit half of a 64-bit block."""

    rounds: int
    active: tuple[int, ...]

    def __post_init__(self) -> None:
        _integer(self.rounds, "rounds", 1, 44)
        if type(self.active) is not tuple or not 2 <= len(self.active) <= 16:
            raise ValueError("active: expected tuple of 2..16 positions")
        for bit in self.active:
            _integer(bit, "active position", 0, 63)
        if len(set(self.active)) != len(self.active):
            raise ValueError("active: repeated position")
        if len({bit // 32 for bit in self.active}) != 1:
            raise ValueError("active: positions must lie in one 32-bit half")

    def to_dict(self) -> dict[str, object]:
        return {"rounds": self.rounds, "active": list(self.active)}


@dataclass(frozen=True, slots=True)
class StageSizes:
    discovery: int
    screen: int
    confirm: int

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            _integer(value, name, 1, 1000000)

    def to_dict(self) -> dict[str, object]:
        return {"discovery": self.discovery, "screen": self.screen, "confirm": self.confirm}


@dataclass(frozen=True, slots=True)
class Recipe:
    schema: int
    method: str
    cells: tuple[Cell, ...]
    groups: StageSizes
    chunk_points: int

    def __post_init__(self) -> None:
        _integer(self.schema, "schema", 1, 1)
        if self.method != _METHOD:
            raise ValueError("method: unsupported method")
        if type(self.cells) is not tuple or not 1 <= len(self.cells) <= 64:
            raise ValueError("cells: expected tuple of 1..64 cells")
        if any(not isinstance(cell, Cell) for cell in self.cells):
            raise ValueError("cells: expected Cell values")
        identities = {(cell.rounds, frozenset(cell.active)) for cell in self.cells}
        if len(identities) != len(self.cells):
            raise ValueError("cells: repeated rounds/active set")
        if not isinstance(self.groups, StageSizes):
            raise ValueError("groups: expected StageSizes")
        _integer(self.chunk_points, "chunk_points", 1, 65536)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "method": self.method,
            "cells": [cell.to_dict() for cell in self.cells],
            "groups": self.groups.to_dict(),
            "chunk_points": self.chunk_points,
        }


@dataclass(frozen=True, slots=True)
class Policy:
    schema: int
    wall_seconds: float
    cpu_seconds: float
    deadline_utc: str
    real_blocks: int
    null_words: int
    rss_stop_bytes: int
    output_bytes: int
    optimizer_updates: int
    forward_rows: int

    def __post_init__(self) -> None:
        _integer(self.schema, "schema", 2, 2)
        _positive_seconds(self.wall_seconds, "wall_seconds")
        _positive_seconds(self.cpu_seconds, "cpu_seconds")
        _deadline(self.deadline_utc)
        for name in ("real_blocks", "null_words", "rss_stop_bytes", "output_bytes",
                     "optimizer_updates", "forward_rows"):
            _integer(getattr(self, name), name, 0)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "optimizer_updates": self.optimizer_updates,
            "forward_rows": self.forward_rows,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "deadline_utc": self.deadline_utc,
            "real_blocks": self.real_blocks,
            "null_words": self.null_words,
            "rss_stop_bytes": self.rss_stop_bytes,
            "output_bytes": self.output_bytes,
        }


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON: repeated field")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError("JSON: non-finite number")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON: non-finite number")
    return parsed


def _load(path: Path) -> object:
    with path.open("rb") as stream:
        raw = stream.read(_MAX_INPUT_BYTES + 1)
    if len(raw) > _MAX_INPUT_BYTES:
        raise ValueError("JSON: input exceeds 64 KiB")
    try:
        value: object = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=_reject_constant, parse_float=_finite_float,
        )
    except (UnicodeError, ValueError, RecursionError):
        raise ValueError("JSON: invalid UTF-8, structure, fields, or number") from None
    return value


def load_recipe(path: Path) -> Recipe:
    return _parse_recipe(_load(path))


def load_experiment(
    path: Path,
) -> Recipe | PredictionRecipe | SharedMRecipe | SpeckStudyRecipe:
    """Read strict JSON exactly once, then explicitly dispatch by public method."""
    value = _load(path)
    if not isinstance(value, dict):
        raise ValueError("experiment: expected object")
    method = value.get("method")
    if method == METHOD:
        return SpeckStudyRecipe.parse(value)
    if method == _METHOD:
        return _parse_recipe(value)
    if method == "bit-prediction-v1":
        return parse_prediction(value)
    if method == "shared-m-four-task-v1":
        return parse_shared_m(value)
    raise ValueError("method: unsupported experiment")


def _parse_recipe(raw: object) -> Recipe:
    value = _object(raw, {"schema", "method", "cells", "groups", "chunk_points"}, "recipe")
    raw_cells = value["cells"]
    if not isinstance(raw_cells, list) or not 1 <= len(raw_cells) <= 64:
        raise ValueError("cells: expected array of 1..64 cells")
    cells: list[Cell] = []
    for raw_cell in raw_cells:
        cell = _object(raw_cell, {"rounds", "active"}, "cell")
        active = cell["active"]
        if not isinstance(active, list):
            raise ValueError("active: expected array")
        cells.append(Cell(
            _integer(cell["rounds"], "rounds", 1, 44),
            tuple(_integer(bit, "active position", 0, 63) for bit in active),
        ))
    groups = _object(value["groups"], _STAGES, "groups")
    method = value["method"]
    if not isinstance(method, str):
        raise ValueError("method: expected string")
    return Recipe(
        schema=_integer(value["schema"], "schema", 1, 1),
        method=method, cells=tuple(cells),
        groups=StageSizes(**{name: _integer(group, name, 1, 1000000)
                              for name, group in groups.items()}),
        chunk_points=_integer(value["chunk_points"], "chunk_points", 1, 65536),
    )


def load_policy(path: Path) -> Policy:
    value = _object(_load(path), {
        "wall_seconds", "cpu_seconds", "deadline_utc", "real_blocks", "null_words",
        "rss_stop_bytes", "output_bytes", "schema", "optimizer_updates", "forward_rows",
    }, "policy")
    _positive_seconds(value["wall_seconds"], "wall_seconds")
    _positive_seconds(value["cpu_seconds"], "cpu_seconds")
    _deadline(value["deadline_utc"])
    return Policy(
        schema=_integer(value["schema"], "schema", 2, 2),
        optimizer_updates=_integer(value["optimizer_updates"], "optimizer_updates", 0),
        forward_rows=_integer(value["forward_rows"], "forward_rows", 0),
        wall_seconds=float(cast("int | float", value["wall_seconds"])),
        cpu_seconds=float(cast("int | float", value["cpu_seconds"])),
        deadline_utc=cast("str", value["deadline_utc"]),
        real_blocks=_integer(value["real_blocks"], "real_blocks", 0),
        null_words=_integer(value["null_words"], "null_words", 0),
        rss_stop_bytes=_integer(value["rss_stop_bytes"], "rss_stop_bytes", 0),
        output_bytes=_integer(value["output_bytes"], "output_bytes", 0),
    )

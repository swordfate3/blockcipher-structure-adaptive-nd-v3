"""Strict two-tier search configuration, no execution."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cipher_lab.config import Cell, StageSizes, _integer, _object
from cipher_lab.config import _load as _load

_SEARCH_METHOD = "integral-search-v1"
_STAGES = {"discovery", "screen", "confirm"}

@dataclass(frozen=True, slots=True)
class SearchRecipe:
    """Two ordered public patterns and two fixed search tiers, not a workflow DSL."""

    schema: int
    method: str
    rounds: tuple[int, ...]
    patterns: tuple[tuple[int, ...], tuple[int, ...]]
    discovery_active: int
    expanded_active: int
    groups: StageSizes
    chunk_points: int

    def __post_init__(self) -> None:
        _integer(self.schema, "schema", 1, 1)
        if self.method != _SEARCH_METHOD:
            raise ValueError("method: unsupported search method")
        if type(self.rounds) is not tuple or not 1 <= len(self.rounds) <= 8:
            raise ValueError("rounds: expected tuple of 1..8 rounds")
        for rounds in self.rounds:
            _integer(rounds, "rounds", 1, 44)
        if tuple(sorted(set(self.rounds))) != self.rounds:
            raise ValueError("rounds: expected ascending unique rounds")
        if type(self.patterns) is not tuple or len(self.patterns) != 2:
            raise ValueError("patterns: expected exactly two ordered patterns")
        for pattern in self.patterns:
            Cell(1, pattern)
            if tuple(sorted(pattern)) != pattern:
                raise ValueError("patterns: expected ascending positions")
        if self.patterns[0] == self.patterns[1]:
            raise ValueError("patterns: repeated pattern")
        _integer(self.discovery_active, "discovery_active", 2)
        _integer(self.expanded_active, "expanded_active", self.discovery_active + 1,
                 min(map(len, self.patterns)))
        if not isinstance(self.groups, StageSizes):
            raise ValueError("groups: expected StageSizes")
        _integer(self.chunk_points, "chunk_points", 1, 65536)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema, "method": self.method, "rounds": list(self.rounds),
            "patterns": [list(pattern) for pattern in self.patterns],
            "discovery_active": self.discovery_active, "expanded_active": self.expanded_active,
            "groups": self.groups.to_dict(), "chunk_points": self.chunk_points,
        }


def _parse_search_recipe(raw: object) -> SearchRecipe:
    value = _object(raw, {
        "schema", "method", "rounds", "patterns", "discovery_active", "expanded_active",
        "groups", "chunk_points",
    }, "search recipe")
    rounds, patterns = value["rounds"], value["patterns"]
    if not isinstance(rounds, list):
        raise ValueError("rounds: expected array")
    if not isinstance(patterns, list) or len(patterns) != 2:
        raise ValueError("patterns: expected two arrays")
    parsed: list[tuple[int, ...]] = []
    for pattern in patterns:
        if not isinstance(pattern, list):
            raise ValueError("patterns: expected position arrays")
        parsed.append(tuple(_integer(bit, "pattern position", 0, 63) for bit in pattern))
    groups = _object(value["groups"], _STAGES, "groups")
    method = value["method"]
    if not isinstance(method, str):
        raise ValueError("method: expected string")
    return SearchRecipe(
        schema=_integer(value["schema"], "schema", 1, 1), method=method,
        rounds=tuple(_integer(r, "rounds", 1, 44) for r in rounds),
        patterns=(parsed[0], parsed[1]),
        discovery_active=_integer(value["discovery_active"], "discovery_active", 2),
        expanded_active=_integer(value["expanded_active"], "expanded_active", 3),
        groups=StageSizes(**{name: _integer(group, name, 1, 1000000)
                             for name, group in groups.items()}),
        chunk_points=_integer(value["chunk_points"], "chunk_points", 1, 65536),
    )


def load_experiment(path: Path) -> SearchRecipe:
    return _parse_search_recipe(_load(path))

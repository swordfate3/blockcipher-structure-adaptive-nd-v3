"""Search settings are public, strict and independent of scientific imports."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from cipher_lab.config import load_recipe
from cipher_lab_plugins.integral_search import config


def settings() -> dict[str, Any]:
    return {
        "schema": 1, "method": "integral-search-v1", "rounds": [11, 12, 13, 14],
        "patterns": [list(range(0, 32, 2)), list(range(32, 64, 2))],
        "discovery_active": 10, "expanded_active": 16,
        "groups": {"discovery": 2112, "screen": 320, "confirm": 256},
        "chunk_points": 65536,
    }


def test_search_recipe_roundtrip_frozen_and_reads_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "recipe.json"
    value = settings()
    path.write_text(json.dumps(value))
    original = config._load
    reads: list[Path] = []

    def load(path: Path) -> object:
        reads.append(path)
        return original(path)

    monkeypatch.setattr(config, "_load", load)
    recipe = config.load_experiment(path)
    assert hasattr(config, "SearchRecipe"), "missing typed search recipe"
    assert isinstance(recipe, config.SearchRecipe)
    assert reads == [path]
    assert recipe.to_dict() == value
    assert recipe.rounds == (11, 12, 13, 14)
    assert recipe.patterns[1] == tuple(range(32, 64, 2))
    with pytest.raises(FrozenInstanceError):
        setattr(recipe, "discovery_active", 2)
    detached = recipe.to_dict()
    detached["rounds"] = []
    assert recipe.to_dict() == value
    with pytest.raises(ValueError):
        load_recipe(path)
    invalid_changes: list[dict[str, Any]] = [
        {"rounds": [11]}, {"patterns": ([0, 1], [32, 33])},
        {"groups": {}}, {"discovery_active": True}, {"method": "old"},
    ]
    for changes in invalid_changes:
        with pytest.raises(ValueError):
            replace(recipe, **changes)


@pytest.mark.parametrize("field,bad", [
    ("schema", True), ("schema", 2), ("method", False),
    ("rounds", []), ("rounds", list(range(1, 10))), ("rounds", [12, 11]),
    ("rounds", [11, 11]), ("rounds", [0]), ("rounds", [45]),
    ("rounds", [True]), ("rounds", [1.0]), ("rounds", "11"),
    ("patterns", [[0, 1]]), ("patterns", [[0, 1]] * 3),
    ("patterns", [[0, 1], [0, 1]]), ("patterns", [[0, 32], [32, 33]]),
    ("patterns", [[1, 0], [32, 33]]), ("patterns", [[0, 0], [32, 33]]),
    ("patterns", [[-1, 1], [32, 33]]), ("patterns", [[0, 1], [63, 64]]),
    ("patterns", [[True, 1], [32, 33]]), ("patterns", [[0.0, 1], [32, 33]]),
    ("patterns", [[0], [32, 33]]), ("patterns", [list(range(21)), [32, 33]]),
    ("patterns", ["01", [32, 33]]), ("patterns", None),
    ("discovery_active", 1), ("discovery_active", 16), ("discovery_active", True),
    ("expanded_active", 10), ("expanded_active", 17), ("expanded_active", 16.0),
    ("chunk_points", 0), ("chunk_points", 65537), ("chunk_points", True),
    ("groups", {"discovery": 1, "screen": 1}),
    ("groups", {"discovery": True, "screen": 1, "confirm": 1}),
    ("groups", {"discovery": 0, "screen": 1, "confirm": 1}),
])
def test_invalid_search_fields(tmp_path: Path, field: str, bad: object) -> None:
    value = settings()
    value[field] = bad
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        config.load_experiment(path)


@pytest.mark.parametrize("target", ["recipe", "groups"])
@pytest.mark.parametrize("change", ["missing", "unknown"])
def test_search_exact_fields(tmp_path: Path, target: str, change: str) -> None:
    value = settings()
    node = value if target == "recipe" else value["groups"]
    if change == "missing":
        del node[next(iter(node))]
    else:
        node["hidden"] = 1
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        config.load_experiment(path)


@pytest.mark.parametrize("fragment", [
    '"schema": 1, "schema": 1', '"schema": NaN', '"schema": Infinity',
    '"schema": -Infinity', '"schema": 1e999',
])
def test_search_strict_json(tmp_path: Path, fragment: str) -> None:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(settings()).replace('"schema": 1', fragment))
    with pytest.raises(ValueError):
        config.load_experiment(path)


def test_search_short_patterns_and_round_chunk_boundaries(tmp_path: Path) -> None:
    value = settings()
    value.update(rounds=[1, 44], patterns=[[0, 1, 2], [4, 5, 6]],
                 discovery_active=2, expanded_active=3, chunk_points=1)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(value))
    assert config.load_experiment(path).to_dict() == value
    value["patterns"][1] = [4, 5]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        config.load_experiment(path)


def test_config_import_has_no_science(tmp_path: Path) -> None:
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(p in fullname.split('.') for p in
               ('numpy', 'torch', 'ciphers', 'integral', 'data', 'models', 'training')):
            raise AssertionError('scientific import: ' + fullname)
sys.meta_path.insert(0, Block())
from cipher_lab.config import StageSizes
from cipher_lab_plugins.integral_search.config import SearchRecipe
assert SearchRecipe(1, 'integral-search-v1', (1,), ((0,1,2),(32,33,34)),
                    2, 3, StageSizes(1,1,1), 1).method == 'integral-search-v1'
'''
    project = Path(__file__).resolve().parents[4]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(project / name)
                                               for name in ("src", "plugins")),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_high_search_roundtrip_and_typed_invariants(tmp_path: Path) -> None:
    value = settings()
    value.update(rounds=[11], patterns=[list(range(20)), list(range(32, 52))],
                 discovery_active=18, expanded_active=20)
    path = tmp_path / "high.json"
    path.write_text(json.dumps(value))
    recipe = config.load_experiment(path)
    assert isinstance(recipe, config.SearchRecipe)
    assert recipe.to_dict() == value
    assert recipe.patterns == (tuple(range(20)), tuple(range(32, 52)))
    invalid: list[dict[str, Any]] = [
        {"patterns": [tuple(range(20)), tuple(range(32, 52))]},
        {"patterns": (list(range(20)), tuple(range(32, 52)))},
        {"patterns": (tuple(reversed(range(20))), tuple(range(32, 52)))},
        {"patterns": (tuple(range(20)), tuple(range(20)))},
        {"patterns": (tuple(range(19)) + (32,), tuple(range(32, 52)))},
        {"patterns": ((True,) + tuple(range(1, 20)), tuple(range(32, 52)))},
        {"patterns": (tuple(range(21)), tuple(range(32, 53)))},
        {"discovery_active": True}, {"expanded_active": 20.0},
        {"expanded_active": 21}, {"expanded_active": 18},
    ]
    for changes in invalid:
        with pytest.raises(ValueError):
            replace(recipe, **changes)

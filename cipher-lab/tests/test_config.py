"""Synthetic settings only: these tests never encrypt, sample, or train."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from cipher_lab.cli import main
from cipher_lab.config import Cell, Policy, Recipe, StageSizes, load_policy, load_recipe


@pytest.fixture
def sandbox() -> Iterator[Path]:
    with TemporaryDirectory(prefix="cipher-lab-task1-") as directory:
        yield Path(directory)


def recipe_json() -> dict[str, Any]:
    return {
        "schema": 1,
        "method": "integral-empirical-v1",
        "cells": [{"rounds": 11, "active": [0, 1]}],
        "groups": {"discovery": 2, "screen": 3, "confirm": 4},
        "chunk_points": 3,
    }


def policy_json() -> dict[str, Any]:
    return {
        "schema": 2,
        "optimizer_updates": 0,
        "forward_rows": 0,
        "wall_seconds": 10.5,
        "cpu_seconds": 10,
        "deadline_utc": "2000-01-01T00:00:00Z",
        "real_blocks": 1000,
        "null_words": 1000,
        "rss_stop_bytes": 1048576,
        "output_bytes": 10000,
    }


def save(root: Path, value: object, name: str = "settings.json") -> Path:
    path = root / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("rounds", [1, 11, 44])
@pytest.mark.parametrize("active", [[0, 31], list(range(32, 42)), list(range(16))])
def test_supported_cells(sandbox: Path, rounds: int, active: list[int]) -> None:
    value = recipe_json()
    value["cells"] = [{"rounds": rounds, "active": active}]
    recipe = load_recipe(save(sandbox, value))
    assert recipe.cells == (Cell(rounds, tuple(active)),)
    assert recipe.to_dict() == value


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("rounds", 0), ("rounds", 45), ("rounds", True), ("rounds", 1.0),
        ("rounds", "11"), ("rounds", None),
        ("active", []), ("active", [0]), ("active", list(range(17))),
        ("active", [0, 32]), ("active", [31, 32]), ("active", [1, 1]),
        ("active", [-1, 0]), ("active", [62, 64]), ("active", [True, 2]),
        ("active", [0.0, 1]), ("active", ["0", 1]), ("active", "01"),
        ("active", None),
    ],
)
def test_invalid_cell(sandbox: Path, field: str, bad: object) -> None:
    value = recipe_json()
    value["cells"][0][field] = bad
    with pytest.raises(ValueError):
        load_recipe(save(sandbox, value))


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("schema", 0), ("schema", 2), ("schema", True), ("schema", 1.0),
        ("method", "old"), ("method", None), ("method", 1),
        ("chunk_points", 0), ("chunk_points", 65537), ("chunk_points", True),
        ("chunk_points", 1.0), ("chunk_points", "3"),
        ("cells", []), ("cells", [None]), ("cells", {}),
        ("groups", []), ("groups", None),
    ],
)
def test_invalid_recipe_field(sandbox: Path, field: str, bad: object) -> None:
    value = recipe_json()
    value[field] = bad
    with pytest.raises(ValueError):
        load_recipe(save(sandbox, value))


@pytest.mark.parametrize("stage", ["discovery", "screen", "confirm"])
@pytest.mark.parametrize("bad", [0, -1, 1000001, True, 1.5, "2", None])
def test_invalid_groups(sandbox: Path, stage: str, bad: object) -> None:
    value = recipe_json()
    value["groups"][stage] = bad
    with pytest.raises(ValueError):
        load_recipe(save(sandbox, value))


def test_cell_count_and_duplicate_set(sandbox: Path) -> None:
    value = recipe_json()
    value["cells"] = [{"rounds": r, "active": [0, 1]} for r in range(1, 45)] + [
        {"rounds": r, "active": [32, 33]} for r in range(1, 21)
    ]
    assert len(load_recipe(save(sandbox, value)).cells) == 64
    value["cells"].append({"rounds": 21, "active": [32, 33]})
    with pytest.raises(ValueError):
        load_recipe(save(sandbox, value))
    value["cells"] = [
        {"rounds": 11, "active": [0, 1]}, {"rounds": 11, "active": [1, 0]}
    ]
    with pytest.raises(ValueError):
        load_recipe(save(sandbox, value))


@pytest.mark.parametrize("chunk", [1, 65536])
@pytest.mark.parametrize("groups", [1, 1000000])
def test_recipe_boundaries(sandbox: Path, chunk: int, groups: int) -> None:
    value = recipe_json()
    value["chunk_points"] = chunk
    value["groups"] = dict.fromkeys(("discovery", "screen", "confirm"), groups)
    assert load_recipe(save(sandbox, value)).to_dict() == value


@pytest.mark.parametrize("target", ["recipe", "groups", "cell", "policy"])
@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_exact_fields(sandbox: Path, target: str, mutation: str) -> None:
    value = policy_json() if target == "policy" else recipe_json()
    node = value["groups"] if target == "groups" else (
        value["cells"][0] if target == "cell" else value
    )
    if mutation == "missing":
        del node[next(iter(node))]
    else:
        node["device"] = "cuda"
    with pytest.raises(ValueError):
        (load_policy if target == "policy" else load_recipe)(save(sandbox, value))


@pytest.mark.parametrize("field", ["wall_seconds", "cpu_seconds"])
@pytest.mark.parametrize("bad", [0, -1, True, "1", None, float("inf"), float("nan"), 10**400])
def test_invalid_time_limits(sandbox: Path, field: str, bad: object) -> None:
    value = policy_json()
    value[field] = bad
    with pytest.raises(ValueError):
        load_policy(save(sandbox, value))


@pytest.mark.parametrize("field", ["real_blocks", "null_words", "rss_stop_bytes", "output_bytes"])
@pytest.mark.parametrize("bad", [-1, True, 1.0, "1", None])
def test_invalid_budgets(sandbox: Path, field: str, bad: object) -> None:
    value = policy_json()
    value[field] = bad
    with pytest.raises(ValueError):
        load_policy(save(sandbox, value))


@pytest.mark.parametrize(
    "bad", ["2000-01-01", "2000-01-01T00:00:00", "2000-01-01T00:00:00+01:00",
            "2000-02-30T00:00:00Z", "tomorrow", None, 123],
)
def test_invalid_deadline(sandbox: Path, bad: object) -> None:
    value = policy_json()
    value["deadline_utc"] = bad
    with pytest.raises(ValueError):
        load_policy(save(sandbox, value))


@pytest.mark.parametrize("deadline", ["2000-01-01T00:00:00Z", "2099-01-01T00:00:00+00:00"])
def test_policy_zero_budgets_and_no_clock_binding(sandbox: Path, deadline: str) -> None:
    value = policy_json()
    value.update(dict.fromkeys(("real_blocks", "null_words", "rss_stop_bytes", "output_bytes"), 0))
    value["deadline_utc"] = deadline
    policy = load_policy(save(sandbox, value))
    assert policy.to_dict() == value


@pytest.mark.parametrize("loader", [load_recipe, load_policy])
@pytest.mark.parametrize(
    "raw", [b'[]', b'null', b'{', b'{} trailing', b'\xff',
            b'{"x":1,"x":2}', b'{"x":{"y":1,"y":2}}',
            b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}',
            b'{"x":1e999}', b'[' * 2000 + b']' * 2000],
)
def test_strict_json(sandbox: Path, loader: Any, raw: bytes) -> None:
    path = sandbox / "bad.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        loader(path)


@pytest.mark.parametrize(
    "loader,value", [(load_recipe, recipe_json()), (load_policy, policy_json())]
)
def test_input_byte_limit(sandbox: Path, loader: Any, value: object) -> None:
    raw = json.dumps(value).encode()
    path = sandbox / "large.json"
    path.write_bytes(raw + b' ' * (65536 - len(raw)))
    loader(path)
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        loader(path)


@pytest.mark.parametrize("target", ["recipe", "cell", "groups", "policy"])
def test_duplicates_inside_otherwise_valid_json(sandbox: Path, target: str) -> None:
    value = policy_json() if target == "policy" else recipe_json()
    raw = json.dumps(value)
    original, duplicate = {
        "recipe": ('"schema": 1', '"schema": 1, "schema": 1'),
        "cell": ('"rounds": 11', '"rounds": 11, "rounds": 11'),
        "groups": ('"screen": 3', '"screen": 3, "screen": 3'),
        "policy": ('"cpu_seconds": 10', '"cpu_seconds": 10, "cpu_seconds": 10'),
    }[target]
    path = sandbox / "duplicate.json"
    path.write_text(raw.replace(original, duplicate), encoding="utf-8")
    with pytest.raises(ValueError):
        (load_policy if target == "policy" else load_recipe)(path)


def test_types_are_immutable_and_snapshot_is_detached(sandbox: Path) -> None:
    recipe = load_recipe(save(sandbox, recipe_json()))
    assert isinstance(recipe, Recipe)
    assert recipe.groups == StageSizes(2, 3, 4)
    with pytest.raises(FrozenInstanceError):
        setattr(recipe.cells[0], "rounds", 44)
    snapshot = recipe.to_dict()
    snapshot["cells"] = []
    assert len(recipe.cells) == 1
    assert isinstance(load_policy(save(sandbox, policy_json())), Policy)


def test_check_estimates_without_outputs(sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    value = recipe_json()
    value["cells"].append({"rounds": 44, "active": [32, 33, 34]})
    recipe = save(sandbox, value, "recipe.json")
    policy = save(sandbox, policy_json(), "policy.json")
    before = sorted(sandbox.rglob("*"))
    assert main(["check", "--config", str(recipe), "--policy", str(policy)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["device"] == "cpu"
    assert result["recipe"] == value
    assert result["policy"] == policy_json()
    assert result["estimate"]["planned_real_groups"] == 18
    assert result["estimate"]["planned_null_groups"] == 18
    assert result["estimate"]["planned_real_blocks"] == 108
    assert result["estimate"]["planned_null_words"] == 108
    assert result["estimate"]["cells"][0]["points_per_group"] == 4
    assert result["estimate"]["cells"][1]["points_per_group"] == 8
    assert result["planned_calls_fit_policy"] == {"real_blocks": True, "null_words": True,
                                                        "optimizer_updates": True,
                                                        "forward_rows": True}
    assert sorted(sandbox.rglob("*")) == before


def test_check_does_not_approve_or_reject_execution_budget(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipe = save(sandbox, recipe_json(), "recipe.json")
    value = policy_json()
    value["real_blocks"] = 0
    value["null_words"] = 35
    policy = save(sandbox, value, "policy.json")
    assert main(["check", "--config", str(recipe), "--policy", str(policy)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["planned_calls_fit_policy"] == {"real_blocks": False, "null_words": False,
                                                        "optimizer_updates": True,
                                                        "forward_rows": True}


def test_check_with_poisoned_scientific_imports(sandbox: Path) -> None:
    recipe = save(sandbox, recipe_json(), "recipe.json")
    policy = save(sandbox, policy_json(), "policy.json")
    script = '''
import importlib.abc, sys
class BlockScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(part in fullname.split('.') for part in
               ('data', 'ciphers', 'integral', 'models', 'training', 'numpy', 'torch',
                'cipher_experiments')):
            raise AssertionError('scientific import prohibited: ' + fullname)
sys.meta_path.insert(0, BlockScience())
from cipher_lab.cli import main
raise SystemExit(main(sys.argv[1:]))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               PYTHONDONTWRITEBYTECODE="1")
    before = sorted(sandbox.rglob("*"))
    result = subprocess.run(
        [sys.executable, "-c", script, "check", "--config", str(recipe), "--policy", str(policy)],
        cwd=sandbox, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["estimate"]["planned_real_blocks"] == 36
    assert sorted(sandbox.rglob("*")) == before


def test_cli_module_and_error_paths(sandbox: Path) -> None:
    config = save(sandbox, recipe_json(), "recipe.json")
    policy = save(sandbox, policy_json(), "policy.json")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               PYTHONDONTWRITEBYTECODE="1")
    for args, expected in [
        (["check", "--config", str(config), "--policy", str(policy)], 0),
        (["run"], 2), (["evaluate"], 2), (["check"], 2),
        (["check", "--config", str(config), "--policy", str(sandbox / "missing")], 2),
    ]:
        result = subprocess.run([sys.executable, "-m", "cipher_lab.cli", *args],
                                cwd=sandbox, env=env, capture_output=True, text=True, check=False)
        assert result.returncode == expected, result.stderr
        assert "Traceback" not in result.stderr
    config.write_text('{"schema":true}', encoding="utf-8")
    assert main(["check", "--config", str(config), "--policy", str(policy)]) == 2

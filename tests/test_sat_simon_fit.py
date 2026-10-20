"""Synthetic contracts for bounded SAT fitting on a public SIMON bit cone."""

from __future__ import annotations

import ast
import builtins
from dataclasses import FrozenInstanceError
from itertools import product
from pathlib import Path
from typing import Any, cast

import pytest

from cipher_experiments.fitting import sat_simon
from cipher_experiments.fitting.sat_simon import (
    SatSimonFitResult,
    _and_clauses,
    _xor_clauses,
    check_sat_backend,
    fit_sat_simon,
)
from cipher_experiments.representations.simon_cone import SimonCone, simon_cone


def _interpret(cone: SimonCone, plaintext: int, assignment: int) -> int:
    """Independent scalar interpreter; assignment bits follow sorted coordinates."""
    positions = {coordinate: index for index, coordinate in enumerate(cone.round_key_coordinates)}
    values: list[int] = []
    for node in cone.nodes:
        if node.kind == "plaintext":
            (bit_index,) = node.coordinate
            values.append((plaintext >> (63 - bit_index)) & 1)
        elif node.kind == "round_key":
            index = positions[(node.coordinate[0], node.coordinate[1])]
            values.append((assignment >> index) & 1)
        else:
            left, right = node.args
            if node.kind == "and":
                values.append(values[left] & values[right])
            else:
                values.append(values[left] ^ values[right])
    return values[cone.output]


def _clause_value(clause: tuple[int, ...], values: dict[int, bool]) -> bool:
    return any(values[abs(literal)] == (literal > 0) for literal in clause)


def test_check_sat_backend_only_probes_and_closes_glucose4() -> None:
    pytest.importorskip("pysat")
    result = check_sat_backend()
    assert result["backend"] == "Glucose4"
    assert result["python_sat_version"]


def test_check_sat_backend_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = cast(Any, builtins.__import__)

    def without_pysat(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pysat" or name.startswith("pysat."):
            raise ImportError("fixture: PySAT is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pysat)
    with pytest.raises(RuntimeError, match="already-installed python-sat"):
        check_sat_backend()


def test_and_and_xor_cnf_match_every_truth_table_row() -> None:
    for gate, encoding in (("and", _and_clauses), ("xor", _xor_clauses)):
        for left, right, output in product((False, True), repeat=3):
            values = {1: left, 2: right, 3: output}
            encoded = all(
                _clause_value(clause, values)
                for clause in encoding(output=3, left=1, right=2)
            )
            expected = left & right if gate == "and" else left ^ right
            assert encoded is (output == expected)


@pytest.mark.parametrize(("rounds", "target_bit"), [(3, 0), (4, 7), (5, 23), (6, 0)])
def test_fit_witness_matches_shared_key_across_four_train_rows(
    rounds: int, target_bit: int
) -> None:
    pytest.importorskip("pysat")
    plaintexts = (0, 1, 0x0123456789ABCDEF, 0xFEDCBA9876543210)
    cone = simon_cone(rounds, target_bit)
    planted = (0x13579B * 0x9E3779B1) & ((1 << len(cone.round_key_coordinates)) - 1)
    targets = tuple(_interpret(cone, plaintext, planted) for plaintext in plaintexts)

    result = fit_sat_simon(
        plaintexts,
        targets,
        rounds=rounds,
        target_bit=target_bit,
        max_seconds=15,
    )

    assert isinstance(result, SatSimonFitResult)
    assert result.status == "fitted"
    assert result.assignment is not None
    assert result.variable_coordinates == cone.round_key_coordinates
    assert result.train_rows == 4
    assert result.variables >= len(cone.round_key_coordinates)
    assert result.clauses > 0
    assert result.elapsed_seconds >= 0
    assert tuple(_interpret(cone, p, result.assignment) for p in plaintexts) == targets


def test_contradictory_duplicate_plaintext_labels_are_unsat() -> None:
    pytest.importorskip("pysat")
    result = fit_sat_simon(
        (0x123456789ABCDEF0, 0x123456789ABCDEF0),
        (0, 1),
        rounds=3,
        target_bit=0,
        max_seconds=5,
    )
    assert result.status == "unsat"
    assert result.assignment is None
    assert result.train_rows == 2
    assert result.variables > 0
    assert result.clauses > 0


def test_solve_limited_none_is_unknown_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pysat")
    calls = 0

    def interrupted(solver: object, seconds: float) -> None:
        nonlocal calls
        calls += 1
        assert seconds > 0
        return None

    monkeypatch.setattr(sat_simon, "_solve_limited", interrupted)
    result = fit_sat_simon(
        (0, 1, 2, 3),
        (0, 1, 0, 1),
        rounds=3,
        target_bit=0,
        max_seconds=5,
    )
    assert calls == 1
    assert result.status == "unknown"
    assert result.assignment is None


def test_tiny_budget_returns_unknown() -> None:
    result = fit_sat_simon(
        (0,), (0,), rounds=6, target_bit=0, max_seconds=1e-12
    )
    assert result.status == "unknown"
    assert result.assignment is None
    assert result.train_rows == 1


@pytest.mark.parametrize("max_seconds", [0, -1, float("inf"), float("nan"), True])
def test_rejects_invalid_time_budgets(max_seconds: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        fit_sat_simon(
            (0,), (0,), rounds=1, target_bit=0, max_seconds=cast(int | float, max_seconds)
        )


@pytest.mark.parametrize(
    ("plaintexts", "targets"),
    [
        ((), ()),
        ((0,), ()),
        ((-1,), (0,)),
        ((1 << 64,), (0,)),
        ((True,), (0,)),
        ((0,), (True,)),
        ((0,), (2,)),
        ([0], (0,)),
    ],
)
def test_rejects_invalid_train_rows(plaintexts: object, targets: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        fit_sat_simon(
            cast(tuple[int, ...], plaintexts),
            cast(tuple[int, ...], targets),
            rounds=1,
            target_bit=0,
            max_seconds=1,
        )


def test_rejects_row_and_cone_bounds() -> None:
    with pytest.raises(ValueError, match="must not exceed 512"):
        fit_sat_simon(
            (0,) * 513, (0,) * 513, rounds=1, target_bit=0, max_seconds=1
        )
    with pytest.raises(ValueError, match="between 1 and 6"):
        fit_sat_simon((0,), (0,), rounds=7, target_bit=0, max_seconds=1)
    with pytest.raises(ValueError, match="between 0 and 63"):
        fit_sat_simon((0,), (0,), rounds=1, target_bit=64, max_seconds=1)


def test_result_is_frozen_and_source_has_only_public_cone_dependencies() -> None:
    pytest.importorskip("pysat")
    result = fit_sat_simon(
        (0,), (0,), rounds=1, target_bit=32, max_seconds=2
    )
    with pytest.raises(FrozenInstanceError):
        result.status = "unknown"  # type: ignore[misc]

    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/cipher_experiments/fitting/sat_simon.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imports <= {
        "__future__",
        "math",
        "time",
        "dataclasses",
        "threading",
        "typing",
        "pysat",
        "pysat.solvers",
        "representations.simon_cone",
    }

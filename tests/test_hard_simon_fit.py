"""Synthetic contracts for exact hard-assignment fitting on a public SIMON cone."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import pytest

from cipher_experiments.fitting import hard_simon
from cipher_experiments.fitting.hard_simon import (
    HardSimonFitResult,
    _block_key_patterns,
    _local_bit_patterns,
    fit_hard_simon,
)
from cipher_experiments.representations.simon_cone import SimonCone, simon_cone


def _interpret(cone: SimonCone, plaintext: int, assignment: int) -> int:
    """Independent scalar test interpreter with assignment bit i for coordinate i."""
    key_positions = {
        coordinate: index for index, coordinate in enumerate(cone.round_key_coordinates)
    }
    values: list[int] = []
    for node in cone.nodes:
        if node.kind == "plaintext":
            (bit_index,) = node.coordinate
            values.append((plaintext >> (63 - bit_index)) & 1)
        elif node.kind == "round_key":
            index = key_positions[(node.coordinate[0], node.coordinate[1])]
            values.append((assignment >> index) & 1)
        else:
            left, right = node.args
            if node.kind == "and":
                values.append(values[left] & values[right])
            else:
                values.append(values[left] ^ values[right])
    return values[cone.output]


def _lowest_witness(
    plaintexts: tuple[int, ...], targets: tuple[int, ...], rounds: int, target_bit: int
) -> int | None:
    """Brute-force small synthetic fixtures independently of production search."""
    cone = simon_cone(rounds, target_bit)
    for assignment in range(1 << len(cone.round_key_coordinates)):
        if all(
            _interpret(cone, plaintext, assignment) == target
            for plaintext, target in zip(plaintexts, targets, strict=True)
        ):
            return assignment
    return None


def _predict(
    plaintexts: tuple[int, ...], rounds: int, target_bit: int, assignment: int
) -> tuple[int, ...]:
    cone = simon_cone(rounds, target_bit)
    return tuple(_interpret(cone, plaintext, assignment) for plaintext in plaintexts)


def test_bit_patterns_cover_local_and_upper_assignment_boundaries() -> None:
    patterns = _local_bit_patterns(width=256, variable_count=8)
    for variable, pattern in enumerate(patterns):
        assert all(((pattern >> candidate) & 1) == ((candidate >> variable) & 1)
                   for candidate in range(256))

    block_mask = (1 << 16) - 1
    first = _block_key_patterns(patterns[:4], 6, 16, block_mask)
    second = _block_key_patterns(patterns[:4], 6, 32, block_mask)
    for local_assignment in range(16):
        full_first = 16 + local_assignment
        full_second = 32 + local_assignment
        assert all(((first[index] >> local_assignment) & 1) == ((full_first >> index) & 1)
                   for index in range(6))
        assert all(((second[index] >> local_assignment) & 1) == ((full_second >> index) & 1)
                   for index in range(6))


def test_solver_matches_independent_first_assignment_and_reports_full_block() -> None:
    rounds, target_bit = 3, 0
    plaintexts = (0, 1, 0x0123456789ABCDEF, 0xFEDCBA9876543210, 0xA55A1234CDEF9876)
    synthetic_assignment = 137
    targets = _predict(plaintexts, rounds, target_bit, synthetic_assignment)
    expected = _lowest_witness(plaintexts, targets, rounds, target_bit)
    assert expected is not None

    result = fit_hard_simon(
        plaintexts,
        targets,
        rounds=rounds,
        target_bit=target_bit,
        max_seconds=5.0,
        chunk_bits=3,
    )

    cone = simon_cone(rounds, target_bit)
    block_size = 1 << 3
    assert isinstance(result, HardSimonFitResult)
    assert result.status == "fitted"
    assert result.assignment == expected
    assert result.variable_coordinates == cone.round_key_coordinates
    assert result.assignment is not None
    assert _predict(plaintexts, rounds, target_bit, result.assignment) == targets
    assert result.candidate_domain_size == 1 << len(cone.round_key_coordinates)
    assert result.blocks_completed == expected // block_size + 1
    assert result.assignments_covered == result.blocks_completed * block_size
    assert result.train_rows == len(plaintexts)
    assert 0 < result.rows_evaluated <= result.train_rows * result.blocks_completed
    assert result.elapsed_seconds >= 0.0


def test_inconsistent_duplicate_labels_exhaust_with_early_row_elimination() -> None:
    plaintexts = (0x123456789ABCDEF0,) * 6
    targets = (0, 1, 0, 1, 0, 1)
    result = fit_hard_simon(
        plaintexts,
        targets,
        rounds=2,
        target_bit=0,
        max_seconds=5.0,
        chunk_bits=2,
    )

    assert result.status == "exhausted"
    assert result.assignment is None
    assert result.candidate_domain_size == 16
    assert result.blocks_completed == 4
    assert result.assignments_covered == 16
    assert result.train_rows == len(plaintexts)
    assert result.rows_evaluated == 2 * result.blocks_completed
    assert result.rows_evaluated < result.train_rows * result.blocks_completed


def test_zero_variable_public_cone_can_be_fitted_without_key_variables() -> None:
    # One-round output bit 32 is a plaintext leaf and has no symbolic key leaf.
    plaintext = 1 << 63
    result = fit_hard_simon(
        (plaintext,),
        (1,),
        rounds=1,
        target_bit=32,
        max_seconds=2.0,
    )
    assert result.assignment == 0
    assert result.variable_coordinates == ()
    assert result.candidate_domain_size == 1
    assert result.blocks_completed == 1
    assert result.assignments_covered == 1
    assert result.status == "fitted"


def test_timeout_raises_instead_of_returning_partial_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = 0

    def expire_during_search(started: float, max_seconds: int | float) -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise TimeoutError("hard SIMON fitting cooperative time budget exceeded")

    monkeypatch.setattr(hard_simon, "_check_time", expire_during_search)
    with pytest.raises(TimeoutError, match="cooperative time budget"):
        fit_hard_simon(
            (0,),
            (0,),
            rounds=1,
            target_bit=0,
            max_seconds=0.25,
        )


@pytest.mark.parametrize("max_seconds", [0, -1, float("inf"), float("nan"), True])
def test_solver_rejects_nonpositive_or_nonfinite_timeout(max_seconds: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        fit_hard_simon(
            (0,),
            (0,),
            rounds=1,
            target_bit=0,
            max_seconds=cast(int | float, max_seconds),
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
def test_solver_rejects_invalid_training_rows(
    plaintexts: object, targets: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        fit_hard_simon(
            cast(tuple[int, ...], plaintexts),
            cast(tuple[int, ...], targets),
            rounds=1,
            target_bit=0,
            max_seconds=1.0,
        )


@pytest.mark.parametrize(("chunk_bits", "max_variables"), [(0, 24), (17, 24), (1, 25)])
def test_solver_rejects_invalid_search_bounds(chunk_bits: int, max_variables: int) -> None:
    with pytest.raises(ValueError):
        fit_hard_simon(
            (0,),
            (0,),
            rounds=1,
            target_bit=0,
            max_seconds=1.0,
            chunk_bits=chunk_bits,
            max_variables=max_variables,
        )


def test_solver_refuses_cones_above_fixed_variable_guard() -> None:
    with pytest.raises(ValueError, match="exceeding max_variables=24"):
        fit_hard_simon(
            (0,),
            (0,),
            rounds=5,
            target_bit=0,
            max_seconds=1.0,
        )


def test_result_is_frozen_and_module_has_only_stdlib_or_public_cone_imports() -> None:
    from dataclasses import FrozenInstanceError

    result = fit_hard_simon(
        (0,),
        (0,),
        rounds=1,
        target_bit=0,
        max_seconds=2.0,
    )
    with pytest.raises(FrozenInstanceError):
        result.status = "exhausted"  # type: ignore[misc]

    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/cipher_experiments/fitting/hard_simon.py"
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
    assert imports <= {"__future__", "math", "time", "dataclasses", "typing",
                       "representations.simon_cone"}

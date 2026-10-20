"""Exact bounded bit-parallel fitting for symbolic SIMON output-bit cones.

The only learned values are assignments to the public cone's symbolic round-key
coordinates. Inputs are TRAIN plaintext/bit pairs; this module has no cipher,
key-material, data-loading, or training dependency.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

from ..representations.simon_cone import SimonCone, simon_cone

_MAX_VARIABLES = 24
_MAX_CHUNK_BITS = 16


@dataclass(frozen=True)
class HardSimonFitResult:
    """Outcome and honest accounting for one exhaustive/first-fit search."""

    assignment: int | None
    variable_coordinates: tuple[tuple[int, int], ...]
    candidate_domain_size: int
    blocks_completed: int
    assignments_covered: int
    train_rows: int
    rows_evaluated: int
    status: Literal["fitted", "exhausted"]
    elapsed_seconds: float


def _validate_inputs(
    plaintexts: tuple[int, ...], targets: tuple[int, ...], max_seconds: int | float,
    chunk_bits: int, max_variables: int,
) -> None:
    if type(plaintexts) is not tuple or any(type(value) is not int for value in plaintexts):
        raise TypeError("plaintexts must be a tuple of built-in ints")
    if type(targets) is not tuple or any(type(value) is not int for value in targets):
        raise TypeError("targets must be a tuple of built-in ints")
    if not plaintexts:
        raise ValueError("training rows must be nonempty")
    if len(plaintexts) != len(targets):
        raise ValueError("plaintexts and targets must have matching lengths")
    if any(not 0 <= plaintext < (1 << 64) for plaintext in plaintexts):
        raise ValueError("plaintexts must be unsigned 64-bit integers")
    if any(target not in (0, 1) for target in targets):
        raise ValueError("targets must contain only 0 or 1")
    if type(max_seconds) not in (int, float):
        raise TypeError("max_seconds must be a built-in int or float")
    if max_seconds <= 0 or (type(max_seconds) is float and not math.isfinite(max_seconds)):
        raise ValueError("max_seconds must be positive and finite")
    if type(chunk_bits) is not int:
        raise TypeError("chunk_bits must be a built-in int")
    if not 1 <= chunk_bits <= _MAX_CHUNK_BITS:
        raise ValueError(f"chunk_bits must be between 1 and {_MAX_CHUNK_BITS}")
    if type(max_variables) is not int:
        raise TypeError("max_variables must be a built-in int")
    if not 0 <= max_variables <= _MAX_VARIABLES:
        raise ValueError(f"max_variables must be between 0 and {_MAX_VARIABLES}")


def _check_time(started: float, max_seconds: int | float) -> None:
    if time.monotonic() - started >= max_seconds:
        raise TimeoutError("hard SIMON fitting cooperative time budget exceeded")


def _local_bit_patterns(width: int, variable_count: int) -> tuple[int, ...]:
    """Make masks whose set positions are local assignment indices with bit i set."""
    full_mask = (1 << width) - 1
    patterns: list[int] = []
    for variable in range(variable_count):
        run = 1 << variable
        period = 2 * run
        # Start with one (run zeroes, run ones) pair, then repeat by doubling.
        pattern = ((1 << run) - 1) << run
        while period < width:
            pattern |= pattern << period
            period *= 2
        patterns.append(pattern & full_mask)
    return tuple(patterns)


def _block_key_patterns(
    local_patterns: tuple[int, ...],
    variable_count: int,
    block_start: int,
    all_candidates: int,
) -> tuple[int, ...]:
    """Combine local alternating masks and constant upper-variable block bits."""
    return tuple(
        local_patterns[position]
        if position < len(local_patterns)
        else (all_candidates if (block_start >> position) & 1 else 0)
        for position in range(variable_count)
    )


def _evaluate_row(
    cone: SimonCone,
    plaintext: int,
    key_patterns: tuple[int, ...],
    key_positions: dict[tuple[int, int], int],
    all_candidates: int,
) -> int:
    """Evaluate one TRAIN plaintext over every assignment in the active block."""
    values: list[int] = []
    for node in cone.nodes:
        if node.kind == "plaintext":
            (bit_index,) = node.coordinate
            values.append(
                all_candidates if (plaintext >> (63 - bit_index)) & 1 else 0
            )
        elif node.kind == "round_key":
            values.append(key_patterns[key_positions[(node.coordinate[0], node.coordinate[1])]])
        else:
            left, right = node.args
            if node.kind == "and":
                values.append(values[left] & values[right])
            else:  # SimonCone validates all remaining nodes as XOR gates.
                values.append(values[left] ^ values[right])
    return values[cone.output]


def fit_hard_simon(
    plaintexts: tuple[int, ...],
    targets: tuple[int, ...],
    *,
    rounds: int,
    target_bit: int,
    max_seconds: int | float,
    chunk_bits: int = 16,
    max_variables: int = 24,
) -> HardSimonFitResult:
    """Find the lowest-index exact TRAIN assignment, or exhaust the bounded domain.

    Assignment bit ``i`` corresponds to ``variable_coordinates[i]``. Search blocks
    contain at most ``2**chunk_bits`` candidate assignments and are visited in
    ascending integer order. Timeout is cooperative and raises ``TimeoutError``;
    it never returns a partial fit or an exhausted result.
    """
    _validate_inputs(plaintexts, targets, max_seconds, chunk_bits, max_variables)
    started = time.monotonic()
    cone = simon_cone(rounds, target_bit)
    variable_coordinates = cone.round_key_coordinates
    variable_count = len(variable_coordinates)
    if variable_count > max_variables:
        raise ValueError(
            f"cone has {variable_count} variables, exceeding max_variables={max_variables}"
        )
    _check_time(started, max_seconds)

    domain_size = 1 << variable_count
    block_size = 1 << min(chunk_bits, variable_count)
    maximum_width = min(block_size, domain_size)
    local_patterns = _local_bit_patterns(
        maximum_width, min(chunk_bits, variable_count)
    )
    key_positions = {coordinate: index for index, coordinate in enumerate(variable_coordinates)}
    blocks_completed = 0
    assignments_covered = 0
    rows_evaluated = 0

    for block_start in range(0, domain_size, block_size):
        _check_time(started, max_seconds)
        block_width = min(block_size, domain_size - block_start)
        all_candidates = (1 << block_width) - 1
        # Check around mask creation as well as at block boundaries: a timeout is
        # deliberately surfaced rather than converted into a partial result.
        _check_time(started, max_seconds)
        patterns = _block_key_patterns(
            local_patterns, variable_count, block_start, all_candidates
        )
        _check_time(started, max_seconds)

        alive = all_candidates
        for plaintext, target in zip(plaintexts, targets, strict=True):
            _check_time(started, max_seconds)
            predicted = _evaluate_row(
                cone, plaintext, patterns, key_positions, all_candidates
            )
            correct = predicted if target == 1 else all_candidates ^ predicted
            alive &= correct
            rows_evaluated += 1
            _check_time(started, max_seconds)
            if alive == 0:
                break

        blocks_completed += 1
        assignments_covered += block_width
        if alive:
            lowest_local_assignment = (alive & -alive).bit_length() - 1
            assignment = block_start + lowest_local_assignment
            _check_time(started, max_seconds)
            return HardSimonFitResult(
                assignment=assignment,
                variable_coordinates=variable_coordinates,
                candidate_domain_size=domain_size,
                blocks_completed=blocks_completed,
                assignments_covered=assignments_covered,
                train_rows=len(plaintexts),
                rows_evaluated=rows_evaluated,
                status="fitted",
                elapsed_seconds=time.monotonic() - started,
            )

    _check_time(started, max_seconds)
    return HardSimonFitResult(
        assignment=None,
        variable_coordinates=variable_coordinates,
        candidate_domain_size=domain_size,
        blocks_completed=blocks_completed,
        assignments_covered=assignments_covered,
        train_rows=len(plaintexts),
        rows_evaluated=rows_evaluated,
        status="exhausted",
        elapsed_seconds=time.monotonic() - started,
    )

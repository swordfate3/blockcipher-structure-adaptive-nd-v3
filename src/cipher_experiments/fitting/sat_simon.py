"""Bounded SAT fitting for public SIMON output-bit cones and TRAIN observations.

Only symbolic round-key coordinates from the public Boolean cone are unknown.
This module does not generate labels, load data, or call a cipher implementation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from threading import TIMEOUT_MAX, Event, Timer
from typing import Literal, Protocol

from ..representations.simon_cone import SimonCone, simon_cone

_MAX_ROWS = 512
_MAX_UNKNOWN = 70
_INTERRUPT_INTERVAL = 0.5


class _LimitedSolver(Protocol):
    def add_clause(self, clause: list[int]) -> bool: ...

    def interrupt(self) -> None: ...

    def solve_limited(
        self, assumptions: list[int] = ..., expect_interrupt: bool = False
    ) -> bool | None: ...

    def get_model(self) -> list[int] | None: ...


@dataclass(frozen=True)
class SatSimonFitResult:
    """SAT fitting outcome and the size of the encoded TRAIN problem."""

    assignment: int | None
    variable_coordinates: tuple[tuple[int, int], ...]
    status: Literal["fitted", "unsat", "unknown"]
    train_rows: int
    variables: int
    clauses: int
    elapsed_seconds: float


def _and_clauses(output: int, left: int, right: int) -> tuple[tuple[int, ...], ...]:
    """Clauses encoding ``output = left AND right``."""
    return ((-left, -right, output), (left, -output), (right, -output))


def _xor_clauses(output: int, left: int, right: int) -> tuple[tuple[int, ...], ...]:
    """Clauses encoding ``output = left XOR right``."""
    return (
        (left, right, -output),
        (-left, -right, -output),
        (left, -right, output),
        (-left, right, output),
    )


def _validate_inputs(
    plaintexts: tuple[int, ...],
    targets: tuple[int, ...],
    rounds: int,
    target_bit: int,
    max_seconds: int | float,
) -> None:
    if type(plaintexts) is not tuple or any(type(value) is not int for value in plaintexts):
        raise TypeError("plaintexts must be a tuple of built-in ints")
    if type(targets) is not tuple or any(type(value) is not int for value in targets):
        raise TypeError("targets must be a tuple of built-in ints")
    if not plaintexts:
        raise ValueError("training rows must be nonempty")
    if len(plaintexts) != len(targets):
        raise ValueError("plaintexts and targets must have matching lengths")
    if len(plaintexts) > _MAX_ROWS:
        raise ValueError(f"training rows must not exceed {_MAX_ROWS}")
    if any(not 0 <= value < (1 << 64) for value in plaintexts):
        raise ValueError("plaintexts must be unsigned 64-bit integers")
    if any(value not in (0, 1) for value in targets):
        raise ValueError("targets must contain only 0 or 1")
    if type(rounds) is not int:
        raise TypeError("rounds must be a built-in int")
    if not 1 <= rounds <= 6:
        raise ValueError("rounds must be between 1 and 6")
    if type(target_bit) is not int:
        raise TypeError("target_bit must be a built-in int")
    if not 0 <= target_bit < 64:
        raise ValueError("target_bit must be between 0 and 63")
    if type(max_seconds) not in (int, float):
        raise TypeError("max_seconds must be a built-in int or float")
    if max_seconds <= 0 or (type(max_seconds) is float and not math.isfinite(max_seconds)):
        raise ValueError("max_seconds must be positive and finite")


def _check_deadline(deadline: float) -> bool:
    return time.monotonic() < deadline


def _encode_train_rows(
    solver: _LimitedSolver,
    cone: SimonCone,
    plaintexts: tuple[int, ...],
    targets: tuple[int, ...],
    key_variables: dict[tuple[int, int], int],
    deadline: float,
) -> tuple[int, int, bool]:
    """Add one fresh circuit per row and return variable/clause counts and completion."""
    next_variable = len(key_variables)
    clause_count = 0
    for plaintext, target in zip(plaintexts, targets, strict=True):
        if not _check_deadline(deadline):
            return next_variable, clause_count, False
        node_variables: list[int] = []
        for node in cone.nodes:
            if node.kind == "round_key":
                node_variables.append(key_variables[(node.coordinate[0], node.coordinate[1])])
                continue

            next_variable += 1
            variable = next_variable
            node_variables.append(variable)
            clauses: tuple[tuple[int, ...], ...]
            if node.kind == "plaintext":
                (bit_index,) = node.coordinate
                bit = (plaintext >> (63 - bit_index)) & 1
                clauses = ((variable if bit else -variable,),)
            else:
                left, right = (node_variables[index] for index in node.args)
                if node.kind == "and":
                    clauses = _and_clauses(variable, left, right)
                else:
                    clauses = _xor_clauses(variable, left, right)

            for clause in clauses:
                solver.add_clause(list(clause))
                clause_count += 1
            if not _check_deadline(deadline):
                return next_variable, clause_count, False

        output_literal = node_variables[cone.output]
        solver.add_clause([output_literal if target else -output_literal])
        clause_count += 1
    return next_variable, clause_count, _check_deadline(deadline)


def _solve_limited(solver: _LimitedSolver, deadline: float) -> bool | None:
    """Run exactly one interruptible solve, joining the interrupt timer on return."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    cancelled = Event()

    def interrupt_at_deadline() -> None:
        while not cancelled.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                solver.interrupt()
                # Measured overruns show this call can be swallowed by the native
                # search, so keep re-asserting it until the solve returns.
                while not cancelled.is_set():
                    solver.interrupt()
                    cancelled.wait(_INTERRUPT_INTERVAL)
                return
            cancelled.wait(min(remaining, TIMEOUT_MAX))

    timer = Timer(min(remaining, TIMEOUT_MAX), interrupt_at_deadline)
    timer.daemon = True
    timer.start()
    try:
        return solver.solve_limited(expect_interrupt=True)
    finally:
        cancelled.set()
        timer.cancel()
        timer.join()


def _result(
    assignment: int | None,
    coordinates: tuple[tuple[int, int], ...],
    status: Literal["fitted", "unsat", "unknown"],
    train_rows: int,
    variables: int,
    clauses: int,
    started: float,
) -> SatSimonFitResult:
    return SatSimonFitResult(
        assignment=assignment,
        variable_coordinates=coordinates,
        status=status,
        train_rows=train_rows,
        variables=variables,
        clauses=clauses,
        elapsed_seconds=time.monotonic() - started,
    )


def check_sat_backend() -> dict[str, str]:
    """Preflight the installed Glucose4 backend without constructing a fit problem."""
    try:
        import pysat  # type: ignore[import-untyped]
        from pysat.solvers import Solver  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "SAT Simon fitting requires the already-installed python-sat package"
        ) from exc

    try:
        with Solver(name="g4"):
            pass
    except Exception as exc:
        raise RuntimeError("SAT Simon fitting requires a usable PySAT Glucose4 backend") from exc
    return {
        "backend": "Glucose4",
        "python_sat_version": str(getattr(pysat, "__version__", "unknown")),
    }


def fit_sat_simon(
    plaintexts: tuple[int, ...],
    targets: tuple[int, ...],
    *,
    rounds: int,
    target_bit: int,
    max_seconds: float | int,
) -> SatSimonFitResult:
    """Fit shared symbolic key bits to bounded TRAIN rows using one SAT solve.

    Assignment bit ``i`` corresponds to ``variable_coordinates[i]``. Time spent
    building the CNF counts against ``max_seconds``. Deadline expiry, including a
    solver interruption, is reported as ``unknown`` rather than a partial fit.
    """
    _validate_inputs(plaintexts, targets, rounds, target_bit, max_seconds)
    started = time.monotonic()
    try:
        budget_seconds = float(max_seconds)
    except OverflowError:
        budget_seconds = math.inf
    deadline = started + budget_seconds
    cone = simon_cone(rounds, target_bit)
    coordinates = cone.round_key_coordinates
    if len(coordinates) > _MAX_UNKNOWN:
        raise ValueError(f"cone has more than {_MAX_UNKNOWN} unknown variables")
    if not _check_deadline(deadline):
        return _result(None, coordinates, "unknown", len(plaintexts), len(coordinates), 0, started)

    try:
        from pysat.solvers import Solver
    except ImportError as exc:
        raise RuntimeError(
            "SAT Simon fitting requires the already-installed python-sat package"
        ) from exc

    key_variables = {coordinate: index for index, coordinate in enumerate(coordinates, start=1)}
    variables = len(key_variables)
    clauses = 0
    try:
        with Solver(name="g4") as solver:
            variables, clauses, complete = _encode_train_rows(
                solver, cone, plaintexts, targets, key_variables, deadline
            )
            if not complete:
                return _result(
                    None, coordinates, "unknown", len(plaintexts), variables, clauses, started
                )

            if time.monotonic() >= deadline:
                return _result(
                    None, coordinates, "unknown", len(plaintexts), variables, clauses, started
                )
            solved = _solve_limited(solver, deadline)
            # A callback can race with a boundary result; never claim SAT/UNSAT
            # after the advertised wall-clock budget has expired.
            if not _check_deadline(deadline) or solved is None:
                status: Literal["fitted", "unsat", "unknown"] = "unknown"
                assignment = None
            elif solved is False:
                status = "unsat"
                assignment = None
            else:
                status = "fitted"
                model = solver.get_model()
                if model is None:
                    status, assignment = "unknown", None
                else:
                    true_variables = {literal for literal in model if literal > 0}
                    assignment = sum(
                        1 << index
                        for index, coordinate in enumerate(coordinates)
                        if key_variables[coordinate] in true_variables
                    )
            return _result(
                assignment, coordinates, status, len(plaintexts), variables, clauses, started
            )
    except ImportError as exc:
        raise RuntimeError(
            "SAT Simon fitting requires the already-installed python-sat package"
        ) from exc

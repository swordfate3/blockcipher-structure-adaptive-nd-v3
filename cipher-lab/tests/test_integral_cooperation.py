"""Cooperative GF(2) checks on artificial public rows, never scientific calls."""

import ast
import inspect
import sys
from collections.abc import Callable, Iterator, Mapping
from types import FrameType
from typing import Any

import pytest

import cipher_lab.integral.linear as linear
import cipher_lab.integral.method as method
import cipher_lab.integral.stages as stages
from cipher_lab.config import Cell, Recipe, StageSizes
from cipher_lab.execution import Stopped
from cipher_lab.integral.linear import canonical_basis, confirms, kernel, survivor_basis

UNITS = tuple(1 << bit for bit in reversed(range(2080)))


@pytest.fixture(autouse=True)
def forbid_science(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cipher_lab.ciphers", None)
    monkeypatch.setitem(sys.modules, "torch", None)


@pytest.mark.parametrize("operation,args,expected", [
    (canonical_basis, (UNITS,), UNITS),
    (kernel, (UNITS,), ()),
    (survivor_basis, ((3, 5), (1,)), (6,)),
    (confirms, (UNITS, (0, 0, 1)), False),
])
def test_callback_keeps_full_space_values(
    operation: Callable[..., object], args: tuple[object, ...], expected: object,
) -> None:
    checks: list[None] = []
    assert operation(*args, check=lambda: checks.append(None)) == expected
    assert len(checks) >= 2
    assert operation(*args) == expected


@pytest.mark.parametrize("operation,args,expected", [
    (canonical_basis, ((),), ()), (kernel, ((), 1), (1,)),
    (survivor_basis, ((), ()), ()), (confirms, ((), ()), True),
    (confirms, ((1,), (1,)), False),
])
def test_entry_and_exit_even_empty_or_false(
    operation: Callable[..., object], args: tuple[object, ...], expected: object,
) -> None:
    calls: list[None] = []
    assert operation(*args, check=lambda: calls.append(None)) == expected
    assert len(calls) >= 2
    error = Stopped("CPU_LIMIT")

    def stop_at_exit() -> None:
        if calls:
            calls.clear()
        else:
            raise error

    # The first call consumes the marker; the final check must propagate unchanged.
    calls[:] = [None]
    with pytest.raises(Stopped) as caught:
        operation(*args, check=stop_at_exit)
    assert caught.value is error


@pytest.mark.parametrize("operation,args,error", [
    (canonical_basis, ((True,),), TypeError),
    (canonical_basis, ((1,), False), TypeError),
    (kernel, ((1,), 0), ValueError),
    (kernel, ((1 << 2080,),), ValueError),
    (survivor_basis, ((), (True,)), TypeError),
    (survivor_basis, ((True,), ()), TypeError),
    (confirms, ((), (-1,)), ValueError),
    (confirms, ((1,), (1, True)), TypeError),
])
def test_validation_not_hidden_by_empty_or_false(
    operation: Callable[..., object], args: tuple[object, ...], error: type[Exception],
) -> None:
    with pytest.raises(error):
        operation(*args)
    with pytest.raises(error):
        operation(*args, check=lambda: None)


def test_no_callback_retains_iterator_error_before_vector_validation() -> None:
    error = RuntimeError("artificial iterator failure")

    def rows() -> Iterator[int]:
        yield True
        raise error

    with pytest.raises(RuntimeError) as caught:
        canonical_basis(rows())
    assert caught.value is error


@pytest.mark.parametrize("operation", [canonical_basis, kernel, survivor_basis, confirms])
def test_generator_consumption_checked_before_65_more_rows(
    operation: Callable[..., object],
) -> None:
    yielded = 0
    last = 0
    error = Stopped("CPU_LIMIT")

    def rows() -> Iterator[int]:
        nonlocal yielded
        for _ in range(4096):
            yielded += 1
            yield 0

    def check() -> None:
        nonlocal last
        assert yielded - last <= 64
        last = yielded
        if yielded >= 64:
            raise error

    args = (rows(),) if operation in (canonical_basis, kernel) else ((), rows())
    with pytest.raises(Stopped) as caught:
        operation(*args, check=check)
    assert caught.value is error
    assert yielded == 64


# Locate a callback's active mathematical loop, without replacing that loop or
# relying on a global callback ordinal that would hide missed inner-loop checks.
@pytest.mark.parametrize("phase,operation,args", [
    ("validation", canonical_basis, (UNITS,)),
    ("eliminate", canonical_basis, (UNITS + ((1 << 2080) - 1,),)),
    ("prior", canonical_basis, (tuple((1 << bit) | 1 for bit in range(2079, 0, -1)),)),
    ("kernel", kernel, ((1,),)),
    ("coefficient", survivor_basis, (UNITS, (0,))),
    ("backsub", survivor_basis, (UNITS[:128], (0,))),
    ("confirm", confirms, (UNITS, (0,) * 128)),
])
def test_stop_inside_each_long_math_phase(
    phase: str, operation: Callable[..., object], args: tuple[object, ...],
) -> None:
    error = Stopped("CPU_LIMIT")
    locations = {
        "validation": ("_validated", "row"),
        "eliminate": ("_canonical", "pivot"),
        "prior": ("_canonical", "old_pivot"),
        "kernel": ("_kernel", "vector"),
        "coefficient": ("survivor_basis", "j"),
        "backsub": ("survivor_basis", "original"),
        "confirm": ("confirms", "row"),
    }
    name, variable = locations[phase]

    def check() -> None:
        frame = inspect.currentframe()
        try:
            while frame is not None:
                if frame.f_code.co_name == name and variable in frame.f_locals:
                    if phase != "eliminate" or len(frame.f_locals["pivots"]) == 2080:
                        raise error
                frame = frame.f_back
        finally:
            del frame

    with pytest.raises(Stopped) as caught:
        operation(*args, check=check)
    assert caught.value is error


@pytest.mark.parametrize("operation,args", [
    (canonical_basis, (UNITS + ((1 << 2080) - 1,),)),
    (canonical_basis, (tuple((1 << bit) | 1 for bit in range(2079, 0, -1)) + (1,),)),
    (kernel, ((1,),)),
    (survivor_basis, (UNITS[:128], (0, 0))),
    (confirms, (UNITS, (0, 0, 1))),
    (confirms, (UNITS, ())),
])
def test_every_loop_iteration_and_inner_xor_within_64_steps(
    operation: Callable[..., object], args: tuple[object, ...],
) -> None:
    # Count source-level mathematical iterations independently of the counter.
    # Include XOR assignments separately: counting only rows misses full-rank
    # pivot scans and the dense elimination/back-substitution work.
    source = inspect.getsource(linear)
    tree = ast.parse(source)
    lines: set[int] = set()
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.For):
                lines.add(node.body[0].lineno)
            if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.BitXor):
                lines.add(node.lineno)
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp)
                    and isinstance(node.value.op, ast.BitXor)):
                lines.add(node.lineno)
    steps = 0
    total = 0
    checks = 0

    def trace(frame: FrameType, event: str, arg: object) -> Any:
        nonlocal steps, total
        if (event == "line" and frame.f_code.co_filename == linear.__file__
                and frame.f_lineno in lines):
            steps += 1
            total += 1
            assert steps <= 64, "more than 64 atomic loop/XOR steps without callback"
        return trace

    def check() -> None:
        nonlocal steps, checks
        steps = 0
        checks += 1

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        operation(*args, check=check)
    finally:
        sys.settrace(previous)
    assert total > 64
    assert checks > 2
    assert steps == 0  # Final callback, including confirmation failure.


@pytest.mark.parametrize("operation,args", [
    (canonical_basis, ((),)), (kernel, ((),)),
    (survivor_basis, ((), ())), (confirms, ((), ())),
])
def test_entry_callback_exception_is_original_object(
    operation: Callable[..., object], args: tuple[object, ...],
) -> None:
    error = RuntimeError("artificial callback failure")

    def fail() -> None:
        raise error

    with pytest.raises(RuntimeError) as caught:
        operation(*args, check=fail)
    assert caught.value is error


class FakeCollector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def row(self, cell: Cell, domain: tuple[str, str, int]) -> int:
        self.calls.append(domain)
        return 0

    def counts(self) -> dict[str, int]:
        return {"completed_groups": len(self.calls), "attempted_groups": len(self.calls)}


class FakeStore:
    def __init__(self) -> None:
        self.payloads: dict[str, object] = {}
        self.final: dict[str, object] = {}

    def current_source_identity(self) -> dict[str, str]:
        return self.source_snapshot()

    def source_snapshot(self) -> dict[str, str]:
        return {"artificial.py": "0" * 64}

    def write_snapshot(self, value: Mapping[str, object]) -> None:
        pass

    def append_event(self, value: Mapping[str, object]) -> None:
        pass

    def write_payload(self, name: str, value: object) -> str:
        self.payloads[name] = value
        return name

    def finalize(self, value: Mapping[str, object]) -> None:
        self.final = dict(value)


@pytest.mark.parametrize("phase", ["kernel", "survivor_basis", "confirms"])
@pytest.mark.parametrize("change_source", [False, True])
def test_method_math_stop_or_source_drift_without_extra_collection(
    monkeypatch: pytest.MonkeyPatch, phase: str, change_source: bool,
) -> None:
    source = {"artificial.py": "0" * 64}
    scans = 0
    active = False
    checks = 0
    error = Stopped("CPU_LIMIT")
    collector = FakeCollector()
    store = FakeStore()
    original: Callable[..., Any] = getattr(stages, phase)

    def identity() -> dict[str, str]:
        nonlocal scans
        assert not active, "GF2 callbacks must not run full source guards"
        scans += 1
        return source.copy()

    monkeypatch.setattr(store, "current_source_identity", identity)

    def wrapped(*args: object, check: Callable[[], None] | None = None) -> Any:
        nonlocal active, checks
        assert check is not None
        active = True
        before = scans
        try:
            # Actual pure mathematics, not a replacement result or cipher data.
            return original(*args, check=check)
        finally:
            assert scans == before
            checks = 0
            active = False

    monkeypatch.setattr(stages, phase, wrapped)

    class FakeMeter:
        def check(self) -> None:
            nonlocal checks
            if active:
                checks += 1
                if checks == 3:
                    if change_source:
                        source["artificial.py"] = "1" * 64
                    else:
                        raise error

        def snapshot(self) -> dict[str, object]:
            return {}

    r = Recipe(1, "integral-empirical-v1", (Cell(1, (0, 1)),), StageSizes(1, 1, 1), 4)
    result = method.run_integral(r, collector, FakeMeter(), store)
    assert result == store.final
    assert result["status"] == "STOPPED"
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert summary["reason"] == ("IDENTITY_CHANGED" if change_source else "CPU_LIMIT")
    expected = {"kernel": 1, "survivor_basis": 2, "confirms": 3}[phase]
    assert len(collector.calls) == expected
    assert result["counts"] == {"completed_groups": expected, "attempted_groups": expected}
    assert store.payloads["c0-real-discovery"] == {"rows": [0], "complete": True}
    if phase == "confirms":
        assert store.payloads["c0-real-basis"] == {
            "discovery_basis": UNITS, "frozen_basis": UNITS}
        assert store.payloads["c0-real-confirm"] == {"rows": [0], "complete": True}

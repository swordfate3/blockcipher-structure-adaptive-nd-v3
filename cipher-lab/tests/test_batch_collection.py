"""Artificial batch data only; complete production cipher entries are poisoned."""
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from test_collection import FiniteBytes, make_meter

from cipher_lab.config import Cell
from cipher_lab.data import cubes
from cipher_lab.data.cubes import CubeCollector
from cipher_lab.demo import DemoBytes
from cipher_lab.execution import Stopped
from cipher_lab.integral.observables import aggregate_batch, aggregate_words

Words = NDArray[np.uint64]


class ArraySubclass(np.ndarray):
    pass


@pytest.fixture(autouse=True)
def poison(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("complete production cipher entry called")
    monkeypatch.setattr("cipher_lab.ciphers.simon.encrypt_many", forbidden)
    monkeypatch.setattr("cipher_lab.ciphers.simon.encrypt_batch", forbidden, raising=False)
    monkeypatch.setattr("secrets.token_bytes", forbidden)


def scalar_forbidden(*args: object) -> tuple[int, ...]:
    pytest.fail("scalar oracle in batch mode")


class Capture:
    def __init__(self) -> None:
        self.calls: list[tuple[Words, tuple[int, ...], int]] = []

    def __call__(self, points: Words, key: tuple[int, ...], rounds: int) -> Words:
        assert points.dtype == np.dtype(np.uint64) and points.ndim == 1
        assert points.flags.c_contiguous and len(points) <= 65536
        self.calls.append((points.copy(), key, rounds))
        return np.zeros(len(points), dtype=np.uint64)


@pytest.mark.parametrize("k,chunk", [(10, 257), (16, 65536), (3, 3)])
def test_batch_enumeration_matches_independent_scalar_order(k: int, chunk: int) -> None:
    active = (63, 32, 40) if k == 3 else tuple(reversed(range(0, 2*k, 2)))
    background = 0xFEDCBA9876543210
    oracle = Capture()
    source = FiniteBytes([bytes(range(16)), background.to_bytes(8, "big"),
                          b"n"*16, b"\xff"*8])
    collector = CubeCollector(make_meter(), scalar_forbidden, source, chunk, batch_oracle=oracle)
    cell = Cell(11, active)
    assert collector.row(cell, ("real", "discovery", 0)) == 0
    points = np.concatenate([call[0] for call in oracle.calls])
    # Deliberately use the independent scalar engineering reference.
    assert points.tolist() == list(cubes._points(tuple(sorted(active)), background))
    masks = sum(1 << (63-bit) for bit in active)
    assert int(points[0]) == background & ~masks
    assert all(int(point) & ~masks == background & ~masks for point in points)
    assert len(set(points.tolist())) == 1 << k
    sizes = [len(call[0]) for call in oracle.calls]
    assert sizes == [min(chunk, (1 << k)-offset) for offset in range(0, 1 << k, chunk)]
    assert {call[1] for call in oracle.calls} == {
        (0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203)}
    assert all(call[2] == 11 for call in oracle.calls)
    before = len(oracle.calls)
    collector.row(cell, ("real", "screen", 0))
    assert oracle.calls[before][1] != oracle.calls[0][1]
    counts = collector.counts()
    assert counts["completed_real_blocks"] == counts["attempted_real_blocks"] == 2*(1 << k)
    assert counts["completed_real_groups"] == 2


@pytest.mark.parametrize("chunk", [1, 3, 4])
def test_bulk_null_big_endian_pointwise_and_no_key_or_oracle(chunk: int) -> None:
    values = [0, 1 << 63, 1 << 62, 3 << 62]
    raw = b"".join(value.to_bytes(8, "big") for value in values)
    source = FiniteBytes([raw[offset*8:min(offset+chunk, 4)*8]
                          for offset in range(0, 4, chunk)])
    collector = CubeCollector(make_meter(), scalar_forbidden, source, chunk,
                              batch_oracle=cast(cubes.BatchOracle, scalar_forbidden))
    assert collector.row(Cell(1, (0, 1)), ("null", "confirm", 0)) == 1 << 2015
    assert source.requests == [8*min(chunk, 4-offset) for offset in range(0, 4, chunk)]
    counts = collector.counts()
    assert counts["completed_null_words"] == counts["attempted_null_words"] == 4
    assert counts["completed_null_groups"] == 1 and counts["attempted_real_blocks"] == 0


@pytest.mark.parametrize("bad", [None, (), True, 1, np.uint64(1),
    np.zeros(4, dtype=np.uint32), np.zeros(4, dtype=np.int64),
    np.full(4, np.nan), np.zeros((2, 2), dtype=np.uint64),
    np.zeros(3, dtype=np.uint64), np.zeros(5, dtype=np.uint64),
    np.zeros(8, dtype=np.uint64)[::2], np.zeros(4, dtype=">u8" if np.little_endian else "<u8"),
    np.zeros(4, dtype=np.uint64).view(ArraySubclass)])
def test_batch_response_strict_validation_sanitized_and_domain_consumed(bad: object) -> None:
    def invalid(points: Words, key: tuple[int, ...], rounds: int) -> Words:
        return cast(Words, bad)
    collector = CubeCollector(make_meter(), scalar_forbidden,
                              FiniteBytes([b"a"*16, b"b"*8]), 4, batch_oracle=invalid)
    domain = ("real", "confirm", 0)
    with pytest.raises(Stopped) as caught:
        collector.row(Cell(1, (0, 1)), domain)
    assert str(caught.value) == "ORACLE_RESPONSE" and caught.value.__cause__ is None
    assert collector.counts()["attempted_real_blocks"] == 4
    assert collector.counts()["completed_real_blocks"] == 0
    with pytest.raises(Stopped, match="DOMAIN_REUSED"):
        collector.row(Cell(1, (1, 0)), domain)


@pytest.mark.parametrize("side,failure", [
    ("real", "oracle"), ("real", "response"), ("real", "budget"), ("real", "callback"),
    ("null", "entropy"), ("null", "budget"), ("null", "callback"),
])
def test_second_chunk_failure_retains_attempts_and_only_previous_completions(
    failure: str, side: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    meter = make_meter(real=3 if failure == "budget" else 100,
                       null=3 if failure == "budget" else 100)
    requests: list[int] = []
    def random(size: int) -> bytes:
        requests.append(size)
        if failure == "entropy" and len(requests) == (3 if side == "real" else 2):
            raise RuntimeError("SECRET entropy canary")
        return b"a"*size
    calls = 0
    def oracle(points: Words, key: tuple[int, ...], rounds: int) -> Words:
        nonlocal calls
        calls += 1
        if calls == 2:
            if failure == "oracle":
                raise RuntimeError("SECRET key canary")
            if failure == "response":
                return np.zeros(2, dtype=np.uint64)
        if failure == "entropy" and calls == 2:
            pytest.fail("real entropy failure must occur before second oracle")
        return np.zeros(len(points), dtype=np.uint64)
    if failure == "callback":
        original = aggregate_batch
        aggregates = 0
        def interrupt(words: Words, *, check: Callable[[], None] | None = None) -> int:
            nonlocal aggregates
            aggregates += 1
            assert check is not None
            if aggregates == 2:
                raise Stopped("CPU_LIMIT")
            return original(words, check=check)
        monkeypatch.setattr(cubes, "aggregate_batch", interrupt)
    collector = CubeCollector(meter, scalar_forbidden, random, 3, batch_oracle=oracle)
    reason = {"entropy": "RANDOM_UNAVAILABLE", "oracle": "ORACLE_FAILED",
              "response": "ORACLE_RESPONSE", "budget": f"{side.upper()}_"
              + ("BLOCKS_LIMIT" if side == "real" else "WORDS_LIMIT"), "callback": "CPU_LIMIT"}
    with pytest.raises(Stopped) as caught:
        collector.row(Cell(1, (0, 1)), (side, "discovery", 0))
    assert str(caught.value) == reason[failure]
    counts = collector.counts()
    unit = "real_blocks" if side == "real" else "null_words"
    assert counts[f"attempted_{unit}"] == (3 if failure == "budget" else 4)
    assert counts[f"completed_{unit}"] == 3 and counts[f"completed_{side}_groups"] == 0
    assert "SECRET" not in repr(counts) + str(caught.value)


@pytest.mark.parametrize("side", ["real", "null"])
@pytest.mark.parametrize("raw", [b"", b"short", bytearray(32)])
def test_initial_entropy_failure_keeps_whole_chunk(side: str, raw: bytes) -> None:
    collector = CubeCollector(make_meter(), scalar_forbidden, lambda _: raw, 4,
                              batch_oracle=Capture())
    with pytest.raises(Stopped, match="RANDOM_UNAVAILABLE"):
        collector.row(Cell(1, (0, 1)), (side, "discovery", 0))
    unit = "real_blocks" if side == "real" else "null_words"
    assert collector.counts()[f"attempted_{unit}"] == 4
    assert collector.counts()[f"completed_{unit}"] == 0


def test_batch_key_rejection_bounded_and_resampling() -> None:
    oracle = Capture()
    requests: list[int] = []
    def duplicate(size: int) -> bytes:
        requests.append(size)
        return b"a"*size
    collector = CubeCollector(make_meter(), scalar_forbidden, duplicate, 4, batch_oracle=oracle)
    collector.row(Cell(1, (0, 1)), ("real", "discovery", 0))
    with pytest.raises(Stopped, match="KEY_REUSE_LIMIT"):
        collector.row(Cell(1, (0, 1)), ("real", "discovery", 1))
    assert requests == [16, 8] + [16]*128
    assert len(oracle.calls) == 1
    assert collector.counts()["attempted_real_blocks"] == 8
    source = FiniteBytes([b"a"*16, b"b"*8, b"a"*16, b"c"*16, b"d"*8])
    collector = CubeCollector(make_meter(), scalar_forbidden, source, 4, batch_oracle=oracle)
    collector.row(Cell(1, (0, 1)), ("real", "screen", 0))
    collector.row(Cell(1, (0, 1)), ("real", "screen", 1))
    assert oracle.calls[-1][1] != oracle.calls[-2][1]


@pytest.mark.parametrize("error", [Stopped("CPU_LIMIT"), RuntimeError("meter callback canary")])
def test_meter_callback_interrupts_real_pair_loop_before_credit(
    monkeypatch: pytest.MonkeyPatch, error: Exception,
) -> None:
    meter = make_meter()
    original = meter.check
    inside = False
    checks = 0
    def check() -> None:
        nonlocal checks
        original()
        if inside:
            checks += 1
            if checks == 3:
                raise error
    monkeypatch.setattr(meter, "check", check)
    aggregate = aggregate_batch
    def wrapped(words: Words, *, check: Callable[[], None] | None = None) -> int:
        nonlocal inside
        inside = True
        return aggregate(words, check=check)
    monkeypatch.setattr(cubes, "aggregate_batch", wrapped)
    collector = CubeCollector(meter, scalar_forbidden, FiniteBytes([b"a"*16, b"b"*8]),
                              4, batch_oracle=Capture())
    with pytest.raises(type(error)) as caught:
        collector.row(Cell(1, (0, 1)), ("real", "discovery", 0))
    assert caught.value is error
    assert checks == 3 and collector.counts()["completed_real_blocks"] == 0


def test_monitor_check_after_preparation_and_after_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    for trigger in ("random", "oracle"):
        meter = make_meter()
        original = meter.check
        expired = False
        calls = 0
        def check() -> None:
            original()
            if expired:
                raise Stopped("WALL_LIMIT")
        monkeypatch.setattr(meter, "check", check)
        def random(size: int) -> bytes:
            nonlocal expired
            if size == 8 and trigger == "random":
                expired = True
            return b"a"*size
        def oracle(points: Words, key: tuple[int, ...], rounds: int) -> Words:
            nonlocal expired, calls
            calls += 1
            expired = True
            return np.zeros(len(points), dtype=np.uint64)
        collector = CubeCollector(meter, scalar_forbidden, random, 4, batch_oracle=oracle)
        with pytest.raises(Stopped, match="WALL_LIMIT"):
            collector.row(Cell(1, (0, 1)), ("real", "discovery", 0))
        assert calls == (0 if trigger == "random" else 1)
        assert collector.counts()["attempted_real_blocks"] == 4
        assert collector.counts()["completed_real_blocks"] == 0


def test_fake_batch_nonzero_aggregation_and_separate_demo_null_adapter() -> None:
    source = DemoBytes()
    def oracle(points: Words, key: tuple[int, ...], rounds: int) -> Words:
        return np.array([0, 1 << 63, 1 << 62, 3 << 62], dtype=np.uint64)
    collector = CubeCollector(make_meter(), scalar_forbidden, source, 4, batch_oracle=oracle,
                              null_random=lambda size: source.bulk_words(size//8))
    assert collector.row(Cell(1, (0, 1)), ("real", "discovery", 0)) == 1 << 2015
    # Real background consumed one word; key's 16 bytes must NOT consume two words.
    reference = DemoBytes()
    reference(8)
    words = [int.from_bytes(reference(8), "big") for _ in range(4)]
    assert collector.row(Cell(1, (0, 1)), ("null", "discovery", 0)) == aggregate_words(words)
    assert source(16) == b"DEMO" + (2).to_bytes(12, "big")


def test_hostile_ndarray_view_is_rejected_before_aggregation() -> None:
    view_calls = 0
    secret = "SECRET response view canary"

    class Hostile(np.ndarray):
        def view(self, *args: Any, **kwargs: Any) -> Any:
            nonlocal view_calls
            view_calls += 1
            raise RuntimeError(secret)

    response = np.zeros(4, dtype=np.uint64).view(Hostile)
    # All previously checked metadata is legitimate; only view() is hostile.
    assert response.dtype == np.dtype(np.uint64) and response.dtype.isnative
    assert response.ndim == 1 and response.shape == (4,) and response.flags.c_contiguous

    def oracle(points: Words, key: tuple[int, ...], rounds: int) -> Words:
        return response

    collector = CubeCollector(make_meter(), None, FiniteBytes([b"a"*16, b"b"*8]),
                              4, batch_oracle=oracle)
    with pytest.raises(Stopped) as caught:
        collector.row(Cell(1, (0, 1)), ("real", "confirm", 0))
    assert str(caught.value) == "ORACLE_RESPONSE" and caught.value.__cause__ is None
    assert view_calls == 0
    counts = collector.counts()
    assert secret not in str(caught.value) + repr(counts)
    assert counts["attempted_real_blocks"] == 4 and counts["completed_real_blocks"] == 0
    assert counts["attempted_real_groups"] == 1 and counts["completed_real_groups"] == 0


def test_hostile_ndarray_validation_error_is_fixed_public_stop() -> None:
    class Hostile(np.ndarray):
        @property
        def dtype(self) -> np.dtype[np.uint64]:
            raise RuntimeError("SECRET response canary")
    def oracle(points: Words, key: tuple[int, ...], rounds: int) -> Words:
        return np.zeros(len(points), dtype=np.uint64).view(Hostile)
    collector = CubeCollector(make_meter(), None, FiniteBytes([b"a"*16, b"b"*8]),
                              4, batch_oracle=oracle)
    with pytest.raises(Stopped) as caught:
        collector.row(Cell(1, (0, 1)), ("real", "confirm", 0))
    assert str(caught.value) == "ORACLE_RESPONSE" and caught.value.__cause__ is None
    counts = collector.counts()
    assert "SECRET" not in str(caught.value) + repr(counts)
    assert counts["attempted_real_blocks"] == 4 and counts["completed_real_blocks"] == 0
    assert counts["attempted_real_groups"] == 1 and counts["completed_real_groups"] == 0


@pytest.mark.parametrize("side", ["real", "null"])
def test_batch_initial_budget_refuses_before_any_private_input(side: str) -> None:
    source = FiniteBytes([])
    oracle = Capture()
    collector = CubeCollector(make_meter(real=0, null=0), None, source,
                              4, batch_oracle=oracle)
    with pytest.raises(Stopped):
        collector.row(Cell(1, (0, 1)), (side, "confirm", 0))
    assert source.requests == [] and oracle.calls == []
    assert collector.counts()[f"attempted_{side}_groups"] == 1
    assert collector.counts()["attempted_real_blocks"] == 0
    assert collector.counts()["attempted_null_words"] == 0


def test_oracle_cannot_redefine_requested_shape_by_mutating_input() -> None:
    def oracle(points: Words, key: tuple[int, ...], rounds: int) -> Words:
        points.resize((2,), refcheck=False)
        return np.zeros(2, dtype=np.uint64)
    collector = CubeCollector(make_meter(), None, FiniteBytes([b"a"*16, b"b"*8]),
                              4, batch_oracle=oracle)
    with pytest.raises(Stopped, match="ORACLE_RESPONSE"):
        collector.row(Cell(1, (0, 1)), ("real", "discovery", 0))
    assert collector.counts()["attempted_real_blocks"] == 4
    assert collector.counts()["completed_real_blocks"] == 0

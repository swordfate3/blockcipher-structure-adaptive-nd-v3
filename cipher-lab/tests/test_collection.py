"""Only finite fake randomness and fake oracles; production cipher never runs."""
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

import pytest

from cipher_lab.config import Cell, Policy
from cipher_lab.data import cubes
from cipher_lab.data.cubes import CubeCollector
from cipher_lab.execution import Clocks, Meter, Stopped


class FiniteBytes:
    def __init__(self, values: list[bytes]) -> None:
        self.values = iter(values)
        self.requests: list[int] = []

    def __call__(self, size: int) -> bytes:
        self.requests.append(size)
        value = next(self.values)
        assert len(value) == size
        return value


class FakeOracle:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []

    def __call__(self, points: tuple[int, ...], key: tuple[int, ...],
                 rounds: int) -> Sequence[int]:
        self.calls.append((points, key, rounds))
        return (0,) * len(points)


def make_meter(real: int = 200000, null: int = 200000) -> Meter:
    return Meter(Policy(2, 10, 10, "2031-01-01T00:00:00Z", real, null, 100, 100, 0, 0),
                 Clocks(lambda: 0.0, lambda: 0.0,
                        lambda: datetime(2030, 1, 1, tzinfo=UTC)), lambda: 0)


@pytest.fixture(autouse=True)
def forbid_cipher(monkeypatch: pytest.MonkeyPatch) -> None:
    # Importing this pure module is allowed; executing encryption is not.
    def forbidden(*args: object, **kwargs: object) -> tuple[int, ...]:
        raise AssertionError("production cipher called by an engineering test")
    monkeypatch.setattr("cipher_lab.ciphers.simon.encrypt_many", forbidden)


@pytest.mark.parametrize("chunk, sizes", [(1, [1, 1, 1, 1]), (3, [3, 1])])
@pytest.mark.parametrize("active", [(0, 2), (32, 34)])
def test_cube_fixed_background_key_and_msb0_points(
    chunk: int, sizes: list[int], active: tuple[int, ...],
) -> None:
    oracle = FakeOracle()
    source = FiniteBytes([bytes(range(16)), bytes.fromhex("0000000000000005")])
    collector = CubeCollector(make_meter(), oracle, source, chunk)
    assert collector.row(Cell(11, active), ("real", "discovery", 0)) == 0
    shift = 32 if active[0] == 32 else 0
    points = tuple(p for call in oracle.calls for p in call[0])
    assert set(points) == {5, 5 | (1 << (63-shift)), 5 | (1 << (61-shift)),
                           5 | (1 << (63-shift)) | (1 << (61-shift))}
    assert len(points) == len(set(points)) == 4
    assert [len(call[0]) for call in oracle.calls] == sizes
    assert all(call[1] == (0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203)
               and call[2] == 11 for call in oracle.calls)
    assert collector.counts() == {
        "attempted_real_blocks": 4, "completed_real_blocks": 4,
        "attempted_null_words": 0, "completed_null_words": 0,
        "attempted_real_groups": 1, "completed_real_groups": 1,
        "attempted_null_groups": 0, "completed_null_groups": 0,
    }


def test_chunk_65536_generates_entire_cube_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Boundary test isolates generation; it does NOT replace production 2080 semantics.
    consumed: list[int] = []
    def boundary_aggregate(words: Iterable[int]) -> int:
        consumed.append(sum(1 for _ in words))
        return 0
    monkeypatch.setattr(cubes, "aggregate_words", boundary_aggregate)
    oracle = FakeOracle()
    collector = CubeCollector(make_meter(), oracle, FiniteBytes([b"k"*16, b"\xff"*8]),
                              65536)
    collector.row(Cell(44, tuple(range(16))), ("real", "confirm", 0))
    assert [len(call[0]) for call in oracle.calls] == [65536]
    points = oracle.calls[0][0]
    assert len(set(points)) == 65536
    assert min(points) == 0x0000FFFFFFFFFFFF
    assert max(points) == 0xFFFFFFFFFFFFFFFF
    assert sum(consumed) == 65536


def test_null_uses_pointwise_words_not_fake_observation_and_never_real_key() -> None:
    # [0, MSB, next MSB, both]: unit parity zero; exactly pair(0,1) survives.
    source = FiniteBytes([word.to_bytes(8, "big") for word in (0, 1 << 63, 1 << 62,
                                                             3 << 62)])
    def no_call(points: tuple[int, ...], key: tuple[int, ...], rounds: int) -> Sequence[int]:
        raise AssertionError("null must not call oracle")
    collector = CubeCollector(make_meter(), no_call, source, 3)
    assert collector.row(Cell(1, (0, 1)), ("null", "screen", 0)) == 1 << 2015
    assert source.requests == [8, 8, 8, 8]
    counts = collector.counts()
    assert counts["completed_null_words"] == 4
    assert counts["completed_null_groups"] == 1
    assert counts["attempted_real_blocks"] == 0


def test_domains_include_canonical_cell_and_groups_have_independent_keys() -> None:
    oracle = FakeOracle()
    collector = CubeCollector(make_meter(), oracle,
                              FiniteBytes([b"a"*16, b"b"*8, b"c"*16, b"d"*8,
                                           b"e"*16, b"f"*8]), 4)
    collector.row(Cell(1, (0, 1)), ("real", "discovery", 0))
    with pytest.raises(Stopped, match="DOMAIN_REUSED"):
        collector.row(Cell(1, (1, 0)), ("real", "discovery", 0))
    collector.row(Cell(2, (0, 1)), ("real", "discovery", 0))
    collector.row(Cell(1, (0, 1)), ("real", "screen", 0))
    assert len({call[1] for call in oracle.calls}) == 3


@pytest.mark.parametrize("domain", [("fake", "discovery", 0), ("real", "bad", 0),
                                    ("real", "screen", -1), ("null", "confirm", True)])
def test_invalid_domains_do_not_call_oracle(domain: tuple[str, str, int]) -> None:
    oracle = FakeOracle()
    collector = CubeCollector(make_meter(), oracle, FiniteBytes([]), 4)
    with pytest.raises(ValueError):
        collector.row(Cell(1, (0, 1)), domain)
    assert not oracle.calls


def test_budget_checked_before_randomness_and_oracle() -> None:
    source = FiniteBytes([])
    oracle = FakeOracle()
    collector = CubeCollector(make_meter(real=0), oracle, source, 4)
    with pytest.raises(Stopped, match="REAL_BLOCKS_LIMIT"):
        collector.row(Cell(1, (0, 1)), ("real", "discovery", 0))
    assert not oracle.calls and not source.requests


def test_duplicate_key_bounded_failure() -> None:
    oracle = FakeOracle()
    # An infinite deterministic duplicate provider must still terminate.
    def duplicates(size: int) -> bytes:
        return b"x" * size
    collector = CubeCollector(make_meter(), oracle, duplicates, 4)
    collector.row(Cell(1, (0, 1)), ("real", "discovery", 0))
    with pytest.raises(Stopped, match="KEY_REUSE_LIMIT"):
        collector.row(Cell(1, (0, 1)), ("real", "discovery", 1))
    assert len(oracle.calls) == 1
    assert collector.counts()["attempted_real_blocks"] == 8


@pytest.mark.parametrize("response", [(), (0,), (0, 0, 0, -1), (0, 0, 0, 1 << 64),
                                     (0, 0, 0, True)])
def test_bad_response_keeps_attempts_not_completed(response: tuple[int, ...]) -> None:
    def bad(points: tuple[int, ...], key: tuple[int, ...], rounds: int) -> Sequence[int]:
        return response
    collector = CubeCollector(make_meter(), bad, FiniteBytes([b"a"*16, b"b"*8]), 4)
    with pytest.raises(Stopped, match="ORACLE_RESPONSE"):
        collector.row(Cell(1, (0, 1)), ("real", "confirm", 0))
    assert collector.counts()["attempted_real_blocks"] == 4
    assert collector.counts()["completed_real_blocks"] == 0
    assert collector.counts()["completed_real_groups"] == 0


def test_private_oracle_exception_never_propagates_text_and_domain_stays_used() -> None:
    def broken(points: tuple[int, ...], key: tuple[int, ...], rounds: int) -> Sequence[int]:
        raise RuntimeError("SECRET-KEY-CONTENT")
    collector = CubeCollector(make_meter(), broken, FiniteBytes([b"a"*16, b"b"*8]), 4)
    with pytest.raises(Stopped) as caught:
        collector.row(Cell(1, (0, 1)), ("real", "confirm", 0))
    assert str(caught.value) == "ORACLE_FAILED"
    assert caught.value.__cause__ is None
    assert "SECRET" not in repr(collector.counts())
    assert collector.counts()["attempted_real_blocks"] == 4
    assert collector.counts()["completed_real_blocks"] == 0
    with pytest.raises(Stopped, match="DOMAIN_REUSED"):
        collector.row(Cell(1, (0, 1)), ("real", "confirm", 0))


@pytest.mark.parametrize("side, values", [("real", []), ("null", [])])
def test_exhausted_random_is_fixed_error(side: str, values: list[bytes]) -> None:
    collector = CubeCollector(make_meter(), FakeOracle(), FiniteBytes(values), 4)
    with pytest.raises(Stopped, match="RANDOM_UNAVAILABLE"):
        collector.row(Cell(1, (0, 1)), (side, "discovery", 0))
    assert collector.counts()[f"attempted_{'real_blocks' if side == 'real' else 'null_words'}"] == 4


def test_partial_chunk_progress_survives_later_failure() -> None:
    calls = 0
    def oracle(points: tuple[int, ...], key: tuple[int, ...], rounds: int) -> Sequence[int]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("private")
        return (0,) * len(points)
    collector = CubeCollector(make_meter(), oracle, FiniteBytes([b"a"*16, b"b"*8]), 3)
    with pytest.raises(Stopped, match="ORACLE_FAILED"):
        collector.row(Cell(1, (0, 1)), ("real", "discovery", 0))
    assert collector.counts()["attempted_real_blocks"] == 4
    assert collector.counts()["completed_real_blocks"] == 3


def test_deadline_during_private_preparation_prevents_oracle_call() -> None:
    wall = 0.0
    meter = Meter(Policy(2, 1, 10, "2031-01-01T00:00:00Z", 100, 100, 100, 100, 0, 0),
                  Clocks(lambda: wall, lambda: 0.0,
                         lambda: datetime(2030, 1, 1, tzinfo=UTC)), lambda: 0)
    def slow_random(size: int) -> bytes:
        nonlocal wall
        if size == 8:
            wall = 2.0
        return b"a" * size
    oracle = FakeOracle()
    collector = CubeCollector(meter, oracle, slow_random, 4)
    with pytest.raises(Stopped, match="WALL_LIMIT"):
        collector.row(Cell(1, (0, 1)), ("real", "discovery", 0))
    assert not oracle.calls
    assert collector.counts()["attempted_real_blocks"] == 4
    assert collector.counts()["completed_real_blocks"] == 0


def test_real_aggregates_nonzero_pointwise_responses() -> None:
    def response(points: tuple[int, ...], key: tuple[int, ...], rounds: int) -> Sequence[int]:
        assert len(points) == 4
        return (0, 1 << 63, 1 << 62, 3 << 62)
    collector = CubeCollector(make_meter(), response, FiniteBytes([b"a"*16, b"b"*8]), 4)
    assert collector.row(Cell(11, (0, 1)), ("real", "screen", 0)) == 1 << 2015


def test_second_chunk_budget_refusal_keeps_first_chunk_progress() -> None:
    oracle = FakeOracle()
    collector = CubeCollector(make_meter(real=3), oracle,
                              FiniteBytes([b"a"*16, b"b"*8]), 3)
    with pytest.raises(Stopped, match="REAL_BLOCKS_LIMIT"):
        collector.row(Cell(1, (0, 1)), ("real", "screen", 0))
    assert len(oracle.calls) == 1
    assert collector.counts()["attempted_real_blocks"] == 3
    assert collector.counts()["completed_real_blocks"] == 3
    assert collector.counts()["completed_real_groups"] == 0


def test_duplicate_key_is_resampled_then_new_key_is_used() -> None:
    oracle = FakeOracle()
    collector = CubeCollector(make_meter(), oracle, FiniteBytes([
        b"a"*16, b"b"*8, b"a"*16, b"c"*16, b"d"*8,
    ]), 4)
    collector.row(Cell(1, (0, 1)), ("real", "discovery", 0))
    collector.row(Cell(1, (0, 1)), ("real", "discovery", 1))
    assert oracle.calls[0][1] != oracle.calls[1][1]
    assert collector.counts()["completed_real_groups"] == 2


@pytest.mark.parametrize("raw", [b"", b"short", bytearray(16)])
def test_malformed_private_bytes_have_fixed_error(raw: bytes) -> None:
    collector = CubeCollector(make_meter(), FakeOracle(), lambda _: raw, 4)
    with pytest.raises(Stopped, match="RANDOM_UNAVAILABLE"):
        collector.row(Cell(1, (0, 1)), ("real", "discovery", 0))
    assert collector.counts()["completed_real_blocks"] == 0


def test_monitor_expiry_after_oracle_preserves_completed_work() -> None:
    wall = 0.0
    meter = Meter(Policy(2, 1, 10, "2031-01-01T00:00:00Z", 100, 100, 100, 100, 0, 0),
                  Clocks(lambda: wall, lambda: 0.0,
                         lambda: datetime(2030, 1, 1, tzinfo=UTC)), lambda: 0)
    def slow_oracle(points: tuple[int, ...], key: tuple[int, ...], rounds: int) -> Sequence[int]:
        nonlocal wall
        wall = 2.0
        return (0,) * len(points)
    collector = CubeCollector(meter, slow_oracle, FiniteBytes([b"a"*16, b"b"*8]), 4)
    with pytest.raises(Stopped, match="WALL_LIMIT"):
        collector.row(Cell(1, (0, 1)), ("real", "discovery", 0))
    assert collector.counts()["completed_real_blocks"] == 4
    assert collector.counts()["completed_real_groups"] == 0

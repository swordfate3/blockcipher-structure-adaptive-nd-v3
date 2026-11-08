"""Resource checks use synthetic clocks/RSS only; no scientific backend."""
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter, Stopped, production_rss_bytes

NOW = datetime(2030, 1, 1, tzinfo=UTC)


def policy(**changes: object) -> Policy:
    base = Policy(2, 10.0, 8.0, "2030-01-01T00:00:20Z", 12, 16, 1000, 1000, 0, 0)
    return replace(base, **changes)  # type: ignore[arg-type]


class FakeTime:
    wall = 100.0
    cpu = 30.0
    utc = NOW
    rss = 100

    def meter(self, settings: Policy | None = None) -> Meter:
        return Meter(settings or policy(), Clocks(
            lambda: self.wall, lambda: self.cpu, lambda: self.utc,
        ), lambda: self.rss)


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("wall", 110.0, "WALL_LIMIT"), ("cpu", 38.0, "CPU_LIMIT"),
    ("utc", NOW + timedelta(seconds=20), "DEADLINE"),
    ("rss", 1001, "RSS_LIMIT"),
])
def test_checks_stop_before_reservation(field: str, value: object, reason: str) -> None:
    clock = FakeTime()
    meter = clock.meter()
    setattr(clock, field, value)
    with pytest.raises(Stopped) as caught:
        meter.reserve("real_blocks", 1)
    assert caught.value.reason == reason
    assert meter.snapshot()["attempted_real_blocks"] == 0


def test_reserve_complete_units_and_non_refundable_attempts() -> None:
    meter = FakeTime().meter()
    meter.reserve("real_blocks", 7)
    meter.complete("real_blocks", 3)
    meter.reserve("null_words", 5)
    meter.complete("null_words", 5)
    with pytest.raises(ValueError):
        meter.complete("real_blocks", 5)
    snapshot = meter.snapshot()
    assert snapshot["attempted_real_blocks"] == 7
    assert snapshot["completed_real_blocks"] == 3
    assert snapshot["attempted_null_words"] == snapshot["completed_null_words"] == 5
    with pytest.raises(Stopped, match="REAL_BLOCKS_LIMIT"):
        meter.reserve("real_blocks", 6)
    assert meter.snapshot()["attempted_real_blocks"] == 7


@pytest.mark.parametrize("amount", [-1, True, 1.5])
def test_amounts_are_nonnegative_integers(amount: int) -> None:
    meter = FakeTime().meter()
    with pytest.raises(ValueError):
        meter.reserve("real_blocks", amount)
    with pytest.raises(ValueError):
        meter.complete("real_blocks", amount)


def test_null_limit_is_independent() -> None:
    meter = FakeTime().meter(policy(null_words=0))
    with pytest.raises(Stopped, match="NULL_WORDS_LIMIT"):
        meter.reserve("null_words", 1)
    assert meter.snapshot()["attempted_null_words"] == 0


def test_failed_monitor_is_private_and_fail_closed() -> None:
    def broken() -> int:
        raise RuntimeError("private monitor detail")
    meter = Meter(policy(), Clocks(lambda: 1.0, lambda: 1.0, lambda: NOW), broken)
    with pytest.raises(Stopped) as caught:
        meter.check()
    assert str(caught.value) == "RSS_UNAVAILABLE"
    assert "private" not in repr(meter.snapshot())


def test_unsupported_production_rss_is_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cipher_lab.execution.sys.platform", "unsupported")
    with pytest.raises(Stopped, match="RSS_UNSUPPORTED"):
        production_rss_bytes()


@pytest.mark.parametrize("rss", [-1, True])
def test_invalid_rss_fails_closed(rss: int) -> None:
    meter = Meter(policy(), Clocks(lambda: 1.0, lambda: 1.0, lambda: NOW), lambda: rss)
    with pytest.raises(Stopped, match="RSS_UNAVAILABLE"):
        meter.check()


def test_complete_does_not_lose_work_when_clock_expires() -> None:
    clock = FakeTime()
    meter = clock.meter()
    meter.reserve("real_blocks", 4)
    clock.wall += 11
    meter.complete("real_blocks", 4)
    with pytest.raises(Stopped):
        meter.check()
    assert meter.snapshot()["completed_real_blocks"] == 4


def test_clock_exception_is_sanitized() -> None:
    def broken() -> float:
        raise RuntimeError("private clock detail")
    clock: Callable[[], float] = broken
    with pytest.raises(Stopped, match="CLOCK_UNAVAILABLE"):
        Meter(policy(), Clocks(clock, lambda: 1.0, lambda: NOW), lambda: 0)

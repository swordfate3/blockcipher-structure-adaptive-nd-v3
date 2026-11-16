"""Cooperative resource checks and non-refundable unit reservations, not science.

Linux current RSS is supported via procfs; other platforms fail closed. Checks
are cooperative, not a hard memory/process sandbox. Snapshot is side-effect-free
and reports the last successful clock/RSS sample, even after a stop.
"""
from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from cipher_lab.config import Policy

Kind = Literal["real_blocks", "null_words", "optimizer_updates", "forward_rows"]
_KINDS: tuple[Kind, ...] = ("real_blocks", "null_words", "optimizer_updates", "forward_rows")
_REASONS = frozenset({
    "WALL_LIMIT", "CPU_LIMIT", "DEADLINE", "RSS_LIMIT", "RSS_UNSUPPORTED",
    "RSS_UNAVAILABLE", "CLOCK_UNAVAILABLE", "REAL_BLOCKS_LIMIT", "NULL_WORDS_LIMIT",
    "DOMAIN_REUSED", "RANDOM_UNAVAILABLE", "KEY_REUSE_LIMIT", "ORACLE_FAILED",
    "ORACLE_RESPONSE", "OPTIMIZER_UPDATES_LIMIT", "FORWARD_ROWS_LIMIT",
    "ARM_WALL_LIMIT", "ARM_CPU_LIMIT",
})


class Stopped(Exception):
    """Only fixed, public reason codes may cross a private boundary."""

    def __init__(self, reason: str) -> None:
        if reason not in _REASONS:
            raise ValueError("unknown public stop reason")
        self.reason = reason
        super().__init__(reason)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Clocks:
    """Wall/CPU seconds and timezone-aware UTC; inject all three for tests."""

    wall: Callable[[], float] = time.monotonic
    cpu: Callable[[], float] = time.process_time
    utc: Callable[[], datetime] = _utc_now


def production_rss_bytes() -> int:
    """Read Linux resident pages (not peak RSS); unsupported is never zero.

    Windows/macOS are explicitly unsupported by this implementation. procfs
    errors stop execution rather than silently disabling monitoring.
    """
    if sys.platform != "linux":
        raise Stopped("RSS_UNSUPPORTED")
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            pages = int(stream.read(4096).split()[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages < 0 or page_size <= 0:
            raise ValueError
        return pages * page_size
    except Exception:
        raise Stopped("RSS_UNAVAILABLE") from None


class Meter:
    """Single-threaded ledger of blocks, null draws, updates and forward rows.

    Null draws are 64-bit words for integral data, one draw per bit label for
    prediction. Forward rows count model samples, not their component pairs.

    reserve checks resources BEFORE charging attempts. A failed external call
    cannot refund them. complete credits validated work only, never more than
    cumulative reservations; it deliberately does not check time, so already
    completed work can be recorded before the caller's next check.
    """

    def __init__(self, policy: Policy, clocks: Clocks,
                 rss_bytes: Callable[[], int], *, enforce_cpu: bool = True) -> None:
        self._enforce_cpu = enforce_cpu
        self._policy = policy
        self._clocks = clocks
        self._rss_bytes = rss_bytes
        self._attempted: dict[Kind, int] = dict.fromkeys(_KINDS, 0)
        self._completed: dict[Kind, int] = dict.fromkeys(_KINDS, 0)
        wall, cpu, _ = self._sample_clocks()
        self._start_wall, self._start_cpu = wall, cpu
        self._wall_elapsed = self._cpu_elapsed = 0.0
        self._rss: int | None = None
        self._deadline = datetime.fromisoformat(
            policy.deadline_utc[:-1] + "+00:00"
            if policy.deadline_utc.endswith("Z") else policy.deadline_utc)
        self._active_budget: CumulativeBudget | None = None

    def cumulative_budget(self, *, wall_seconds: float, cpu_seconds: float) -> CumulativeBudget:
        """A serial active-time accumulator on this meter, not another executor."""
        return CumulativeBudget(self, wall_seconds, cpu_seconds)

    def _sample_clocks(self) -> tuple[float, float, datetime]:
        try:
            wall, cpu, utc = self._clocks.wall(), self._clocks.cpu(), self._clocks.utc()
            if (type(wall) not in (int, float) or type(cpu) not in (int, float)
                    or not math.isfinite(wall) or not math.isfinite(cpu)
                    or not isinstance(utc, datetime) or utc.utcoffset() != timedelta(0)):
                raise ValueError
            return float(wall), float(cpu), utc
        except Exception:
            raise Stopped("CLOCK_UNAVAILABLE") from None

    def check(self) -> None:
        wall, cpu, utc = self._sample_clocks()
        elapsed_wall, elapsed_cpu = wall - self._start_wall, cpu - self._start_cpu
        if elapsed_wall < self._wall_elapsed or elapsed_cpu < self._cpu_elapsed:
            raise Stopped("CLOCK_UNAVAILABLE")
        self._wall_elapsed, self._cpu_elapsed = elapsed_wall, elapsed_cpu
        if self._active_budget is not None:
            self._active_budget._check(wall, cpu)
        if utc >= self._deadline:
            raise Stopped("DEADLINE")
        if elapsed_wall >= self._policy.wall_seconds:
            raise Stopped("WALL_LIMIT")
        if self._enforce_cpu and elapsed_cpu >= self._policy.cpu_seconds:
            raise Stopped("CPU_LIMIT")
        try:
            rss = self._rss_bytes()
            if type(rss) is not int or rss < 0:
                raise ValueError
        except Stopped as error:
            if error.reason == "RSS_UNSUPPORTED":
                raise Stopped("RSS_UNSUPPORTED") from None
            raise Stopped("RSS_UNAVAILABLE") from None
        except Exception:
            raise Stopped("RSS_UNAVAILABLE") from None
        self._rss = rss
        if rss > self._policy.rss_stop_bytes:
            raise Stopped("RSS_LIMIT")

    @staticmethod
    def _validate(kind: Kind, amount: int) -> None:
        if kind not in _KINDS:
            raise ValueError("unknown reservation unit")
        if type(amount) is not int or amount < 0:
            raise ValueError("amount must be a nonnegative integer")

    def reserve(self, kind: Kind, amount: int) -> None:
        self._validate(kind, amount)
        self.check()
        if self._attempted[kind] + amount > getattr(self._policy, kind):
            raise Stopped(f"{kind.upper()}_LIMIT")
        self._attempted[kind] += amount

    def complete(self, kind: Kind, amount: int) -> None:
        self._validate(kind, amount)
        if self._completed[kind] + amount > self._attempted[kind]:
            raise ValueError("completion exceeds reservation")
        self._completed[kind] += amount

    def snapshot(self) -> dict[str, object]:
        return {
            **{f"{status}_{kind}": counts[kind]
               for status, counts in (
                   ("attempted", self._attempted), ("completed", self._completed))
               for kind in _KINDS},
            "wall_elapsed_seconds": self._wall_elapsed,
            "cpu_elapsed_seconds": self._cpu_elapsed,
            "rss_bytes": self._rss,
        }


class CumulativeBudget:
    """Accumulate only active serial segments using the parent's sampled clocks."""

    def __init__(self, meter: Meter, wall_seconds: float, cpu_seconds: float) -> None:
        if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0
               for v in (wall_seconds, cpu_seconds)):
            raise ValueError('invalid cumulative limits')
        self.meter = meter
        self.limits = wall_seconds, cpu_seconds
        self.total = [0., 0.]
        self.start: tuple[float, float] | None = None
        self.latest: tuple[float, float] | None = None

    def __enter__(self) -> CumulativeBudget:
        if self.meter._active_budget is not None or self.start is not None:
            raise ValueError('nested cumulative budget')
        self.meter.check()
        self.start = self.latest = (self.meter._start_wall + self.meter._wall_elapsed,
                                   self.meter._start_cpu + self.meter._cpu_elapsed)
        self.meter._active_budget = self
        return self

    def _check(self, wall: float, cpu: float) -> None:
        assert self.start is not None
        self.latest = wall, cpu
        elapsed = [self.total[i] + self.latest[i] - self.start[i] for i in range(2)]
        for index, reason in enumerate(('ARM_WALL_LIMIT', 'ARM_CPU_LIMIT')):
            if elapsed[index] >= self.limits[index]:
                raise Stopped(reason)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            try:
                self.meter.check()
            except Stopped:
                # Keep an already propagating storage/identity/failure exception.
                # Successful segments must still fail closed on clocks or limits.
                if exc_type is None:
                    raise
        finally:
            assert self.start is not None and self.latest is not None
            for i in range(2):
                self.total[i] += self.latest[i] - self.start[i]
            self.start = None
            self.meter._active_budget = None

    def snapshot(self) -> dict[str, object]:
        return {'wall_seconds': self.total[0], 'cpu_seconds': self.total[1],
                'wall_limit_seconds': self.limits[0], 'cpu_limit_seconds': self.limits[1]}

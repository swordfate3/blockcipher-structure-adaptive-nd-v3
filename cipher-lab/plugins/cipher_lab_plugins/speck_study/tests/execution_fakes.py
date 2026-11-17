"""Resource checks use synthetic clocks/RSS only; no scientific backend."""
from dataclasses import replace
from datetime import UTC, datetime

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter

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



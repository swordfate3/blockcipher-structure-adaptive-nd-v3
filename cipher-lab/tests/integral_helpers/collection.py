"""Finite bytes and static clocks for artificial cube tests."""
from datetime import UTC, datetime

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter


class FiniteBytes:
    def __init__(self, values: list[bytes]) -> None:
        self.values = iter(values)
        self.requests: list[int] = []

    def __call__(self, size: int) -> bytes:
        self.requests.append(size)
        value = next(self.values)
        assert len(value) == size
        return value


def make_meter(real: int = 200000, null: int = 200000) -> Meter:
    return Meter(Policy(2, 10, 10, "2031-01-01T00:00:00Z", real, null, 100, 100, 0, 0),
                 Clocks(lambda: 0.0, lambda: 0.0,
                        lambda: datetime(2030, 1, 1, tzinfo=UTC)), lambda: 0)



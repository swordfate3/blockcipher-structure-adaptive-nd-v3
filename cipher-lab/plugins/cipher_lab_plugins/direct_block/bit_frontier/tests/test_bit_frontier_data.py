from __future__ import annotations

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.bit_frontier import data
from cipher_lab_plugins.direct_block.bit_structure import config
from cipher_lab_plugins.direct_block.bit_structure.data import Key, Words

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter, Stopped

ROUNDS = (4, 5, 6, 7, 8)
SIZES = {"train": 3, "dev": 2, "test": 4}


@pytest.fixture(autouse=True)
def tiny(monkeypatch: pytest.MonkeyPatch) -> None:
    for split, size in SIZES.items():
        monkeypatch.setattr(config, split.upper(), size)


def meter(blocks: int = 90) -> Meter:
    return Meter(Policy(2, 60, 60, "2099-01-01T00:00:00Z", blocks, 0, 1024, 0, 0, 0),
                 Clocks(), lambda: 0)


def sample(rows: int, ledger: Meter) -> Words:
    assert rows == 50
    return np.arange(rows, dtype=np.uint32)


def test_cycle_partitions_and_lazy_labels() -> None:
    ledger = meter()
    draws: list[int] = []
    calls: list[tuple[Words, Key, int]] = []
    keys = iter((1, 2))

    def draw(bits: int) -> int:
        draws.append(bits)
        return next(keys)

    def oracle(rows: Words, key: Key, rounds: int) -> Words:
        calls.append((rows, key, rounds))
        parity = (rows + np.uint32(rounds + key[0])) & np.uint32(1)
        return ((parity << np.uint32(31)) | np.uint32(1)).astype(np.uint32)

    cycle = data.make_panels(ledger, draw, oracle, sample=sample)
    unique_parts: list[Words] = []
    for rounds in ROUNDS:
        panels = cycle.for_round(rounds)
        for repeat in range(2):
            for split, size in SIZES.items():
                rows = panels.points(repeat, split)
                assert rows.shape == (size,) and rows.dtype == np.uint32
                with pytest.raises(ValueError):
                    rows.setflags(write=True)
                if split == "test" or rounds == ROUNDS[0]:
                    unique_parts.append(rows)
                else:
                    assert rows is cycle.for_round(ROUNDS[0]).points(repeat, split)
    assert len(unique_parts) == 14
    assert len(np.unique(np.concatenate(unique_parts))) == 50
    assert draws == [64, 64] and not calls
    assert ledger.snapshot()["attempted_real_blocks"] == 0

    for rounds in ROUNDS:
        panels = cycle.for_round(rounds)
        for repeat in range(2):
            for split, size in SIZES.items():
                labels = panels.label(repeat, split)
                rows, key, actual_round = calls[-1]
                assert rows is panels.points(repeat, split)
                assert key == (repeat + 1, 0, 0, 0) and actual_round == rounds
                assert labels.shape == (size, 1) and labels.dtype == np.uint8
                assert not labels.flags.writeable
                np.testing.assert_array_equal(labels[:, 0], (rows + rounds + repeat + 1) % 2)
                with pytest.raises(Stopped, match="DOMAIN_REUSED"):
                    cycle.for_round(rounds).label(repeat, split)
    assert len(calls) == 30
    assert ledger.snapshot()["attempted_real_blocks"] == 90
    assert ledger.snapshot()["completed_real_blocks"] == 90
    with pytest.raises(ValueError):
        cycle.for_round(3)
    with pytest.raises(ValueError):
        cycle.for_round(4).points(True, "train")


@pytest.mark.parametrize("fault", ["alias", "dtype", "shape", "stride", "exception", "budget"])
def test_failed_label_is_consumed(fault: str) -> None:
    ledger = meter(0 if fault == "budget" else 90)
    calls = 0

    def oracle(rows: Words, key: Key, rounds: int) -> Words:
        nonlocal calls
        calls += 1
        if fault == "exception":
            raise RuntimeError("private detail")
        if fault == "alias":
            return rows.view()
        if fault == "dtype":
            return rows.astype(np.uint8)
        if fault == "shape":
            return rows[:-1].copy()
        return np.zeros(2 * len(rows), dtype=np.uint32)[::2]

    keys = iter((1, 2))
    panels = data.make_panels(ledger, lambda _: next(keys), oracle, sample=sample).for_round(4)
    reason = {"exception": "ORACLE_FAILED", "budget": "REAL_BLOCKS_LIMIT"}.get(
        fault, "ORACLE_RESPONSE")
    with pytest.raises(Stopped, match=f"^{reason}$"):
        panels.label(0, "train")
    with pytest.raises(Stopped, match="DOMAIN_REUSED"):
        panels.label(0, "train")
    assert calls == (0 if fault == "budget" else 1)
    assert ledger.snapshot()["attempted_real_blocks"] == 3 * calls
    assert ledger.snapshot()["completed_real_blocks"] == 0


def test_invalid_plaintext_and_duplicate_keys() -> None:
    def invalid_sample(rows: int, ledger: Meter) -> Words:
        points = sample(rows, ledger)
        points[-1] = points[0]
        return points

    def unexpected_draw(bits: int) -> int:
        pytest.fail("invalid sample reached the key draw")

    def unexpected_oracle(rows: Words, key: Key, rounds: int) -> Words:
        pytest.fail("setup reached the oracle")

    with pytest.raises(ValueError, match="plaintext"):
        data.make_panels(meter(), unexpected_draw, unexpected_oracle, sample=invalid_sample)
    with pytest.raises(Stopped, match="RANDOM_UNAVAILABLE"):
        data.make_panels(meter(), lambda _: 1, unexpected_oracle, sample=sample)

"""High cubes use finite artificial inputs only; k20 tests structure, not aggregation."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from integral_helpers.collection import FiniteBytes, make_meter
from numpy.typing import NDArray

from cipher_lab.config import Cell, Recipe, StageSizes
from cipher_lab.data import cubes
from cipher_lab.data.cubes import CubeCollector
from cipher_lab.execution import Stopped

Words = NDArray[np.uint64]
PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_production(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("production cipher/private RNG/RSS called")
    monkeypatch.setattr("cipher_lab.ciphers.simon.encrypt_many", forbidden)
    monkeypatch.setattr("cipher_lab.ciphers.simon.encrypt_batch", forbidden)
    monkeypatch.setattr("secrets.token_bytes", forbidden)
    monkeypatch.setattr("cipher_lab.execution.production_rss_bytes", forbidden)


@pytest.mark.parametrize("k", [18, 20])
@pytest.mark.parametrize("half", [0, 32])
def test_high_active_cell_valid(k: int, half: int) -> None:
    active = tuple(range(half, half + k))
    assert Cell(11, active).to_dict() == {"rounds": 11, "active": list(active)}


def test_cell_21_still_rejected() -> None:
    with pytest.raises(ValueError, match="active"):
        Cell(11, tuple(range(21)))


@pytest.mark.parametrize("k", [17, 18, 20])
def test_ordinary_recipe_explicitly_keeps_16_limit(k: int) -> None:
    # Construct a legal sampling Cell first: Recipe, not Cell, must reject it.
    cell = Cell(11, tuple(range(k)))
    ordinary = Recipe(1, "integral-empirical-v1", (Cell(11, (0, 1)),), StageSizes(1, 1, 1), 4)
    with pytest.raises(ValueError, match="active"):
        replace(ordinary, cells=(cell,))


class StreamingOracle:
    """Check one bounded array at a time, retaining only chunk endpoints/counts.

    Outputs are artificial polynomials in two bits on the first four assignments
    of the entire cube. They vanish elsewhere, so global parity is nontrivial
    despite the even number of chunks. No actual encryption is performed.
    """

    def __init__(self, k: int, half: int) -> None:
        self.k, self.half = k, half
        self.offset = 0
        self.ends: list[tuple[int, int]] = []
        self.background = 0xFEDCBA9876543210
        self.mask = ((1 << k) - 1) << (64 - half - k)
        self.base = self.background & ~self.mask

    def point(self, assignment: int) -> int:
        # Independent scalar endpoint formula; not production _points.
        return self.base | sum(((assignment >> i) & 1) << (63 - self.half - i)
                               for i in range(self.k))

    def __call__(self, points: Words, key: tuple[int, ...], rounds: int) -> Words:
        assert type(points) is np.ndarray and points.dtype == np.dtype(np.uint64)
        assert points.ndim == 1 and points.flags.c_contiguous
        assert len(points) == 65536
        assert key == (0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203)
        assert rounds == 11
        assert int(points[0]) == self.point(self.offset)
        assert int(points[-1]) == self.point(self.offset + len(points) - 1)
        assert np.all((points & np.uint64(~self.mask & ((1 << 64) - 1))) == self.base)
        # Invert MSB0 coordinates inside the chunk. Exact range equality proves
        # ordering and uniqueness across chunks without storing a cube or set.
        assignments = np.zeros(len(points), dtype=np.uint64)
        for i in range(self.k):
            assignments |= ((points >> np.uint64(63 - self.half - i)) & np.uint64(1)) << i
        assert np.array_equal(assignments, np.arange(self.offset, self.offset + len(points),
                                                   dtype=np.uint64))
        output = np.zeros(len(points), dtype=np.uint64)
        if self.offset == 0:
            a = assignments[:4] & np.uint64(1)
            b = (assignments[:4] >> np.uint64(1)) & np.uint64(1)
            output[:4] = a | ((a & b) << np.uint64(1))
        self.ends.append((int(points[0]), int(points[-1])))
        self.offset += len(points)
        return output


def independent_formula_expectation() -> int:
    # Compute all 2080 terms independently on just the four nonzero-support
    # assignments, using public integer arithmetic, not either aggregator.
    row = 0
    for i in range(64):
        parity = 0
        for a in range(2):
            for b in range(2):
                word = a | ((a & b) << 1)
                parity ^= (word >> (63 - i)) & 1
        row = (row << 1) | parity
    for i in range(64):
        for j in range(i + 1, 64):
            parity = 0
            for a in range(2):
                for b in range(2):
                    word = a | ((a & b) << 1)
                    parity ^= ((word >> (63 - i)) & 1) & ((word >> (63 - j)) & 1)
            row = (row << 1) | parity
    # Unit 62 and pair (62,63) survive: tests products BEFORE XOR.
    assert row == (1 << 2017) | 1
    return row


@pytest.mark.parametrize("k,half", [(18, 0), (18, 32), (20, 0), (20, 32)])
def test_high_cube_streams_all_chunks_with_full_k18_aggregation(
    monkeypatch: pytest.MonkeyPatch, k: int, half: int,
) -> None:
    oracle = StreamingOracle(k, half)
    source = FiniteBytes([bytes(range(16)), oracle.background.to_bytes(8, "big")])
    if k == 20:
        # k20 is deliberately ONLY enumeration/accounting structure coverage.
        def structural_only(words: Words, *, check: Callable[[], None] | None = None) -> int:
            assert words.shape == (65536,)
            assert check is not None
            check()
            return 0
        monkeypatch.setattr(cubes, "aggregate_batch", structural_only)
    collector = CubeCollector(make_meter(real=1 << k, null=0), None, source, 65536,
                              batch_oracle=oracle)
    domain = ("real", "discovery", 0)
    row = collector.row(Cell(11, tuple(range(half, half + k))), domain)
    assert row == (independent_formula_expectation() if k == 18 else 0)
    assert oracle.offset == 1 << k
    assert len(oracle.ends) == (4 if k == 18 else 16)
    assert oracle.ends[0][0] == oracle.base
    assert oracle.ends[-1][1] == oracle.base | oracle.mask
    assert source.requests == [16, 8]
    counts = collector.counts()
    assert counts["attempted_real_blocks"] == counts["completed_real_blocks"] == 1 << k
    assert counts["attempted_real_groups"] == counts["completed_real_groups"] == 1
    assert counts["attempted_null_words"] == counts["completed_null_words"] == 0
    with pytest.raises(Stopped, match="DOMAIN_REUSED"):
        collector.row(Cell(11, tuple(reversed(range(half, half + k)))), domain)
    assert collector.counts() == counts
    assert source.requests == [16, 8]


@pytest.mark.parametrize("k", [18, 20])
def test_high_cube_second_reserve_failure_keeps_first_chunk_but_no_group(k: int) -> None:
    oracle = StreamingOracle(k, 0)
    source = FiniteBytes([bytes(range(16)), oracle.background.to_bytes(8, "big")])
    collector = CubeCollector(make_meter(real=65536, null=0), None, source, 65536,
                              batch_oracle=oracle)
    cell, domain = Cell(11, tuple(range(k))), ("real", "discovery", 0)
    with pytest.raises(Stopped, match="REAL_BLOCKS_LIMIT"):
        collector.row(cell, domain)
    assert oracle.offset == 65536 and len(oracle.ends) == 1
    assert source.requests == [16, 8]
    counts = collector.counts()
    assert counts["attempted_real_blocks"] == counts["completed_real_blocks"] == 65536
    assert counts["attempted_real_groups"] == 1 and counts["completed_real_groups"] == 0
    with pytest.raises(Stopped, match="DOMAIN_REUSED"):
        collector.row(cell, domain)
    assert collector.counts() == counts



from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Never, cast

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.full32_frontier import config, data

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter, Stopped

ROUNDS = (3, 4, 5, 6)
SIZES = {'train': 3, 'dev': 2, 'test': 4}


@pytest.fixture(autouse=True)
def tiny(monkeypatch: pytest.MonkeyPatch) -> None:
    for split, size in SIZES.items():
        monkeypatch.setattr(config, split.upper(), size)


def meter(blocks: int = 72) -> Meter:
    return Meter(Policy(2, 60, 60, '2099-01-01T00:00:00Z', blocks, 0, 1024, 0, 0, 0),
                 Clocks(), lambda: 0)


def sample(rows: int, ledger: Meter) -> data.Words:
    assert rows == 42
    return np.arange(rows, dtype=np.uint32) + np.uint32(0xffffffff - rows + 1)


def keys() -> Callable[[int], int]:
    values = iter((0xFEDCBA9876543210, 2))

    def draw(bits: int) -> int:
        assert bits == 64
        return next(values)

    return draw


def test_public_sampler() -> None:
    for invalid in (-1, True, 1.5, 2**32 + 1):
        with pytest.raises(ValueError):
            data.sample_points(cast(int, invalid), meter())


def test_sampler_requests_full_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRandom:
        def sample(self, domain: range, size: int) -> list[int]:
            assert domain == range(2**32) and size == 2
            return [0, 2**32 - 1]

    monkeypatch.setattr(data, 'SystemRandom', FakeRandom)
    points = data.sample_points(2, meter())
    assert points.dtype == np.uint32 and points.tolist() == [0, 2**32 - 1]


def test_cycle_partitions_and_lazy_full_state_labels() -> None:
    ledger = meter()
    calls: list[tuple[data.Words, data.Key, int]] = []
    backing = sample(42, ledger)

    def oracle(rows: data.Words, key: data.Key, rounds: int) -> data.Words:
        calls.append((rows, key, rounds))
        return rows ^ np.uint32(0x78563400 | (rounds << 4) | (key[0] & 15))

    cycle = data.make_panels(ledger, keys(), oracle, sample=lambda n, m: backing)
    backing[:] = 0
    assert not calls and ledger.snapshot()['attempted_real_blocks'] == 0
    parts = []
    for rounds in ROUNDS:
        panels = cycle.for_round(rounds)
        for repeat in range(2):
            for split, size in SIZES.items():
                rows = panels.points(repeat, split)
                assert rows.shape == (size,) and rows.dtype == np.uint32
                with pytest.raises(ValueError):
                    rows.setflags(write=True)
                if split == 'test' or rounds == ROUNDS[0]:
                    parts.append(rows)
                else:
                    assert rows is cycle.for_round(ROUNDS[0]).points(repeat, split)
                labels = panels.label(repeat, split)
                actual_rows, key, actual_round = calls[-1]
                assert actual_rows is rows and actual_round == rounds
                assert key == [(0x3210, 0x7654, 0xBA98, 0xFEDC), (2, 0, 0, 0)][repeat]
                expected = [f'{int(row) ^ (0x78563400 | (rounds << 4) | (key[0] & 15)):032b}'
                            for row in rows]
                assert labels.dtype == np.uint8 and labels.shape == (size, 32)
                np.testing.assert_array_equal(labels, [list(map(int, bits)) for bits in expected])
                with pytest.raises(ValueError):
                    labels.setflags(write=True)
                with pytest.raises(Stopped, match='^DOMAIN_REUSED$'):
                    panels.label(repeat, split)
    assert len(parts) == 12
    np.testing.assert_array_equal(np.sort(np.concatenate(parts)), sample(42, ledger))
    assert len(calls) == 24
    assert ledger.snapshot()['attempted_real_blocks'] == 72
    assert ledger.snapshot()['completed_real_blocks'] == 72
    for invalid in (False, 2, 7, 3.0):
        with pytest.raises(ValueError):
            cycle.for_round(cast(int, invalid))
    for repeat, invalid_split in ((False, 'train'), (-1, 'dev'), (2, 'test'), (0, []), (0, 'TEST')):
        with pytest.raises(ValueError):
            cycle.for_round(3).points(repeat, cast(str, invalid_split))


@pytest.mark.parametrize('fault,reason', [
    ('alias', 'ORACLE_RESPONSE'), ('dtype', 'ORACLE_RESPONSE'),
    ('shape', 'ORACLE_RESPONSE'), ('stride', 'ORACLE_RESPONSE'),
    ('exception', 'ORACLE_FAILED'), ('budget', 'REAL_BLOCKS_LIMIT'),
])
def test_failed_label_consumes_domain(fault: str, reason: str) -> None:
    ledger = meter(0 if fault == 'budget' else 72)
    calls: list[data.Words] = []

    def oracle(rows: data.Words, key: data.Key, rounds: int) -> data.Words:
        calls.append(rows)
        if fault == 'exception':
            raise RuntimeError('private detail')
        if fault == 'alias':
            return panels.points(1, 'train')  # A different slice of the shared slab.
        if fault == 'dtype':
            return rows.astype('>u4') if np.little_endian else rows.astype('<u4')
        if fault == 'shape':
            return rows[:, None].copy()
        return np.zeros(2 * len(rows), dtype=np.uint32)[::2]

    panels = data.make_panels(ledger, keys(), oracle, sample=sample).for_round(3)
    with pytest.raises(Stopped, match=f'^{reason}$'):
        panels.label(0, 'train')
    with pytest.raises(Stopped, match='^DOMAIN_REUSED$'):
        panels.label(0, 'train')
    assert len(calls) == (0 if fault == 'budget' else 1)
    assert ledger.snapshot()['attempted_real_blocks'] == 3 * len(calls)
    assert ledger.snapshot()['completed_real_blocks'] == 0


def test_bad_sample_and_duplicate_keys() -> None:
    def unexpected(*args: object) -> Never:
        pytest.fail('unexpected call')

    def duplicate(rows: int, ledger: Meter) -> data.Words:
        points = sample(rows, ledger)
        points[-1] = points[0]
        return points

    with pytest.raises(ValueError, match='plaintext'):
        data.make_panels(meter(), unexpected, unexpected, sample=duplicate)
    draws = []

    def draw(bits: int) -> int:
        draws.append(bits)
        return 1

    with pytest.raises(Stopped, match='^RANDOM_UNAVAILABLE$'):
        data.make_panels(meter(), draw, unexpected, sample=sample)
    assert draws == [64, 64]


def test_panel_batch_repr_and_digest() -> None:
    points = np.array([0x80000001, 2], dtype=np.uint32)
    labels = np.zeros((2, 32), dtype=np.uint8)
    labels[0, 16] = labels[1, 31] = 1
    panel = data.Panel(points, labels)
    assert len(panel) == 2 and 'Panel' in repr(panel)
    encoded, targets = panel.batch(np.array([1, 0, 1], dtype=np.intp))
    expected = np.zeros((3, 32), dtype=np.float32)
    expected[0, 30] = expected[1, 0] = expected[1, 31] = expected[2, 30] = 1
    assert encoded.dtype == np.float32 and targets.dtype == np.uint8
    np.testing.assert_array_equal(encoded, expected)
    np.testing.assert_array_equal(targets, labels[[1, 0, 1]])
    word_bytes = bytes.fromhex('0100008002000000')
    expected_digest = hashlib.sha256(word_bytes + labels.tobytes()).hexdigest()
    assert data.panel_digest(points, labels) == expected_digest
    assert data.panel_digest(points.astype('>u4'), labels) == expected_digest
    invalid_labels = (labels[:, :1], cast(data.Bytes, labels.astype(np.float32)), labels + 2)
    for invalid in invalid_labels:
        with pytest.raises(ValueError):
            data.Panel(points, invalid)
        with pytest.raises(ValueError):
            data.panel_digest(points, invalid)

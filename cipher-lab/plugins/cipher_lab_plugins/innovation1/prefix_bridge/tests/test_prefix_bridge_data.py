from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from cipher_lab_plugins.innovation1.prefix_bridge import config
from cipher_lab_plugins.innovation1.prefix_bridge.data import Words, make_panels
from cipher_lab_plugins.innovation1.prefix_bridge.kat import Key

from cipher_lab.execution import Meter


def small(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('TRAIN_ROWS', 'DEV_ROWS', 'TEST_ROWS'):
        monkeypatch.setattr(config, name, 2)


def test_panels_one_key_balanced_masks_and_deferred_test(
    monkeypatch: pytest.MonkeyPatch, bridge_meter: Meter,
) -> None:
    small(monkeypatch)
    calls: list[Words] = []
    draws: list[int] = []
    samples: list[int] = []

    def draw(bits: int) -> int:
        draws.append(bits)
        return 0x33441122334455667788

    def sample(rng: np.random.Generator, rows: int) -> Words:
        samples.append(rows)
        return (np.arange(16, dtype=np.uint64) + 100 * len(samples)).reshape(2, 4, 2)

    def oracle(points: Words, key: Key, rounds: tuple[int, ...]) -> Words:
        assert key == (0x55667788, 0x11223344, 0x3344) and rounds == (2, 3, 4, 5, 6, 7)
        calls.append(points.copy())
        # Literal per-round outputs; no actual cipher formula.
        return points[:, None] ^ np.array([1, 2, 4, 8, 16, 32], dtype=np.uint64)

    # Only a fake ledger for the artificial 3x2 group panel.
    monkeypatch.setattr(bridge_meter, '_policy', replace(
        bridge_meter._policy, real_blocks=288))
    panels = make_panels(bridge_meter, draw, oracle, sample=sample)
    assert draws == [80] and samples == [2, 2]
    at = np.arange(2, dtype=np.intp)
    x_s, y = panels.train('s2-only').batch(at)
    x_c, cy = panels.train('r2-cipher').batch(at)
    x_j, jy = panels.view('train', 'r2-joint').batch(at)
    assert x_j.shape == (2, 4, 6, 64) and x_j.dtype == np.float32
    assert np.flatnonzero(x_s[0, 0, 0]).tolist() == [57, 58, 61, 63]  # 100 XOR 1
    assert y.shape == (2, 1) and y.dtype == np.uint8 and sorted(y[:, 0]) == [0, 1]
    assert np.array_equal(y, cy) and np.array_equal(y, jy)
    assert np.array_equal(x_s[:, :, :3], x_c[:, :, 3:])
    assert np.array_equal(x_j[:, :, :3], x_j[:, :, 3:])
    assert not np.any(x_s[:, :, 3:]) and not np.any(x_c[:, :, :3])
    assert np.array_equal(x_j[:, :, 2], x_j[:, :, 0] != x_j[:, :, 1])
    assert np.array_equal(panels.train('r7-joint').batch(at)[0][:, :, :3], x_s[:, :, :3])
    with monkeypatch.context() as no_final_round:
        no_final_round.setattr(config, 'ROUNDS', object())  # not iterable / no index
        assert np.array_equal(panels.train('s2-only').batch(at)[0], x_s)
    for index, supplied in enumerate(calls, 1):
        labels = y if index == 1 else panels.dev('s2-only').batch(at)[1]
        actual = supplied.reshape(2, 4, 2)
        expected = (np.arange(16, dtype=np.uint64) + 100 * index).reshape(2, 4, 2)
        expected[labels[:, 0] == 1, :, 1] = expected[labels[:, 0] == 1, :, 0] ^ 0xD00000
        assert np.array_equal(actual, expected)
    with pytest.raises(ValueError, match='TEST_NOT_READY'):
        panels.view('test', 's2-only')
    panels.test()
    assert len(calls) == 3 and draws == [80]
    assert sorted(panels.view('test', 's2-only').batch(at)[1][:, 0]) == [0, 1]
    assert bridge_meter.snapshot()['completed_real_blocks'] == 288
    with pytest.raises(ValueError, match='TEST_ALREADY_REQUESTED'):
        panels.test()


def test_collision_rejected_before_second_oracle_and_private_error_redacted(
    monkeypatch: pytest.MonkeyPatch, bridge_meter: Meter,
) -> None:
    small(monkeypatch)
    monkeypatch.setattr(bridge_meter, '_policy', replace(
        bridge_meter._policy, real_blocks=288))
    monkeypatch.setattr(config, 'DATA_SEED', 1)
    calls: list[int] = []

    def oracle(points: Words, key: Key, rounds: tuple[int, ...]) -> Words:
        calls.append(len(points))
        return np.zeros((len(points), 6), dtype=np.uint64)

    with pytest.raises(ValueError, match='SPLIT_PAIR_OVERLAP'):
        make_panels(bridge_meter, lambda bits: 0, oracle,
                    sample=lambda rng, rows: np.zeros((rows, 4, 2), dtype=np.uint64))
    assert calls == [16]

    def broken(points: Words, key: Key, rounds: tuple[int, ...]) -> Words:
        raise ValueError('secret-sentinel')

    with pytest.raises(ValueError, match='^PREFIX_ORACLE_FAILED$') as error:
        make_panels(bridge_meter, lambda bits: 0, broken,
                    sample=lambda rng, rows: np.zeros((rows, 4, 2), dtype=np.uint64))
    assert error.value.__cause__ is None and error.value.__suppress_context__
    assert bridge_meter.snapshot()['attempted_real_blocks'] == 192
    assert bridge_meter.snapshot()['completed_real_blocks'] == 96


def test_oracle_response_failure_paid_once_no_retry(
    monkeypatch: pytest.MonkeyPatch, bridge_meter: Meter,
) -> None:
    small(monkeypatch)
    monkeypatch.setattr(bridge_meter, '_policy', replace(bridge_meter._policy, real_blocks=288))
    calls: list[int] = []

    def bad(points: Words, key: Key, rounds: tuple[int, ...]) -> Words:
        calls.append(len(points))
        points[0] ^= np.uint64(1)
        return np.zeros((len(points), 6), dtype=np.uint64)

    with pytest.raises(ValueError, match='PREFIX_ORACLE_RESPONSE'):
        make_panels(bridge_meter, lambda bits: 0, bad,
                    sample=lambda rng, n: np.zeros((n, 4, 2), dtype=np.uint64))
    assert calls == [16]
    assert bridge_meter.snapshot()['attempted_real_blocks'] == 96
    assert bridge_meter.snapshot()['completed_real_blocks'] == 0

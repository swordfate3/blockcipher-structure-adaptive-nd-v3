from __future__ import annotations

import itertools
from datetime import UTC, datetime

import numpy as np
import pytest
from numpy.typing import NDArray

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter


@pytest.mark.parametrize('fault', [None, 'overlap', 'duplicate', 'alias', 'mutation',
                                   'cipher-fixed', 'cipher-paired', 'cipher-dev'])
def test_owned_panels(monkeypatch: pytest.MonkeyPatch, fault: str | None) -> None:
    meter = Meter(
        Policy(2, 120, 120, '2026-01-01T01:00:00Z', real_blocks=16, null_words=0,
               optimizer_updates=0, forward_rows=0, rss_stop_bytes=2**31, output_bytes=2**24),
        Clocks(lambda: 0., lambda: 0., lambda: datetime(2026, 1, 1, tzinfo=UTC)),
        lambda: 1,
    )
    from cipher_lab_plugins.innovation1.present_card_per_pair_key import config, data

    for name in ('TRAIN_ROWS', 'DEV_ROWS', 'TEST_ROWS'):
        monkeypatch.setattr(config, name, 2)
    monkeypatch.setattr(config, 'DIFFERENCE', 1)

    def forbidden(*args: object) -> None:
        pytest.fail('Production per-pair oracle called by artificial fixture')

    monkeypatch.setattr(data, 'encrypt_pairs', forbidden)
    samples = 0
    fixed: list[NDArray[np.uint64]] = []
    paired: list[NDArray[np.uint64]] = []
    keys = itertools.count(50)

    def sample(rng: np.random.Generator, rows: int) -> NDArray[np.uint64]:
        assert rows == 2
        nonlocal samples
        samples += 1
        start = samples * 10
        if fault == 'overlap' and samples == 3:
            return fixed[0][:, ::-1].copy()
        return np.array([[start, start + 1], [start + 2, start + 3]], dtype=np.uint64)

    def bits(n: int) -> int:
        assert n == 80
        return 17

    def entropy(n: int) -> bytes:
        assert n == 20
        values = [next(keys), next(keys)]
        if fault == 'duplicate' and values[0] == 54:
            values[0] = 17
        return b''.join(value.to_bytes(10, 'little') for value in values)

    def encrypt(points: NDArray[np.uint64], key: tuple[int, int, int],
                rounds: tuple[int, ...]) -> NDArray[np.uint64]:
        assert key == (17, 0, 0) and rounds == (7,) and points.shape == (4,)
        fixed.append(points.reshape(-1, 2).copy())
        return (points ^ np.uint64(1 << 63)).reshape(-1, 1)

    def encrypt_pairs(points: NDArray[np.uint64], low: NDArray[np.uint64],
                      high: NDArray[np.uint16]) -> NDArray[np.uint64]:
        assert low.dtype == np.uint64 and high.dtype == np.uint16 and not high.any()
        paired.append(points.copy())
        if len(paired) == 3 and fault == 'alias':
            return points
        if len(paired) == 3 and fault == 'mutation':
            low[0] ^= np.uint64(1)
        if len(paired) == 3 and fault == 'cipher-fixed':
            return (fixed[0] ^ np.uint64(1 << 63))[:, ::-1].copy()
        if len(paired) == 3 and fault in ('cipher-paired', 'cipher-dev'):
            other = paired[0 if fault == 'cipher-paired' else 1]
            return (other ^ np.uint64(1 << 62))[:, ::-1].copy()
        return points ^ np.uint64(1 << 62)

    panels = data.make_panels(meter, bits, entropy, encrypt, encrypt_pairs, sample=sample)
    assert samples == 2 and len(fixed) == 1 and len(paired) == 2
    np.testing.assert_array_equal(fixed[0], paired[0])
    assert panels.dev(config.FIXED) is panels.dev(config.PER_PAIR)
    assert set(panels.hashes) == {'train-fixed-key', 'train-per-pair-key', 'dev'}
    x, y = panels.train(config.FIXED).batch(np.arange(2, dtype=np.intp))
    assert x.shape == (2, 1, 2, 64) and x[..., 0].all() and y.sum() == 1
    np.testing.assert_array_equal(y, panels.train(config.PER_PAIR).batch(np.arange(2))[1])
    with pytest.raises(ValueError, match='TEST_NOT_READY'):
        panels.view('test', config.FIXED)
    if fault is None:
        panels.test()
        assert panels.view('test', config.FIXED) is panels.view('test', config.PER_PAIR)
        assert 'test' in panels.hashes
    else:
        errors = {'overlap': 'SPLIT_PAIR_OVERLAP', 'duplicate': 'DUPLICATE_PRIVATE_KEY',
                  'alias': 'PREFIX_ORACLE_RESPONSE', 'mutation': 'PREFIX_ORACLE_RESPONSE',
                  'cipher-fixed': 'SPLIT_PAIR_OVERLAP', 'cipher-paired': 'SPLIT_PAIR_OVERLAP',
                  'cipher-dev': 'SPLIT_PAIR_OVERLAP'}
        with pytest.raises(ValueError, match=errors[fault]):
            panels.test()
        assert 'test' not in panels.hashes
    with pytest.raises(ValueError, match='TEST_ALREADY_REQUESTED'):
        panels.test()
    counts = meter.snapshot()
    clean_oracle = fault is None or fault.startswith('cipher-')
    assert counts['completed_real_blocks'] == (16 if clean_oracle else 12)
    assert counts['attempted_real_blocks'] == (12 if fault in ('overlap', 'duplicate') else 16)

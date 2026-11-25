from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

from cipher_lab.execution import Meter, Stopped


def test_split8_public_draw_and_shared_private_oracle(monkeypatch: pytest.MonkeyPatch,
                                                     split8_meter: Meter) -> None:
    from cipher_lab_plugins.direct_block.split8 import config, data

    sizes = config.Sizes(6, 3, 4, 2)
    monkeypatch.setattr(config, 'SIZES', sizes)
    a = data.sample_points(sizes.total, split8_meter)
    b = data.sample_points(sizes.total, split8_meter)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.uint32 and len(np.unique(a)) == sizes.total and np.any(a > 65535)
    draws: list[int] = []
    calls: list[tuple[int, list[int]]] = []
    monkeypatch.setattr(data, 'MAX_BATCH', 4)

    def keydraw(bits: int) -> int:
        draws.append(bits)
        return 0x1918111009080100

    def oracle(points: data.Words, key: data.Key, rounds: int) -> data.Words:
        assert key == (0x0100, 0x0908, 0x1110, 0x1918)
        assert not points.flags.writeable and 0 < len(points) <= 4
        calls.append((rounds, points.tolist()))
        return ((points & np.uint32(0xff)) << np.uint32(24)) | np.uint32(rounds)

    points = np.arange(sizes.total, dtype=np.intp).astype(np.uint32) | np.uint32(0x80000000)
    with pytest.raises(ValueError, match='plaintext'):
        data.make_panels(split8_meter, keydraw, oracle,
                         sample=lambda n, m: np.zeros(n, dtype=np.uint32))
    assert not draws and not calls
    panels = data.make_panels(split8_meter, keydraw, oracle, sample=lambda n, m: points)
    assert draws == [64] and calls == []
    labels: dict[config.Split, dict[int, data.Bytes]] = {s: {} for s in config.SPLITS}
    for r in config.ROUNDS:
        for s in config.SPLITS:
            labels[s][r] = panels.label(s, r)
    assert draws == [64]
    assert len(calls) == 12  # six TRAIN split into 4+2; one chunk each for DEV/TEST
    for r in config.ROUNDS:
        assert [p for rr, chunk in calls if rr == r for p in chunk] == points.tolist()
    with pytest.raises(Stopped, match='DOMAIN_REUSED'):
        panels.label('test', 1)
    assert split8_meter.snapshot()['completed_real_blocks'] == 3 * sizes.total
    train = panels.points('train')
    for j, cols in enumerate(config.COLS):
        x, y = data.Panel(train, labels['train'][1], j).batch(np.arange(6, dtype=np.intp))
        assert x.shape == (6, 32) and x.dtype == np.float32 and np.all(x[:, 0] == 1)
        assert y.shape == (6, 2) and y.dtype == np.uint8
        np.testing.assert_array_equal(y, data.high_bits(labels['train'][1])[:, cols])
    with np.load(BytesIO(data.dataset(train, labels['train'])), allow_pickle=False) as doc:
        assert set(doc.files) == {'plaintext', 'r1_high8', 'r2_high8', 'r3_high8'}
        np.testing.assert_array_equal(doc['plaintext'], points[:6])
        np.testing.assert_array_equal(doc['r3_high8'], np.arange(6, dtype=np.uint8))


def test_split8_oracle_attempt_not_refunded(monkeypatch: pytest.MonkeyPatch,
                                          split8_meter: Meter) -> None:
    from cipher_lab_plugins.direct_block.split8 import config, data

    monkeypatch.setattr(config, 'SIZES', config.Sizes(4, 1, 1, 2))
    monkeypatch.setattr(data, 'MAX_BATCH', 2)
    calls = 0

    def invalid(points: data.Words, key: data.Key, rounds: int) -> data.Words:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('secret backend detail')
        return points.copy()

    panels = data.make_panels(split8_meter, lambda bits: 5, invalid,
                             sample=lambda n, m: np.arange(n, dtype=np.uint32))
    with pytest.raises(Stopped, match='^ORACLE_FAILED$'):
        panels.label('train', 2)
    with pytest.raises(Stopped, match='DOMAIN_REUSED'):
        panels.label('train', 2)
    counts = split8_meter.snapshot()
    assert calls == 2 and counts['attempted_real_blocks'] == 4
    assert counts['completed_real_blocks'] == 2

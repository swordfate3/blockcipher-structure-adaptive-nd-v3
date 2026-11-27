from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from cipher_lab.execution import Meter


@pytest.mark.parametrize('overlap', [False, True])
def test_compact_endpoints_and_single_test_request(
    monkeypatch: pytest.MonkeyPatch, bridge_meter: Meter, overlap: bool,
) -> None:
    from cipher_lab_plugins.innovation1.present_card_fresh import config, data

    for name in ('TRAIN_ROWS', 'DEV_ROWS', 'TEST_ROWS'):
        monkeypatch.setattr(config, name, 2)
    draws: list[int] = []
    sent: list[NDArray[np.uint64]] = []
    samples: list[NDArray[np.uint64]] = []

    def draw(bits: int) -> int:
        draws.append(bits)
        return 17

    def sample(rng: np.random.Generator, rows: int) -> NDArray[np.uint64]:
        assert rows == 2
        start = len(samples) * 10
        value = np.array([[start, start + 1], [start + 2, start + 3]], dtype=np.uint64)
        if overlap and start == 20:
            # Both possible positive rows collide across splits (their order may differ).
            value[:, 0] = sent[0][sent[0][:, 0] ^ sent[0][:, 1] == config.DIFFERENCE, 0][0]
        samples.append(value.copy())
        return value

    def encrypt(points: NDArray[np.uint64], key: tuple[int, int, int],
                rounds: tuple[int, ...]) -> NDArray[np.uint64]:
        assert key == (17, 0, 0) and rounds == (7,) and points.shape == (4,)
        sent.append(points.reshape(-1, 2).copy())
        return (points ^ np.uint64(1 << 63)).reshape(-1, 1)

    panels = data.make_panels(bridge_meter, draw, encrypt, sample=sample)
    assert draws == [80] and len(sent) == 2 and set(panels.hashes) == {'train', 'dev'}
    train = panels.view('train', config.SLOT)
    assert train._words.shape == (2, 2) and train._words.dtype == np.uint64
    assert not train._words.flags.writeable
    x, y = train.batch(np.arange(2, dtype=np.intp))
    assert x.shape == (2, 1, 2, 64) and x.dtype == np.float32 and np.all(x[..., 0] == 1)
    pos, neg = int(y[:, 0].argmax()), int(y[:, 0].argmin())
    assert sent[0][pos, 1] == sent[0][pos, 0] ^ np.uint64(config.DIFFERENCE)
    assert np.array_equal(sent[0][neg], samples[0][neg])
    with pytest.raises(ValueError, match='TEST_NOT_READY'):
        panels.view('test', config.SLOT)
    if overlap:
        with pytest.raises(ValueError, match='SPLIT_PAIR_OVERLAP'):
            panels.test()
        assert len(sent) == 2 and 'test' not in panels.hashes
    else:
        panels.test()
        assert len(sent) == 3 and set(panels.hashes) == {'train', 'dev', 'test'}
    with pytest.raises(ValueError, match='TEST_ALREADY_REQUESTED'):
        panels.test()
    assert bridge_meter.snapshot()['completed_real_blocks'] == (8 if overlap else 12)

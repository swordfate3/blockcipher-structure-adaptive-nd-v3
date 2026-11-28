import hashlib
from typing import cast

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.bit_structure import config, data, metrics
from numpy.typing import NDArray

from cipher_lab.execution import Meter, Stopped


class Ledger:
    attempted = completed = 0

    def check(self) -> None:
        pass

    def reserve(self, kind: str, rows: int) -> None:
        assert kind == 'real_blocks'
        self.attempted += rows

    def complete(self, kind: str, rows: int) -> None:
        assert kind == 'real_blocks'
        self.completed += rows


@pytest.fixture
def tiny(monkeypatch: pytest.MonkeyPatch) -> Ledger:
    for split in ('TRAIN', 'DEV', 'TEST'):
        monkeypatch.setattr(config, split, 2)
    return Ledger()


def sample(rows: int, meter: Meter) -> NDArray[np.uint32]:
    assert rows == 12
    return np.arange(rows, dtype=np.uint32)


def test_six_shared_panels(tiny: Ledger) -> None:
    draws: list[int] = []
    calls: list[data.Key] = []

    def draw(bits: int) -> int:
        draws.append(bits)
        return len(draws)

    def encrypt(points: data.Words, key: data.Key, rounds: int) -> data.Words:
        calls.append(key)
        assert rounds == 3 and tiny.attempted == tiny.completed + len(points)
        return ((points ^ np.uint32(key[0])) & np.uint32(1)) << np.uint32(31)

    panels = data.make_panels(cast(Meter, tiny), draw, encrypt, sample=sample)
    assert draws == [64, 64] and not calls
    seen: set[int] = set()
    for repeat in range(2):
        for split in ('train', 'dev', 'test'):
            points = panels.points(repeat, split)
            assert not points.flags.writeable and seen.isdisjoint(points)
            seen.update(map(int, points))
            labels = panels.label(repeat, split)
            assert labels.shape == (2, 1) and labels.dtype == np.uint8
            assert not labels.flags.writeable
            assert labels[:, 0].tolist() == [int((p ^ (repeat + 1)) & 1) for p in points]
            with pytest.raises(Stopped, match='DOMAIN_REUSED'):
                panels.label(repeat, split)
    assert seen == set(range(12))
    assert calls == [(1, 0, 0, 0)] * 3 + [(2, 0, 0, 0)] * 3
    assert tiny.attempted == tiny.completed == 12


def test_duplicate_key_stops_after_two_draws(tiny: Ledger) -> None:
    draws: list[int] = []

    def draw(bits: int) -> int:
        draws.append(bits)
        return 71

    def encrypt(points: data.Words, key: data.Key, rounds: int) -> data.Words:
        pytest.fail('duplicate keys must stop before encryption')

    with pytest.raises(Stopped):
        data.make_panels(cast(Meter, tiny), draw, encrypt, sample=sample)
    assert draws == [64, 64] and tiny.attempted == 0


def test_failed_label_spends_attempt(tiny: Ledger) -> None:
    keys = iter((1, 2))

    def encrypt(points: data.Words, key: data.Key, rounds: int) -> data.Words:
        raise RuntimeError('fake failure')

    panels = data.make_panels(cast(Meter, tiny), lambda _: next(keys), encrypt, sample=sample)
    with pytest.raises(Stopped, match='ORACLE_FAILED'):
        panels.label(1, 'test')
    with pytest.raises(Stopped, match='DOMAIN_REUSED'):
        panels.label(1, 'test')
    assert tiny.attempted == 2 and tiny.completed == 0


def test_batch_and_portable_digest() -> None:
    points = np.array([0x80000000, 1], dtype=np.uint32)
    labels = np.array([[1], [0]], dtype=np.uint8)
    x, y = data.Panel(points, labels).batch(np.array([1, 0], dtype=np.intp))
    assert x.shape == (2, 32) and x.dtype == np.float32
    assert np.nonzero(x)[1].tolist() == [31, 0]
    assert y.tolist() == [[0], [1]] and y.dtype == np.uint8
    expected = hashlib.sha256(bytes.fromhex('00000080010000000100')).hexdigest()
    assert data.panel_digest(points.astype('>u4'), labels) == expected
    assert data.panel_digest(points[::-1], labels[::-1]) != expected


def test_metrics() -> None:
    labels = np.array([[0], [0], [1], [0]], dtype=np.uint8)
    logits = np.array([[0], [1000], [1000], [-1000]], dtype=np.float32)
    result, good = metrics.score(logits, labels)
    assert result == {'correct': 3, 'rows': 4, 'accuracy': .75,
                      'bce': pytest.approx((1000 + np.log(2)) / 4)}
    assert good.dtype == np.bool_ and good.ravel().tolist() == [True, False, True, True]
    assert metrics.train_constant(labels) == .3
    base = metrics.constant_score(.5, labels)
    assert base['correct'] == 3 and base['bce'] == pytest.approx(np.log(2))
    assert metrics.paired(good, np.array([[True], [True], [False], [False]])) == {
        'both_correct': 1, 'left_only': 2, 'right_only': 1, 'both_wrong': 0,
    }

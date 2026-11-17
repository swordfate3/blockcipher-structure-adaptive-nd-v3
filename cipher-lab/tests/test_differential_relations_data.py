from pathlib import Path

import numpy as np
import pytest
from cipher_lab_plugins.differential_relations.config import MASKS
from cipher_lab_plugins.differential_relations.data import (
    DataSession,
    FrozenFits,
    OrbitSplits,
    SplitSizes,
    array_sha256,
    canonical_orbit,
    encode_pairs,
    load_dataset,
    make_null_targets,
    relation_labels,
    sample_orbits,
    save_dataset,
)


def splits() -> OrbitSplits:
    return sample_orbits(SplitSizes(4, 12, 8, 8), seed=9)


class Fake:
    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []

    def reserve(self, kind: str, amount: int) -> None:
        assert kind == 'real_blocks'
        self.events.append(('reserve', amount))

    def complete(self, kind: str, amount: int) -> None:
        assert kind == 'real_blocks'
        self.events.append(('complete', amount))

    def oracle(self, block: int, key: object) -> int:
        assert key == 'PRIVATE-NOT-EXPORTED'
        self.events.append(('oracle', block))
        return ((block * 0x9e3779b97f4a7c15) ^ (block >> 5)) & ((1 << 64) - 1)


def session() -> tuple[DataSession, Fake]:
    fake = Fake()
    return DataSession('exp-a', splits(), key='PRIVATE-NOT-EXPORTED',
                       oracle=fake.oracle, meter=fake), fake


def frozen() -> FrozenFits:
    return FrozenFits('exp-a', {f'F{i}': str(i) * 64 for i in range(1, 6)})


def test_orbits_and_msb0() -> None:
    s = splits()
    assert np.array_equal(s.small, s.train[:4])
    assert len(set(map(int, np.concatenate((s.train, s.dev, s.test))))) == 28
    assert canonical_orbit(67) == canonical_orbit(3) == 3
    raw = np.array([0, 1 << 63, 1, 2], dtype=np.uint64)
    bits = encode_pairs(raw)
    assert bits.shape == (1, 4, 2, 64) and bits.dtype == np.float32
    assert bits[0, 1, 0, 0] == 1 and bits[0, 2, 0, -1] == 1
    assert np.all(bits[0, :, 0, 57] == 0) and np.all(bits[0, :, 1, 57] == 1)
    with pytest.raises(ValueError):
        OrbitSplits(s.train, s.train[:8], s.test, 4)
    with pytest.raises(ValueError):
        encode_pairs(np.array([64, 1, 2, 3], dtype=np.uint64))


def test_parity_independent_bit_route() -> None:
    words = np.array([0, 1, 1 << 63, MASKS[0], MASKS[2] ^ 1], dtype=np.uint64)
    actual = relation_labels(words)
    expected = [[sum((int(x) >> b) & 1 for b in range(64) if (mask >> b) & 1) % 2
                 for mask in MASKS] for x in words]
    np.testing.assert_array_equal(actual, expected)


def test_meter_private_and_delayed_test() -> None:
    sess, fake = session()
    p = sess.prepare()
    assert len(fake.events) == (12 + 8) * 4
    assert [e[0] for e in fake.events[:4]] == ['reserve', 'oracle', 'oracle', 'complete']
    assert fake.events[0][1] == fake.events[3][1] == 2
    assert 'PRIVATE' not in repr(sess) + repr(p)
    assert sess.prepare() is p
    with pytest.raises(ValueError):
        FrozenFits('exp-a', {'F1': '1' * 64})
    with pytest.raises(ValueError):
        sess.test_labels(FrozenFits('other', frozen().selected_sha256))
    labels = sess.test_labels(frozen())
    assert labels.shape == (8, 4)
    assert sess.test_labels(frozen()) is labels
    sess.release_key()


def test_failed_oracle_and_all_constant() -> None:
    fake = Fake()
    def bad(block: int, key: object) -> int:
        raise RuntimeError('PRIVATE-NOT-EXPORTED')
    sess = DataSession('x', splits(), key=object(), oracle=bad, meter=fake)
    with pytest.raises(ValueError, match='ORACLE_FAILED') as error:
        sess.prepare()
    assert 'PRIVATE' not in str(error.value)
    assert fake.events == [('reserve', 2)]
    const = DataSession('exp-a', splits(), key=None, oracle=lambda b, k: b, meter=Fake())
    assert const.prepare().all_train_constant
    with pytest.raises(ValueError, match='CONSTANT'):
        const.test_labels(frozen())


def test_null_and_npz_actual_content(tmp_path: Path) -> None:
    sess, _ = session()
    prepared = sess.prepare()
    null = make_null_targets(prepared.train_labels, prepared.dev_labels)
    assert sorted(map(tuple, null.train)) == sorted(map(tuple, prepared.train_labels))
    assert null.train_record['seed'] == 41 and null.dev_record['seed'] == 43
    assert null.train_record['permutation_sha256'] != null.dev_record['permutation_sha256']
    path = tmp_path / 'data.npz'
    meta = save_dataset(path, prepared, null=null, test_labels=sess.test_labels(frozen()),
                        frozen=frozen())
    assert 'PRIVATE' not in str(meta) and 'key' not in str(meta).lower()
    p, n, test, loaded = load_dataset(path, experiment_id='exp-a')
    assert p.train_hash == prepared.train_hash and loaded == meta
    assert n is not None and n.train_hash == array_sha256(null.train)
    assert test is not None and test.shape == (8, 4)
    assert meta['selected_sha256'] == dict(frozen().selected_sha256)
    with pytest.raises(ValueError):
        load_dataset(path, experiment_id='other')
    with np.load(path, allow_pickle=False) as z:
        fields = {name: z[name] for name in z.files}
    fields['train_labels'] = fields['train_labels'].copy()
    fields['train_labels'][0, 0] ^= 1
    np.savez_compressed(path, **fields)
    with pytest.raises(ValueError, match='HASH'):
        load_dataset(path, experiment_id='exp-a')


def test_saved_data_load_does_not_draw_rng(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sess, _ = session()
    p = sess.prepare()
    path = tmp_path / 'saved.npz'
    save_dataset(path, p, null=make_null_targets(p.train_labels, p.dev_labels))
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('saved-only loading must not sample or recreate permutations')
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    load_dataset(path, experiment_id='exp-a')


def test_session_release_and_no_retry() -> None:
    sess, fake = session()
    sess.release_key()
    with pytest.raises(ValueError, match='PRIVATE_STATE_UNAVAILABLE'):
        sess.prepare()
    assert fake.events == []


def test_reverse_and_duplicate_orbits_rejected() -> None:
    s = splits()
    train = s.train.copy()
    train[0] ^= np.uint64(64)
    with pytest.raises(ValueError):
        OrbitSplits(train, s.dev, s.test, 4)
    train = s.train.copy()
    train[1] = train[0]
    with pytest.raises(ValueError):
        OrbitSplits(train, s.dev, s.test, 4)


def test_different_private_instance_same_public_sampling_has_distinct_content() -> None:
    a, _ = session()
    pa = a.prepare()
    # Same public orbits, different fake private function, no private digest is exposed.
    b = DataSession('exp-b', splits(), key=None, oracle=lambda b, k: b, meter=Fake())
    pb = b.prepare()
    np.testing.assert_array_equal(pa.splits.train, pb.splits.train)
    assert pa.train_hash != pb.train_hash
    assert pa.dev_hash != pb.dev_hash


def test_null_rows_keep_joint_frequency_and_coincidences() -> None:
    rows = np.array([[0, 1, 0, 1], [1, 0, 1, 0]] * 16, dtype=np.uint8)
    null = make_null_targets(rows, rows)
    for target, record in ((null.train, null.train_record), (null.dev, null.dev_record)):
        assert sorted(map(tuple, target)) == sorted(map(tuple, rows))
        assert np.all(target[:, 0] == target[:, 2])
        assert np.all(target[:, 0] != target[:, 1])
        assert record['coincident_rows'] == int(np.sum(np.all(target == rows, axis=1)))
        assert 0 <= int(str(record['fixed_rows'])) <= int(str(record['coincident_rows']))
    assert not np.array_equal(null.train, null.dev)


def test_changed_frozen_selection_and_test_without_token(tmp_path: Path) -> None:
    sess, _ = session()
    p = sess.prepare()
    test = sess.test_labels(frozen())
    changed = dict(frozen().selected_sha256) | {'F3': 'a' * 64}
    with pytest.raises(ValueError):
        sess.test_labels(FrozenFits('exp-a', changed))
    with pytest.raises(ValueError):
        save_dataset(tmp_path / 'bad.npz', p, test_labels=test)


def test_partial_oracle_failure_never_completed_or_retried() -> None:
    fake = Fake()
    count = 0
    def oracle(block: int, key: object) -> int:
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError('hidden')
        return block
    sess = DataSession('exp-a', splits(), key=None, oracle=oracle, meter=fake)
    with pytest.raises(ValueError, match='ORACLE_FAILED'):
        sess.prepare()
    assert count == 2 and fake.events == [('reserve', 2)]
    with pytest.raises(ValueError, match='PRIVATE_STATE_UNAVAILABLE'):
        sess.prepare()
    assert count == 2


def test_tuple_batch_oracle_reserves_each_pair_before_call() -> None:
    fake = Fake()
    calls: list[int] = []
    def batch(blocks: tuple[int, ...], key: object) -> tuple[int, ...]:
        assert key == 'PRIVATE-NOT-EXPORTED'
        assert len(blocks) % 2 == 0
        assert fake.events[-1] == ('reserve', 2)
        assert all(blocks[i] ^ blocks[i + 1] == 64 for i in range(0, len(blocks), 2))
        calls.append(len(blocks))
        return tuple(((b * 0x9e3779b97f4a7c15) ^ (b >> 5)) & ((1 << 64) - 1)
                     for b in blocks)
    sess = DataSession('exp-a', splits(), key='PRIVATE-NOT-EXPORTED',
                       batch_oracle=batch, meter=fake)
    p = sess.prepare()
    reference, _ = session()
    np.testing.assert_array_equal(p.train_labels, reference.prepare().train_labels)
    assert calls == [24, 16]
    assert fake.events[:12] == [('reserve', 2)] * 12
    assert fake.events[12:24] == [('complete', 2)] * 12


def test_batch_bad_response_never_completes() -> None:
    fake = Fake()
    sess = DataSession('exp-a', splits(), key=None, batch_oracle=lambda b, k: (0,), meter=fake)
    with pytest.raises(ValueError, match='ORACLE_FAILED'):
        sess.prepare()
    assert fake.events == [('reserve', 2)] * 12
    with pytest.raises(ValueError, match='PRIVATE_STATE_UNAVAILABLE'):
        sess.prepare()


def test_reservation_failure_prevents_batch_call() -> None:
    calls: list[int] = []
    class Deny(Fake):
        def reserve(self, kind: str, amount: int) -> None:
            super().reserve(kind, amount)
            raise ValueError('BUDGET_LIMIT')
    def batch(blocks: tuple[int, ...], key: object) -> tuple[int, ...]:
        calls.append(1)
        return blocks
    sess = DataSession('exp-a', splits(), key=None, batch_oracle=batch, meter=Deny())
    with pytest.raises(ValueError, match='BUDGET_LIMIT'):
        sess.prepare()
    assert not calls

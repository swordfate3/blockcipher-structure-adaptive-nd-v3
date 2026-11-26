from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Any

import numpy as np
import pytest

from cipher_lab.execution import Meter


def test_frozen_recipe() -> None:
    from cipher_lab_plugins.innovation1.mask_aux_single import config

    assert config.counts(30, 30) == dict(real_blocks=10_327_684, null_words=0,
                                         optimizer_updates=585_960, forward_rows=452_228_224)
    assert config.steps_per_epoch() == 9766
    assert config.TRAIN_ROWS % config.BATCH_SIZE == 320
    assert config.recipe()['train_ciphertexts'] == 10_000_000
    assert config.recipe()['early_stop'] is None
    for a, b in ((29, 30), (30, 31), (30., 30), (True, 30)):
        with pytest.raises(ValueError):
            config.counts(a, b)  # type: ignore[arg-type]


@pytest.mark.parametrize('fault', ['', 'reverse', 'oracle-collision', 'alias', 'mutate',
                                  'endian', 'shape', 'raise', 'bad-sample', 'bad-draw'])
def test_compact_panels(
    fault: str, monkeypatch: pytest.MonkeyPatch, bridge_meter: Meter,
) -> None:
    from cipher_lab_plugins.innovation1.mask_aux_single import config, data

    for name, amount in [('TRAIN_ROWS', 6), ('DEV_ROWS', 4), ('TEST_ROWS', 4)]:
        monkeypatch.setattr(config, name, amount)
    monkeypatch.setattr(data, '_CHUNK_PAIRS', 2)
    meter = bridge_meter
    monkeypatch.setattr(meter, '_policy', replace(meter._policy, real_blocks=28))
    samples: list[Any] = []
    oracle: list[Any] = []
    draws: list[int] = []
    start, delta = (1 << 63) + 64, config.DIFFERENCE

    def draw(bits: int) -> int:
        draws.append(bits)
        return (1 << 80) if fault == 'bad-draw' else (9 << 64) + (7 << 32) + 3

    def sample(rng: Any, n: int) -> Any:
        i = len(samples)
        u = start + (i << 32)
        if fault == 'reverse' and i == 6:
            u = start ^ delta
        a = np.tile(np.array([u, u ^ delta], dtype=np.uint64), (n, 1))
        samples.append(a.copy())
        return a.astype('>u8') if fault == 'bad-sample' else a

    def encrypt(p: Any, key: Any, rounds: Any) -> Any:
        assert p.dtype == np.uint64 and p.flags.c_contiguous and p.shape == (4,)
        assert key == (3, 7, 9) and rounds == (7,)
        assert meter.snapshot()['attempted_real_blocks'] == 4*(len(oracle)+1)
        oracle.append(p.copy())
        if len(oracle) == 2:
            if fault == 'raise':
                raise RuntimeError('no secret diagnostic')
            if fault == 'alias':
                return p.reshape(-1, 1)
            if fault == 'mutate':
                p[0] ^= np.uint64(1)
            if fault == 'endian':
                return p.astype('>u8').reshape(-1, 1)
            if fault == 'shape':
                return p.copy()
        source = oracle[0] if fault == 'oracle-collision' and len(oracle) == 7 else p
        return (source ^ np.uint64(5)).reshape(-1, 1)

    def make() -> data.Panels:
        return data.make_panels(meter, draw, encrypt, sample=sample)
    if fault not in ('', 'reverse', 'oracle-collision'):
        message = {'bad-draw': 'PRIVATE_SOURCE_FAILED', 'bad-sample': 'PANEL_SAMPLING_FAILED',
                   'raise': 'PREFIX_ORACLE_FAILED'}.get(fault, 'PREFIX_ORACLE_RESPONSE')
        with pytest.raises(ValueError, match=message):
            make()
        counts = meter.snapshot()
        assert (counts['attempted_real_blocks'], counts['completed_real_blocks']) == (
            (0, 0) if fault.startswith('bad') else (8, 4))
        return
    panels = make()
    assert draws == [80] and len(oracle) == 5 and set(panels.hashes) == {'train', 'dev'}
    with pytest.raises(ValueError, match='TEST_NOT_READY'):
        panels.view('test', config.PANEL)
    train = panels.train(config.PANEL)
    assert train._words.shape == (6, 2) and train._words.nbytes == 96
    assert train._labels.nbytes == 6
    assert not train._words.flags.writeable and not train._labels.flags.writeable
    digest = sha256(f'{config.METHOD}/train/6/N,2/<u8/MSB0/N,1/u8/'.encode('ascii'))
    digest.update(train._labels.tobytes())
    digest.update((np.concatenate(oracle[:3]) ^ np.uint64(5)).astype('<u8').tobytes())
    assert panels.hashes['train'] == digest.hexdigest()
    x, y = train.batch(np.arange(6, dtype=np.intp))
    assert x.shape == (6, 1, 6, 64) and y.shape == (6, 1) and y.sum() == 3
    assert not x[:, :, :3].any() and x[0, 0, 3, 0] == 1 and x[0, 0, 3, -1] == 1
    np.testing.assert_array_equal(x[:, :, 5], x[:, :, 3] != x[:, :, 4])
    if fault:
        with pytest.raises(ValueError, match='SPLIT_PAIR_OVERLAP'):
            panels.test()
        counts = meter.snapshot()
        assert counts['completed_real_blocks'] == 24
        assert counts['attempted_real_blocks'] == (24 if fault == 'reverse' else 28)
        assert set(panels.hashes) == {'train', 'dev'}
    else:
        panels.test()
        assert set(panels.hashes) == {'train', 'dev', 'test'}
        assert meter.snapshot()['completed_real_blocks'] == 28
    with pytest.raises(ValueError, match='TEST_ALREADY_REQUESTED'):
        panels.test()


def test_lookup_unsigned_lexicographic_and_both_endpoints() -> None:
    from cipher_lab_plugins.innovation1.mask_aux_single import data

    previous = np.array([[2**64-1, 0], [31, 2**63], [2**63+1, 2**64-2], [17, 31]],
                        np.uint64)
    keys = np.empty(len(previous), data._PAIR_KEY)
    data._keys(previous, keys)
    keys.sort()
    assert keys.tolist() == [(0, 2**64-1), (17, 31), (31, 2**63), (2**63+1, 2**64-2)]
    later = np.array([[2**64-1, 2**64-2], [31, 2**63+1], [0, 2**64-2], [31, 18]],
                     np.uint64)
    probe = np.empty(len(later), data._PAIR_KEY)
    data._keys(later, probe)
    data._require_disjoint(probe, [keys])
    for pair in previous:
        data._keys(pair[::-1].reshape(1, 2), probe[:1])
        with pytest.raises(ValueError, match='SPLIT_PAIR_OVERLAP'):
            data._require_disjoint(probe, [keys])

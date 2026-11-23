from __future__ import annotations

import os
import random
import secrets
from collections.abc import Callable
from hashlib import sha256
from io import BytesIO

import numpy as np
import pytest
from numpy.typing import NDArray

from cipher_lab.ciphers import simon
from cipher_lab.execution import Meter, Stopped

Words = NDArray[np.uint64]


@pytest.mark.parametrize('api', [
    'os.urandom', 'random._urandom', 'random.SystemRandom', 'secrets.SystemRandom',
    'secrets.randbelow', 'secrets.randbits', 'secrets.choice', 'secrets.token_bytes',
    'secrets.token_hex', 'secrets.token_urlsafe', 'secrets._sysrand.getrandbits',
])
def test_cube_entropy_guard_blocks_before_lower_level(
    cube_entropy_probe: list[int], api: str,
) -> None:
    calls: dict[str, Callable[[], object]] = {
        'os.urandom': lambda: os.urandom(1),
        'random._urandom': lambda: getattr(random, '_urandom')(1),
        'random.SystemRandom': lambda: random.SystemRandom().getrandbits(1),
        'secrets.SystemRandom': lambda: secrets.SystemRandom().getrandbits(1),
        'secrets.randbelow': lambda: secrets.randbelow(2),
        'secrets.randbits': lambda: secrets.randbits(1),
        'secrets.choice': lambda: secrets.choice((1, 2)),
        'secrets.token_bytes': lambda: secrets.token_bytes(1),
        'secrets.token_hex': lambda: secrets.token_hex(1),
        'secrets.token_urlsafe': lambda: secrets.token_urlsafe(1),
        'secrets._sysrand.getrandbits': lambda: getattr(secrets, '_sysrand').getrandbits(1),
    }
    with pytest.raises(AssertionError, match='Task1 scientific or external operation forbidden'):
        calls[api]()
    assert cube_entropy_probe == []


def test_cube_recipe_and_provider_agree(
    cube_meter: Meter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import config, data

    draws: list[int] = []
    calls: list[tuple[tuple[int, ...], int]] = []

    def mock_key(bits: int) -> int:
        draws.append(bits)
        return 1 + (2 << 32) + (3 << 64) + (4 << 96)

    def label(points: Words, words: tuple[int, ...], rounds: int) -> Words:
        calls.append((words, rounds))
        assert points.ndim == 1 and points.dtype == np.uint64 and points.flags.c_contiguous
        return ~points

    monkeypatch.setattr(secrets, 'randbits', mock_key)
    monkeypatch.setattr(simon, 'encrypt_batch', label)
    panels = data.production_panels(cube_meter)
    assert calls == [] and draws == [128]
    # Shrink only the synthetic fixture geometry, never the published recipe.
    monkeypatch.setattr(panels, '_families', (1, 1, 1))
    panels.build('train')
    panels.build('dev')
    assert len(calls) == 2
    panels.label_test()
    settings = config.recipe()
    assert settings['rounds'] == 7 and settings['key_protocol'] == 'single-key'
    assert settings['input_width'] == settings['target_width'] == 128
    assert settings['families'] == {'train': 131072, 'dev': 8192, 'test': 8192}
    assert settings['pairs'] == {'train': 1048576, 'dev': 65536, 'test': 65536}
    assert settings['test_denominator_bits'] == 524288
    assert settings['active_positions_lsb0'] == [0, 21, 42, 60]
    assert settings['xor_mask'] == '0x1000040000200001'
    assert settings['active_columns_msb0'] == [63, 42, 21, 3, 127, 106, 85, 67]
    assert calls == [((1, 2, 3, 4), 7)] * 3 and draws == [128]
    assert config.planned_counts() == dict(real_blocks=2359296, null_words=0,
                                           optimizer_updates=40960, forward_rows=11272192)
    assert settings['total_caps'] == dict(real_blocks=2359314, null_words=0,
                                          optimizer_updates=40962, forward_rows=11272224)


def test_cube_pair_geometry_alignment_and_delayed_labels(cube_meter: Meter) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import config, data

    bases = [2, 4, 8, 16]
    mask = 0x1000040000200001
    draws = iter([2 | mask, 2, 4, 4 | mask, 8, 8, 16])
    calls: list[list[int]] = []

    def oracle(points: Words) -> Words:
        calls.append(points.tolist())
        # Stateless, transparently fake; reverse all bits by complement, not SIMON.
        return ~points

    panels = data.CubePanels(oracle, cube_meter, families=(2, 1, 1), draw=draws.__next__)
    with pytest.raises(ValueError, match='split order'):
        panels.label_test()
    with pytest.raises(ValueError, match='split order'):
        panels.build('dev')
    with pytest.raises(ValueError, match='split order'):
        panels.build('test')  # type: ignore[arg-type]  # deliberately bypass the label gate
    assert calls == []
    train, dev = panels.build('train'), panels.build('dev')
    assert list(map(len, calls)) == [32, 16]
    test = panels.label_test()
    with pytest.raises(ValueError, match='split order'):
        panels.label_test()
    assert list(map(len, calls)) == [32, 16, 16]
    for i, panel in enumerate((train, dev, test)):
        assert panel.plaintext.dtype == panel.ciphertext.dtype == panel.family_ids.dtype == 'u8'
        assert panel.plaintext.shape == panel.ciphertext.shape == (len(panel), 2)
        assert panel.family_ids.shape == (len(panel),)
        assert panel.plaintext.ravel().tolist() == calls[i]
        assert np.array_equal(panel.ciphertext, ~panel.plaintext)
        for base in np.unique(panel.family_ids):
            rows = panel.plaintext[panel.family_ids == base]
            assert rows.shape == (8, 2) and len(np.unique(rows)) == 16
            expected = set()
            for tag in range(8):
                offset = sum(((tag >> j) & 1) << bit for j, bit in enumerate((0, 21, 42, 60)))
                left = int(base) | offset
                expected.add((left, left ^ mask))
            assert set(map(tuple, rows.tolist())) == expected
        at = np.asarray([len(panel) - 1, 0], dtype=np.intp)
        x, y = panel.batch(at)
        assert x.shape == y.shape == (2, 128)
        assert x.dtype == np.float32 and y.dtype == np.uint8
        for row, pair in zip(x, panel.plaintext[at], strict=True):
            assert row.tolist() == [int(b) for word in pair for b in f'{word:064b}']
        assert np.array_equal(y, 1 - x)
        assert np.array_equal(x[:, config.ACTIVE_COLUMNS[:4]], 1 - x[:, config.ACTIVE_COLUMNS[4:]])
        raw = panel.archive()
        assert sha256(panel.archive()).digest() == sha256(raw).digest()
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            assert np.array_equal(archive['family_ids'], panel.family_ids)
            assert np.array_equal(archive['plaintext'], panel.plaintext)
            assert np.array_equal(archive['ciphertext'], panel.ciphertext)
    assert [sorted(np.unique(p.family_ids)) for p in (train, dev, test)] == [bases[:2], [8], [16]]
    counts = cube_meter.snapshot()
    assert counts['attempted_real_blocks'] == counts['completed_real_blocks'] == 64


@pytest.mark.parametrize('failure', ['oracle', 'shape', 'dtype', 'domain', 'draw'])
def test_cube_provider_fault_is_not_refunded_or_retried(cube_meter: Meter, failure: str) -> None:
    from cipher_lab_plugins.direct_block.cube_compare.data import CubePanels

    def oracle(points: Words) -> Words:
        if failure == 'oracle':
            raise RuntimeError('must not leak')
        if failure == 'shape':
            return points.reshape(-1, 2)
        if failure == 'dtype':
            return points.astype(np.int64)  # malformed provider
        return points

    panels = CubePanels(oracle, cube_meter, families=(2, 1, 1),
                        draw=lambda: -1 if failure == 'draw' else 2)
    reason = {'oracle': 'ORACLE_FAILED', 'shape': 'ORACLE_RESPONSE', 'dtype': 'ORACLE_RESPONSE',
              'domain': 'DOMAIN_REUSED', 'draw': 'RANDOM_UNAVAILABLE'}[failure]
    # Scope oracle faults at the first complete family, not at duplicate drawing.
    if failure in {'oracle', 'shape', 'dtype'}:
        panels = CubePanels(oracle, cube_meter, families=(1, 1, 1), draw=lambda: 2)
    with pytest.raises(Stopped, match=reason):
        panels.build('train')
    counts = cube_meter.snapshot()
    assert counts['attempted_real_blocks'] == (16 if reason.startswith('ORACLE') else 0)
    assert counts['completed_real_blocks'] == 0
    with pytest.raises(ValueError, match='split order'):
        panels.build('train')
    with pytest.raises(ValueError, match='split order'):
        panels.label_test()

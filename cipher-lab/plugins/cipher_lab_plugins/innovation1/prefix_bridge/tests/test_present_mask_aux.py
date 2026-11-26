from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cipher_lab.execution import Meter


def test_mask_hides_either_endpoint_and_xor_without_mutating_full() -> None:
    from cipher_lab_plugins.innovation1.mask_aux.masking import mask_for_step, masked_view

    rng = np.random.default_rng(93)
    full = np.zeros((3, 4, 6, 64), np.float32)
    full[:, :, 3:5] = rng.integers(0, 2, (3, 4, 2, 64))
    full[:, :, 5] = full[:, :, 3] != full[:, :, 4]
    original = full.copy()
    mask = mask_for_step(3, 1)
    assert mask.shape == (3, 4, 2, 64) and mask.dtype == np.bool_
    assert mask[:, :, 0].any() and mask[:, :, 1].any()
    assert (mask.sum(axis=2) <= 1).all() and (mask.sum(axis=(2, 3)) == 16).all()
    assert np.array_equal(mask, mask_for_step(3, 1))
    assert not np.array_equal(mask, mask_for_step(3, 2))
    x, indices, y = masked_view(full, mask)
    xor = mask.any(axis=2)
    other = full.copy()
    other[:, :, 3:5][mask] = 1 - other[:, :, 3:5][mask]
    other[:, :, 5] = other[:, :, 3] != other[:, :, 4]
    assert np.all(other[:, :, 5][xor] != full[:, :, 5][xor])
    xx, ii, yy = masked_view(other, mask)
    assert np.array_equal(full, original) and not np.shares_memory(full, x)
    assert np.array_equal(x, xx) and np.array_equal(indices, ii)
    assert np.array_equal(y, full[:, :, 3:5][mask]) and np.array_equal(yy, 1 - y)
    assert np.array_equal(indices, np.flatnonzero(mask)) and len(indices) == 3 * 64
    assert np.array_equal(x[:, :, :2], mask) and np.array_equal(x[:, :, 2], xor)
    assert not x[:, :, 3:5][mask].any() and not x[:, :, 5][xor].any()
    assert np.array_equal(x[:, :, 3:5][~mask], full[:, :, 3:5][~mask])
    assert np.array_equal(x[:, :, 5][~xor], full[:, :, 5][~xor])


def test_recipe_frozen_caps() -> None:
    from cipher_lab_plugins.innovation1.mask_aux import config

    assert config.CAPS == dict(real_blocks=589828, null_words=0, optimizer_updates=15360,
                               forward_rows=3506176)
    assert config.counts(18, 19) == dict(real_blocks=589828, null_words=0,
                                        optimizer_updates=9472, forward_rows=2203648)
    assert config.recipe()['rounds'] == [7]
    for a, b in ((17, 18), (30, 31), (True, 20), (18., 18)):
        with pytest.raises(ValueError):
            config.counts(a, b)  # type: ignore[arg-type]


def test_only_public_31_and_7_gate(bridge_meter: Meter) -> None:
    from cipher_lab_plugins.innovation1.mask_aux import kat

    calls: list[tuple[str, int, int, Any]] = []

    def scalar(p: int, k: Any, r: int) -> int:
        calls.append(('scalar', r, p, k))
        return 0x5579C1387B228445 if r == 31 else 0x987

    def oracle(p: Any, k: Any, r: tuple[int, ...]) -> Any:
        assert len(r) == 1 and p.shape == (1,)
        calls.append(('batch', r[0], int(p[0]), k))
        return np.array([[0x5579C1387B228445 if r == (31,) else 0x987]], np.uint64)

    kat.check(bridge_meter, scalar, oracle)
    assert calls == [('scalar', 31, 0, (0, 0, 0)), ('batch', 31, 0, (0, 0, 0)),
                     ('scalar', 7, 0x0123456789ABCDEF, (0x06070809, 0x02030405, 1)),
                     ('batch', 7, 0x0123456789ABCDEF, (0x06070809, 0x02030405, 1))]
    assert bridge_meter.snapshot()['completed_real_blocks'] == 4


@pytest.mark.parametrize('fault', ['', 'overlap', 'wrong-shape', 'alias', 'mutate'])
def test_shared_panel_c7_only_and_single_use_holdout(
    fault: str, monkeypatch: pytest.MonkeyPatch, bridge_meter: Meter,
) -> None:
    from dataclasses import replace

    from cipher_lab_plugins.innovation1.mask_aux import data
    from cipher_lab_plugins.innovation1.prefix_bridge import config

    for field in ('TRAIN_ROWS', 'DEV_ROWS', 'TEST_ROWS'):
        monkeypatch.setattr(config, field, 2)
    monkeypatch.setattr(bridge_meter, '_policy', replace(bridge_meter._policy, real_blocks=48))
    draws: list[int] = []
    supplied: list[Any] = []
    raw: list[Any] = []
    mask = np.uint64(0xABCD)

    def draw(n: int) -> int:
        draws.append(n)
        return 0x1234567890ABCDEF0123

    def sample(rng: Any, rows: int) -> Any:
        x = np.arange(rows*8, dtype=np.uint64).reshape(rows, 4, 2) + np.uint64(100 * len(raw))
        if fault == 'overlap' and len(raw) == 2:
            # Construct distinct pairs except for one repeated *unordered* pair
            # in both label groups, so label permutation cannot hide the repeat.
            train = supplied[0]
            neg, = [group for group in train if (group[0, 0] ^ group[0, 1]) != 0xD00000]
            pos, = [group for group in train if (group[0, 0] ^ group[0, 1]) == 0xD00000]
            x[:, 0] = neg[0, ::-1]
            x[:, 1] = pos[0, ::-1]
        raw.append(x.copy())
        return x

    def encrypt(p: Any, k: Any, rounds: tuple[int, ...]) -> Any:
        assert k == (0xCDEF0123, 0x567890AB, 0x1234) and rounds == (7,)
        supplied.append(p.copy().reshape(2, 4, 2))
        if fault == 'wrong-shape':
            return np.repeat(p[:, None], 6, axis=1)
        if fault == 'alias':
            return p[:, None]
        if fault == 'mutate':
            p[0] ^= np.uint64(1)
        return (p ^ mask).reshape(-1, 1)

    if fault in ('wrong-shape', 'alias', 'mutate'):
        with pytest.raises(ValueError, match='PREFIX_ORACLE_RESPONSE'):
            data.make_panels(bridge_meter, draw, encrypt, sample=sample)
        assert bridge_meter.snapshot()['attempted_real_blocks'] == 16
        assert bridge_meter.snapshot()['completed_real_blocks'] == 0
        assert draws == [80]
        return
    panels = data.make_panels(bridge_meter, draw, encrypt, sample=sample)
    assert draws == [80] and len(supplied) == 2
    with pytest.raises(ValueError, match='TEST_NOT_READY'):
        panels.view('test', 'r7-cipher')
    for invalid in ('s2-only', 'r7-joint', 'r6-cipher'):
        with pytest.raises(ValueError, match='INVALID_SLOT'):
            panels.train(invalid)
    for i, split in enumerate((panels.train('r7-cipher'), panels.dev('r7-cipher'))):
        x, y = split.batch(np.arange(2, dtype=np.intp))
        assert y.shape == (2, 1) and int(y.sum()) == 1 and not x[:, :, :3].any()
        positive = y[:, 0] == 1
        p, original = supplied[i], raw[i]
        assert np.array_equal(p[~positive], original[~positive])
        assert np.array_equal(p[positive, :, 0], original[positive, :, 0])
        assert np.all((p[positive, :, 0] ^ p[positive, :, 1]) == config.DIFFERENCE)
        for col in (0, 1):
            words = (p[..., col] ^ mask).astype('>u8').copy()
            expected = np.unpackbits(words.view(np.uint8), axis=-1)
            assert np.array_equal(x[:, :, col+3], expected.reshape(2, 4, 64))
        assert np.array_equal(x[:, :, 5], x[:, :, 3] != x[:, :, 4])
    if fault == 'overlap':
        with pytest.raises(ValueError, match='SPLIT_PAIR_OVERLAP'):
            panels.test()
        assert len(supplied) == 2 and bridge_meter.snapshot()['completed_real_blocks'] == 32
        assert 'test' not in panels.hashes
    else:
        panels.test()
        assert len(supplied) == 3 and bridge_meter.snapshot()['completed_real_blocks'] == 48
        assert 'test' in panels.hashes
    with pytest.raises(ValueError, match='TEST_ALREADY_REQUESTED'):
        panels.test()
    assert draws == [80]


def test_round_seams_reject_all_other_overrides_before_calls(bridge_meter: Meter) -> None:
    from cipher_lab_plugins.innovation1.prefix_bridge import data, kat

    def no_call(*args: Any) -> Any:
        pytest.fail('unexpected oracle or entropy call')

    invalid: tuple[Any, ...] = ((2, 7), (7.,), (True,), (), (31, 7))
    for rounds in invalid:
        with pytest.raises(ValueError, match='INVALID_ROUNDS'):
            data.make_panels(bridge_meter, no_call, no_call, rounds=rounds)
        with pytest.raises(ValueError, match='INVALID_ROUNDS'):
            kat.check(bridge_meter, no_call, no_call, rounds=rounds)
    assert bridge_meter.snapshot()['attempted_real_blocks'] == 0

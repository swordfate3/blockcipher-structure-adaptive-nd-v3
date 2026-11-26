from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from cipher_lab_plugins.innovation1.prefix_bridge.data import Split, Words
from cipher_lab_plugins.innovation1.prefix_bridge.kat import Key

from cipher_lab.execution import Meter, Stopped


@pytest.fixture(scope='module')
def s2_inverse_lifecycle() -> Iterator[Callable[..., Any]]:
    from cipher_lab.ciphers import present_s2

    original = present_s2._invert_into
    yield present_s2.invert_second_output
    assert present_s2._invert_into is original


@pytest.fixture
def s2_with_stub(s2_inverse_lifecycle: Callable[..., Any],
                 bridge_manual_only: tuple[pytest.MonkeyPatch, Callable[..., Any]],
                 ) -> Callable[[Callable[..., None]], Callable[..., Any]]:
    from cipher_lab.ciphers import present_s2

    guard, _ = bridge_manual_only
    with pytest.raises(AssertionError, match='PREFIX_BRIDGE_MANUAL_ONLY'):
        # Empty args cannot execute any round even if the guard regresses.
        present_s2._invert_into()  # type: ignore[call-arg]

    def bind(stub: Callable[..., None]) -> Callable[..., Any]:
        guard.setattr(present_s2, '_invert_into', stub)
        return s2_inverse_lifecycle

    return bind


def test_guard_poisoned_inverse_boundaries_and_prebound_aliases(
    bridge_guard_probe: tuple[list[str], tuple[Any, ...]],
) -> None:
    from cipher_lab.ciphers import present_s2

    calls, (*_, aliases) = bridge_guard_probe
    for fn in (present_s2.invert_second_output, present_s2._invert_into,
               aliases.public_inverse, aliases.private_inverse):
        with pytest.raises(AssertionError, match='PREFIX_BRIDGE_MANUAL_ONLY'):
            cast(Callable[[], None], fn)()
    assert not calls


@pytest.mark.parametrize('points,key', [
    (np.empty(0, np.uint64), (0, 0, 0)),
    (np.zeros(65537, np.uint64), (0, 0, 0)),
    (np.zeros((1, 1), np.uint64), (0, 0, 0)),
    (np.zeros(4, np.uint64)[::2], (0, 0, 0)),
    (np.zeros(1, np.int64), (0, 0, 0)),
    (np.zeros(1, np.dtype('u8').newbyteorder('S')), (0, 0, 0)),
    ([0], (0, 0, 0)), (None, (0, 0, 0)),
    (np.zeros(1, np.uint64), [1, 2, 3]),
    (np.zeros(1, np.uint64), (1, 2)),
    (np.zeros(1, np.uint64), (True, 2, 3)),
    (np.zeros(1, np.uint64), (1, np.uint32(2), 3)),
    (np.zeros(1, np.uint64), (-1, 2, 3)),
    (np.zeros(1, np.uint64), (2**32, 2, 3)),
    (np.zeros(1, np.uint64), (1, 2**32, 3)),
    (np.zeros(1, np.uint64), (1, 2, 2**16)),
])
def test_inverse_rejects_noncanonical_inputs(s2_with_stub: Callable[..., Any],
                                             points: Any, key: Any) -> None:
    with pytest.raises(ValueError):
        s2_with_stub(lambda *_: pytest.fail('invalid input reached private inverse'))(points, key)


@pytest.mark.parametrize('n', [1, 65536])
def test_inverse_does_not_mutate_or_alias_input(s2_with_stub: Callable[..., Any], n: int) -> None:
    a = np.arange(n, dtype=np.uint64)
    a.flags.writeable = False
    k = (2**32 - 1, 2**32 - 2, 2**16 - 1)
    calls: list[bool] = []

    def stub(state: Any, key: object) -> None:
        assert key is k and np.array_equal(state, a) and not np.shares_memory(state, a)
        calls.append(True)
        state.fill(71)

    out = s2_with_stub(stub)(a, k)
    assert calls == [True] and not np.shares_memory(a, out)
    assert out.dtype == np.uint64 and out.flags.c_contiguous and out.shape == (n,)
    assert np.all(out == 71) and np.array_equal(a, np.arange(n, dtype=np.uint64))


def test_static_inverse_layout_matches_independent_scalar_formula() -> None:
    from cipher_lab.ciphers import present, present_prefix, present_s2

    assert vars(present_s2)['present_prefix'] is present_prefix
    assert present._SBOX == present_prefix._SBOX
    assert tuple(present_s2._INV_SBOX[v] for v in present._SBOX) == tuple(range(16))
    for i, sources in enumerate(present_s2._NIBBLE_SOURCES):
        assert sources == tuple(63 if b == 63 else (16*b) % 63 for b in range(4*i, 4*i+4))


@pytest.mark.parametrize('fault,pay', [('', (5, 5)), ('scalar', (1, 0)), ('scalar2', (2, 1)),
                                       ('inverse', (3, 2)), ('forward', (5, 3))])
def test_gate_order_and_accounting(bridge_meter: Meter, fault: str, pay: tuple[int, int]) -> None:
    from cipher_lab_plugins.innovation1.s2_window import kat

    meter = bridge_meter
    events: list[str] = []

    def scalar(p: int, k: Any, r: int) -> int:
        assert p == kat._PUBLIC_POINT and k == kat._PUBLIC_KEY and p and all(k)
        events.append(f'scalar{r}')
        assert meter.snapshot()['attempted_real_blocks'] == len(events)
        return -1 if fault == ('scalar' if r == 2 else 'scalar2') else 40 + r

    def inverse(s: Any, k: Any) -> Any:
        assert list(s) == [42] and k == kat._PUBLIC_KEY
        assert meter.snapshot()['attempted_real_blocks'] == 3
        events.append('inverse')
        return np.array([0 if fault == 'inverse' else kat._PUBLIC_POINT], np.uint64)

    def forward(p: Any, k: Any, r: Any) -> Any:
        assert list(p) == [kat._PUBLIC_POINT] and k == kat._PUBLIC_KEY and r == (2, 7)
        assert meter.snapshot()['attempted_real_blocks'] == 5
        events.append('forward')
        return np.array([[42, 48 if fault else 47]], np.uint64)

    if fault:
        with pytest.raises(ValueError):
            kat.check(meter, scalar, inverse, forward)
    else:
        kat.check(meter, scalar, inverse, forward)
    assert events == ['scalar2', 'scalar7', 'inverse', 'forward'][:{
        'scalar': 1, 'scalar2': 2, 'inverse': 3, 'forward': 4, '': 4}[fault]]
    assert tuple(meter.snapshot()[f'{p}_real_blocks'] for p in ('attempted', 'completed')) == pay


@pytest.mark.parametrize('fault', ['', 'inverse-alias', 'inverse-shape', 'inverse-mutate',
                                   'forward-alias', 'forward-s2-positive', 'forward-s2-negative',
                                   'forward-mutate', 'forward-type', 'overlap', 'reverse-overlap',
                                   'capacity', 'test'])
def test_provider_validates_both_classes_and_retains_only_c7(
    bridge_meter: Meter, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    from cipher_lab_plugins.innovation1.s2_window import config, data

    meter = bridge_meter
    monkeypatch.setattr(meter, '_policy', replace(meter._policy,
                        real_blocks=47 if fault == 'capacity' else 144))
    for kind in ('TRAIN_ROWS', 'DEV_ROWS', 'TEST_ROWS'):
        monkeypatch.setattr(config, kind, 2)
    # Exercise chunking at the smallest group boundary without changing the recipe.
    monkeypatch.setattr(data, '_MAX_POINTS', 8)
    key = (0x1234, 0x5678, 9)
    draws: list[int] = []
    samples: list[Any] = []
    boundaries: list[Any] = []
    finals: list[Any] = []
    log: list[str] = []
    hit = 0

    def draw(bits: int) -> int:
        draws.append(bits)
        return key[0] | key[1] << 32 | key[2] << 64

    def sample(rng: Any, rows: int) -> Any:
        assert rows == 2
        if fault == 'test' and len(samples) == 2:
            log.append('test-failed')
            raise ValueError('private detail')
        a = np.arange(16, dtype=np.uint64).reshape(2, 4, 2) + np.uint64(100*len(samples))
        if fault in ('overlap', 'reverse-overlap') and samples:
            a = samples[0].copy()
            if fault == 'reverse-overlap':
                a = a[..., ::-1]
        samples.append(a.copy())
        return a

    def inverse(s: Any, k: Any) -> Any:
        assert k == key
        assert tuple(meter.snapshot()[f'{p}_real_blocks'] for p in ('attempted', 'completed')) == (
            24*len(boundaries) + 8, 24*len(boundaries))
        boundaries.append(s.copy())
        log.append('i')
        if fault == 'inverse-mutate':
            s[0] ^= np.uint64(1)
        if fault == 'inverse-alias':
            return s
        result = s ^ np.uint64(0x7F00)
        return result.reshape(-1, 1) if fault == 'inverse-shape' else result

    def forward(p: Any, k: Any, rounds: Any) -> Any:
        nonlocal hit
        assert k == key and rounds == (2, 7)
        recovered = p ^ np.uint64(0x7F00)
        assert np.array_equal(recovered, boundaries[-1])
        assert tuple(meter.snapshot()[f'{q}_real_blocks'] for q in ('attempted', 'completed')) == (
            24*len(boundaries), 24*len(boundaries) - 16)
        log.append('f')
        c = p ^ np.uint64(2**63)
        finals.append(c)
        result = np.stack((recovered, c), axis=1)
        positive = int(result[0, 0] ^ result[1, 0]) == config.DIFFERENCE
        if (fault == 'forward-s2-positive' and positive
                or fault == 'forward-s2-negative' and not positive):
            # Flip BOTH endpoints: a mere XOR-difference check would miss this.
            result[:2, 0] ^= np.uint64(1)
            hit += 1
        if fault == 'forward-alias':
            p.resize((len(c), 2), refcheck=False)
            return p.view()
        if fault == 'forward-mutate':
            p[0] ^= np.uint64(1)
        return result.astype(np.int64) if fault == 'forward-type' else result

    if fault not in ('', 'test'):
        with pytest.raises((ValueError, Stopped)) as err:
            data.make_panels(meter, draw, inverse, forward, sample=sample)
        assert 'private detail' not in str(err.value)
        assert draws == [80]
        if fault in ('forward-s2-positive', 'forward-s2-negative'):
            assert hit == 1
        if fault in ('overlap', 'reverse-overlap'):
            assert 'SPLIT_PAIR_OVERLAP' in str(err.value) and log == ['i', 'f'] * 2
        elif fault.startswith('inverse'):
            assert log == ['i'] and meter.snapshot()['completed_real_blocks'] == 0
        elif fault == 'capacity':
            assert log == ['i', 'f', 'i'] and meter.snapshot()['completed_real_blocks'] == 32
        else:
            assert log[-2:] == ['i', 'f']
            assert meter.snapshot()['attempted_real_blocks'] == (
                cast(int, meter.snapshot()['completed_real_blocks']) + 16)
        return
    panels = data.make_panels(meter, draw, inverse, forward, sample=sample)
    assert log == ['i', 'f'] * 4 and len(samples) == 2 and draws == [80]
    with pytest.raises(ValueError, match='TEST_NOT_READY'):
        panels.view('test')
    if fault == 'test':
        with pytest.raises(ValueError, match='PANEL_SAMPLING_FAILED'):
            panels.test()
        assert log.count('test-failed') == 1 and set(panels.hashes) == {'train', 'dev'}
    else:
        panels.test()
        assert log == ['i', 'f'] * 6 and len(samples) == 3 and draws == [80]
        splits: tuple[Split, ...] = ('train', 'dev', 'test')
        for j, split in enumerate(splits):
            panel = panels.view(split)
            x, y = panel.batch(np.arange(2, dtype=np.intp))
            s2 = np.stack(boundaries[2*j:2*j+2]).reshape(2, 4, 2)
            expected = samples[j].copy()
            pos = y[:, 0] == 1
            assert y.shape == (2, 1) and int(y.sum()) == 1
            expected[pos, :, 1] = expected[pos, :, 0] ^ np.uint64(config.DIFFERENCE)
            assert np.array_equal(s2, expected)
            c = np.stack(finals[2*j:2*j+2]).reshape(2, 4, 2)
            assert panel._final is not None and not panel._final.flags.writeable
            assert panel._s2 is None and np.array_equal(panel._final, c)
            assert np.all(x[:, :, :3] == 0)
            for channel, words in enumerate((c[:, :, 0], c[:, :, 1], c[:, :, 0] ^ c[:, :, 1]), 3):
                bits = [[[int(b) for b in f'{w:064b}'] for w in row] for row in words]
                assert x[:, :, channel].tolist() == bits
        assert set(panels.hashes) == {'train', 'dev', 'test'}
        assert meter.snapshot()['completed_real_blocks'] == 144
    with pytest.raises(ValueError, match='TEST_ALREADY_REQUESTED'):
        panels.test()
    assert panels._generate_test is None


def test_forward_rejects_alias_of_recovered_plain_even_when_s2_matches(bridge_meter: Meter) -> None:
    from cipher_lab_plugins.innovation1.s2_window import data

    output = np.array([[12, 90], [13, 91]], dtype=np.uint64)
    s2 = np.array([12, 13], dtype=np.uint64)
    p = output.ravel()[:2]
    key = (3, 2, 1)

    def aliases(supplied: Words, k: Key, rounds: tuple[int, ...]) -> Words:
        assert k == key and rounds == (2, 7) and np.array_equal(supplied, p)
        assert not np.shares_memory(supplied, output)
        return output

    with pytest.raises(ValueError, match='PREFIX_ORACLE_RESPONSE'):
        data._full_forward(bridge_meter, p, s2, key, aliases)
    assert bridge_meter.snapshot()['attempted_real_blocks'] == 4
    assert bridge_meter.snapshot()['completed_real_blocks'] == 0

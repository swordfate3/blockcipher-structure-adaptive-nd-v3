from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from cipher_lab.execution import Meter, Stopped

Words = NDArray[np.uint64]
Key = tuple[int, int, int]
_ZERO_KAT = 0x5579C1387B228445
_PUBLIC_KEY = (0x06070809, 0x02030405, 0x0001)
_PUBLIC_POINT = 0x0123456789ABCDEF


@pytest.mark.parametrize(('failure', 'attempted', 'completed'), [
    ('', 14, 14), ('scalar-zero', 1, 0), ('batch-zero', 2, 1),
    ('scalar-width', 3, 2), ('batch-compare', 4, 3), ('batch-dtype', 4, 3),
    ('batch-shape', 4, 3), ('batch-alias', 4, 3), ('batch-mutate', 4, 3),
    ('batch-error', 4, 3),
])
def test_public_gate_inputs_and_fail_closed(
    bridge_meter: Meter, failure: str, attempted: int, completed: int,
) -> None:
    from cipher_lab_plugins.innovation1.prefix_bridge import kat

    calls: list[tuple[str, int, Key, int]] = []
    downstream: list[str] = []
    assert sum(w << (32 * i) for i, w in enumerate(_PUBLIC_KEY)) == 0x00010203040506070809

    def scalar(point: int, key: Key, r: int) -> int:
        calls.append(('scalar', point, key, r))
        if r == 31:
            return _ZERO_KAT if failure != 'scalar-zero' else 1
        return (1 << 64) if failure == 'scalar-width' else r + 500

    def prefixes(points: Words, key: Key, rounds: tuple[int, ...]) -> Any:
        assert len(rounds) == 1 and points.dtype == np.uint64 and points.shape == (1,)
        r, = rounds
        calls.append(('prefixes', int(points[0]), key, r))
        if r == 31:
            return np.array([[_ZERO_KAT if failure != 'batch-zero' else 1]], dtype=np.uint64)
        if failure == 'batch-error':
            raise RuntimeError('injected public batch failure')
        if failure == 'batch-alias':
            return points.reshape((1, 1))
        if failure == 'batch-mutate':
            points[0] = 0
        if failure == 'batch-shape':
            return np.array([r + 500], dtype=np.uint64)
        value = r + (501 if failure == 'batch-compare' else 500)
        return np.array([[value]], dtype=np.int64 if failure == 'batch-dtype' else np.uint64)

    def gate_then_manual_downstream() -> None:
        kat.check(bridge_meter, scalar, prefixes)
        downstream.extend(('fake-key', 'fake-fit'))

    if failure:
        with pytest.raises((RuntimeError, ValueError), match='public'):
            gate_then_manual_downstream()
        assert downstream == []
    else:
        gate_then_manual_downstream()
        assert downstream == ['fake-key', 'fake-fit']
        with pytest.raises(Stopped, match='REAL_BLOCKS_LIMIT'):
            bridge_meter.reserve('real_blocks', 1)
    plan = [(arm, 0 if r == 31 else _PUBLIC_POINT, (0, 0, 0) if r == 31 else _PUBLIC_KEY, r)
            for r in (31, 2, 3, 4, 5, 6, 7) for arm in ('scalar', 'prefixes')]
    assert calls == plan[:attempted]
    counts = bridge_meter.snapshot()
    assert counts['attempted_real_blocks'] == attempted
    assert counts['completed_real_blocks'] == completed
    assert counts['attempted_optimizer_updates'] == counts['attempted_forward_rows'] == 0


def test_production_adapter_uses_explicit_harmless_bindings(
    bridge_meter: Meter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.innovation1.prefix_bridge import kat

    from cipher_lab.ciphers import present, present_prefix

    calls: list[tuple[tuple[int, ...], Key, int]] = []

    def fake_many(points: tuple[int, ...], words: Key, r: int) -> tuple[int, ...]:
        calls.append((points, words, r))
        return (_ZERO_KAT if r == 31 else r,)

    def fake_prefixes(points: Words, key: Key, rounds: tuple[int, ...]) -> Words:
        assert (tuple(map(int, points)), key, rounds[0]) == calls[-1]
        return np.array([[_ZERO_KAT if rounds == (31,) else rounds[0]]], dtype=np.uint64)

    monkeypatch.setattr(present, 'encrypt_many', fake_many)
    monkeypatch.setattr(present_prefix, 'encrypt_prefixes', fake_prefixes)
    kat.production(bridge_meter)  # both cipher names rebound to the artificial adapters
    assert calls == [((0,), (0, 0, 0), 31), *[((_PUBLIC_POINT,), _PUBLIC_KEY, r)
                                            for r in range(2, 8)]]
    assert bridge_meter.snapshot()['completed_real_blocks'] == 14

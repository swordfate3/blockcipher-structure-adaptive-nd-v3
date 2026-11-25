from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

Words = NDArray[np.uint64]
Key = tuple[int, int, int]
BindStub = Callable[[Callable[..., None]], Callable[..., Words]]


@pytest.mark.parametrize(('points', 'key', 'rounds'), [
    ([1], (0, 0, 0), (2,)),
    (np.array([1], dtype=np.int64), (0, 0, 0), (2,)),
    (np.array([1], dtype=np.dtype(np.uint64).newbyteorder('S')), (0, 0, 0), (2,)),
    (np.zeros((2, 1), dtype=np.uint64), (0, 0, 0), (2,)),
    (np.zeros(6, dtype=np.uint64)[::2], (0, 0, 0), (2,)),
    (np.zeros(0, dtype=np.uint64), (0, 0, 0), (2,)),
    (np.zeros(65537, dtype=np.uint64), (0, 0, 0), (2,)),
    *[(np.zeros(1, dtype=np.uint64), key, (2,)) for key in (
        (2**32, 0, 0), (0, 2**32, 0), (0, 0, 2**16), (0, -1, 0),
        (0, 0, True), (0, 0), [0, 0, 0], (0, 0, np.uint16(0)),
    )],
    *[(np.zeros(1, dtype=np.uint64), (0, 0, 0), rounds) for rounds in (
        (2, 2), (3, 2), (0, 2), (2, 32), (True,), (np.int64(1),), (), [2],
    )],
])
def test_rejects_malformed_before_kernel(
    points: Any, key: Any, rounds: Any, bridge_prefix_with_stub: BindStub,
) -> None:
    def reached(*args: object) -> None:
        pytest.fail('malformed input reached kernel')

    encrypt = bridge_prefix_with_stub(reached)
    with pytest.raises(ValueError):
        encrypt(points, key, rounds)


@pytest.mark.parametrize('rows', [3, 65536])
def test_matrix_dispatch_to_harmless_kernel(rows: int, bridge_prefix_with_stub: BindStub) -> None:
    points = np.arange(rows, dtype=np.uint64)
    points.flags.writeable = False
    key = (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFF)
    rounds = (1, 2, 7, 31)
    calls: list[int] = []

    def sentinel(state: Words, words: Key, wanted: tuple[int, ...], out: Words) -> None:
        calls.append(len(state))
        assert words == key and wanted == rounds
        assert np.array_equal(state, points) and state.flags.writeable
        assert not np.shares_memory(state, points)
        for i, value in enumerate(wanted):
            out[:, i] = state + np.uint64(value)
        state[:] = 0  # emulate a mutating private work buffer, not a round function

    encrypt = bridge_prefix_with_stub(sentinel)
    one = encrypt(points, key, rounds)
    two = encrypt(points, key, rounds)
    assert one.shape == (rows, len(rounds)) and one.dtype == np.uint64 and one.flags.c_contiguous
    for i, value in enumerate(rounds):
        assert np.array_equal(one[:, i], points + np.uint64(value))
    assert calls == [rows, rows] and np.array_equal(points, np.arange(rows, dtype=np.uint64))
    assert np.array_equal(one, two)
    assert not np.shares_memory(one, two) and not np.shares_memory(one, points)


def test_owner_guard_stops_project_aliases_and_cached_dispatch(
    bridge_guard_probe: tuple[list[str], tuple[object, ...]],
) -> None:
    import torch

    from cipher_lab.ciphers import present, present_prefix

    calls, (*handles, aliases) = bridge_guard_probe
    tests = (*handles, present.encrypt_many, present_prefix.encrypt_prefixes,
             present_prefix._encrypt_into,
             torch.nn.functional.linear, torch.nn.Conv1d.forward,
             getattr(aliases, 'public_cipher'), getattr(aliases, 'public_prefix'),
             getattr(aliases, 'private_loop'), getattr(aliases, 'functional'),
             getattr(aliases, 'conv'), getattr(aliases, 'native_cached'),
             getattr(torch._C, '_VariableFunctions').linear, getattr(torch._VF, 'vf').linear)
    for blocked in tests:
        with pytest.raises(AssertionError, match='PREFIX_BRIDGE_MANUAL_ONLY'):
            cast(Callable[[], object], blocked)()  # no tensors/real kernel even on regression
    assert not calls


@pytest.mark.usefixtures('bridge_prefix_lifecycle_probe')
def test_owner_guard_restores_private_prefix_after_binding(
    bridge_prefix_with_stub: BindStub,
) -> None:
    from cipher_lab.ciphers import present_prefix

    def harmless(*args: object) -> None:
        pytest.fail('identity probe must not invoke the stub')

    blocked = present_prefix._encrypt_into
    bridge_prefix_with_stub(harmless)
    assert blocked is not harmless and present_prefix._encrypt_into is harmless


@pytest.mark.usefixtures('bridge_vf_lifecycle_probe')
def test_owner_guard_vf_restores_dynamic_and_explicit_attributes() -> None:
    import torch

    module = torch._VF
    assert getattr(module, 'linear') is getattr(getattr(module, 'vf'), 'linear')
    assert getattr(module, 'linear') is getattr(module, 'relu')

"""Public single-round arithmetic and fake wiring only. NO real encryption/KAT."""
from __future__ import annotations

import pytest
from speck_cipher_guards import (  # noqa: F401
    science_guard,
    torch,
)

from cipher_lab.ciphers import speck, speck_tensor

# Save wrappers only to run after every real round/schedule dependency is faked.
_scalar_wrapper = speck.encrypt_scalar
_tensor_wrapper = speck_tensor.encrypt_tensor


def literal_round(x: int, y: int, k: int) -> tuple[int, int]:
    a = ((x >> 7) | ((x & 127) << 9))
    a = ((a + y) % 65536) ^ k
    b = ((y * 4) % 65536) | (y // 16384)
    return a, b ^ a


def test_public_single_round_literal_arithmetic() -> None:
    for x, y, k in ((0, 0, 0), (65535, 1, 7), (1, 32768, 99), (1234, 5678, 42)):
        assert speck.enc_one_round((x, y), k) == literal_round(x, y, k)
        assert speck.rol16(x, 2) == ((x << 2) | (x >> 14)) & 65535
        assert speck.ror16(x, 7) == (x >> 7) | ((x & 127) << 9)
    tx = torch.tensor([0, 65535, 1, 1234], dtype=torch.int64)
    ty = torch.tensor([0, 1, 32768, 5678], dtype=torch.int64)
    tk = torch.tensor([0, 7, 99, 42], dtype=torch.int64)
    a, b = speck_tensor.enc_one_round((tx, ty), tk)
    assert list(zip(a.tolist(), b.tolist())) == [literal_round(*row) for row in zip(
        tx.tolist(), ty.tolist(), tk.tolist())]


def test_fake_multi_round_wrapper_and_public_key_word_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scalar_calls: list[object] = []

    def fake_round(pair: tuple[int, int], k: int) -> tuple[int, int]:
        scalar_calls.append((pair, k))
        return pair[0] + 1, pair[1] + 2

    monkeypatch.setattr(speck, 'enc_one_round', fake_round)
    # Schedule wiring is exercised only against fake arithmetic.
    assert speck.expand_key((4, 3, 2, 1), 3) == (1, 3, 5)
    assert scalar_calls == [((2, 1), 0), ((3, 3), 1)]
    scalar_calls.clear()
    monkeypatch.setattr(speck, 'expand_key', lambda key, rounds: (11, 12, 13))
    assert _scalar_wrapper((10 << 16) | 20, (4, 3, 2, 1), 3) == (13 << 16) | 26
    assert len(scalar_calls) == 3

    monkeypatch.setattr(speck_tensor, 'enc_one_round', lambda pair, k: (pair[0]+1, pair[1]+2))
    single = torch.tensor([4, 3, 2, 1], dtype=torch.int64)
    assert [int(v.item()) for v in speck_tensor.expand_key(single, 3)] == [1, 3, 5]
    monkeypatch.setattr(speck_tensor, 'expand_key', lambda key, rounds: (11, 12, 13))
    points = torch.tensor([(10 << 16) | 20, (30 << 16) | 40], dtype=torch.int64)
    for keys in (single, single.repeat(2, 1)):
        assert _tensor_wrapper(points, keys, 3).tolist() == [(13 << 16) | 26, (33 << 16) | 46]


def test_real_entries_and_cuda_are_poisoned() -> None:
    for call in (lambda: speck.encrypt_scalar(0, (0, 0, 0, 0), 22),
                 lambda: speck_tensor.encrypt_tensor(
                     torch.tensor([0]), torch.tensor([0, 0, 0, 0]), 22),
                 torch.cuda.is_available):
        with pytest.raises(AssertionError, match='forbids'):
            call()


def test_fake_wrapper_input_rejection_never_reaches_cipher(monkeypatch: pytest.MonkeyPatch) -> None:
    from speck_cipher_guards import forbidden

    monkeypatch.setattr(speck, 'enc_one_round', forbidden)
    for point in (True, -1, 2**32):
        with pytest.raises(ValueError):
            _scalar_wrapper(point, (4, 3, 2, 1), 3)
    for keys, rounds in (((4, 3, 2, 1), True), ((4, 3, 2, 65536), 3)):
        with pytest.raises(ValueError):
            speck.expand_key(keys, rounds)
    monkeypatch.setattr(speck_tensor, 'enc_one_round', forbidden)
    for points, tensor_keys in ((torch.tensor([-1]), torch.tensor([4, 3, 2, 1])),
                         (torch.tensor([0]), torch.tensor([4, 3, 2, 65536])),
                         (torch.tensor([0, 1]), torch.tensor([[4, 3, 2, 1]]))):
        with pytest.raises(ValueError):
            _tensor_wrapper(points, tensor_keys, 3)

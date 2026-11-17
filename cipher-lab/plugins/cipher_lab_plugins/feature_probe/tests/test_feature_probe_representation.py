from __future__ import annotations

import pytest
import torch

from cipher_lab_plugins.feature_probe.tests.feature_probe_fakes import poison

__all__ = ['poison']


def test_literal_features_and_structure() -> None:
    from cipher_lab_plugins.feature_probe.config import FeatureMLPSpec
    from cipher_lab_plugins.feature_probe.model import FeatureBitMLP
    from cipher_lab_plugins.feature_probe.representation import simon_r3_residual13
    x = torch.zeros(4, 64)
    x[1, 63-29] = 1  # Y29 -> l
    x[2, 63-22] = 1  # Y22 -> m
    x[3, 31-30] = 1  # X30 -> p
    expected = torch.zeros(4, 13)
    expected[:, 0] = 1
    expected[1, 1] = expected[2, 2] = expected[3, 4] = 1
    assert torch.equal(simon_r3_residual13(x), expected)
    model = FeatureBitMLP(FeatureMLPSpec(128, 'simon-r3-residual13-v1'))
    assert model.layers[0].weight.shape == (128, 13)
    assert sum(p.numel() for p in model.parameters()) == 18433


def test_public_symbolic_coordinates() -> None:
    from cipher_lab_plugins.feature_probe.representation import simon_r3_residual13
    # Enumerate seven independent public coordinates, X=0 except p/q source.
    x = torch.zeros(128, 64)
    expected = []
    for row in range(128):
        l, m, n, p, q, s, t = [(row >> j) & 1 for j in range(7)]  # noqa: E741
        for j, bit in ((29, l), (22, m), (15, n), (28, s), (21, t)):
            x[row, 63-j] = bit
        x[row, 31-30] = p ^ s
        x[row, 31-23] = q ^ t
        expected.append([1, l, m, n, p, q, s, t, l*m, m*n, m*q, p*m,
                         (l*n) ^ (l*q) ^ (p*n)])
    assert torch.equal(simon_r3_residual13(x), torch.tensor(expected, dtype=torch.float32))


@pytest.mark.parametrize('x', [torch.zeros(1, 13), torch.zeros(0, 64),
    torch.zeros(1, 64, dtype=torch.int64), torch.full((1, 64), .5),
    torch.full((1, 64), float('nan'))])
def test_invalid(x: torch.Tensor) -> None:
    from cipher_lab_plugins.feature_probe.representation import simon_r3_residual13
    with pytest.raises((TypeError, ValueError)):
        simon_r3_residual13(x)


def test_dense_public_wiring_against_scalar_boolean_definition() -> None:
    from cipher_lab_plugins.feature_probe.representation import simon_r3_residual13
    # Deterministic public patterns exercise the nonlinear A coordinates too;
    # this is not encryption, a round implementation, or a secret-key solver.
    words = [(0xAAAAAAAA ^ (i * 0x01010101), 0x55555555 ^ (i * 0x00010101))
             for i in range(32)]
    rows = [[(word >> j) & 1 for word in pair for j in range(31, -1, -1)]
            for pair in words]
    expected = []
    for high, low in words:
        def x(j: int) -> int:
            return (high >> (j % 32)) & 1

        def a(j: int) -> int:
            return ((low >> (j % 32)) & 1) ^ (x(j-1) & x(j-8)) ^ x(j-2)
        ell, m, n, s, t = a(29), a(22), a(15), a(28), a(21)
        p, q = x(30) ^ s, x(23) ^ t
        expected.append([1, ell, m, n, p, q, s, t, ell*m, m*n, m*q, p*m,
                         (ell*n) ^ (ell*q) ^ (p*n)])
    actual = simon_r3_residual13(torch.tensor(rows, dtype=torch.float32))
    assert torch.equal(actual, torch.tensor(expected, dtype=torch.float32))

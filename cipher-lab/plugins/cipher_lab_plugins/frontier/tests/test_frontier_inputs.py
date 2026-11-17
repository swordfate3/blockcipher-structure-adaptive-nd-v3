"""Fixed public rows, no model/labels/cipher/randomness/optimizer calls."""
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from cipher_lab_plugins.frontier.frontier_inputs import frontier_features
from cipher_lab_plugins.frontier.simon_frontier import PublicFeaturePlan


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError('network/optimizer forbidden in Task1')
    for cls in (torch.nn.Module, torch.nn.Linear):
        monkeypatch.setattr(cls, 'forward', forbidden)
    monkeypatch.setattr(torch.nn.functional, 'linear', forbidden)
    monkeypatch.setattr(Tensor, 'backward', forbidden)
    monkeypatch.setattr(torch.optim, 'Adam', forbidden)


def plan() -> PublicFeaturePlan:
    # leaf0=x0 XOR x2*x63, leaf1=1 XOR x1; bit0 is rawcol0, not LSB.
    return PublicFeaturePlan(5, 0, 'simon-r5-factored-v1',
                             ((0, 2), (1, (1 << 2) | (1 << 63))),
                             ((), (0,), (0, 1), (1,)))


def literal(row: list[int], p: PublicFeaturePlan) -> list[int]:
    leaves = []
    for leaf in p.leaves:
        value = 0
        for mask in leaf:
            term = 1
            for i in range(64):
                if mask >> i & 1:
                    term &= row[i]
            value ^= term
        leaves.append(value)
    features = []
    for feature in p.features:
        value = 1
        for i in feature:
            value &= leaves[i]
        features.append(value)
    return features


def test_fixed_public_rows_hand_calculation_and_independent_literal() -> None:
    rows = [[0] * 64, [1] + [0] * 63, [i % 2 for i in range(64)],
            [1] * 64, [0, 0, 1] + [0] * 60 + [1]]
    inputs = torch.tensor(rows, dtype=torch.float32)
    before = inputs.clone()
    actual = frontier_features(inputs, plan())
    assert actual.dtype == torch.float32
    assert actual.shape == (5, 4)
    assert actual.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0],
                               [1, 0, 0, 0], [1, 1, 1, 1]]
    assert actual.tolist() == [literal(row, plan()) for row in rows]
    assert torch.equal(inputs, before)
    assert torch.equal(torch.cat([frontier_features(inputs[:2], plan()),
                                  frontier_features(inputs[2:], plan())]), actual)


@pytest.mark.parametrize('bad', [torch.zeros((0, 64)), torch.zeros((2, 63)),
    torch.zeros((64,)), torch.zeros((2, 64), dtype=torch.float64),
    torch.zeros((2, 64), dtype=torch.bool), torch.full((2, 64), 0.5),
    torch.full((2, 64), float('nan')), torch.full((2, 64), float('inf')),
    torch.zeros((2, 64)).to_sparse()])
def test_adapter_rejects_shape_dtype_value_layout(bad: Tensor) -> None:
    with pytest.raises(ValueError):
        frontier_features(bad, plan())

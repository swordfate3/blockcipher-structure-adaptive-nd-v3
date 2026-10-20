"""Artificial task truth function and relevant-domain separation."""
from __future__ import annotations

from itertools import product

import pytest
import torch

from cipher_experiments.data.circuit import circuit_labels, make_circuit_inputs


def test_all_toy_labels_match_independent_scalar_truth_function() -> None:
    assignments = list(product((0, 1), repeat=8))
    x = torch.zeros(256, 64)
    x[:, :8] = torch.tensor(assignments).float()
    expected = [float((b[0] * b[1] + b[2] * b[3] + b[4] * b[5] + b[6] * b[7]) % 2)
                for b in assignments]
    assert torch.equal(circuit_labels(x), torch.tensor(expected).reshape(-1, 1))
    # Ignore padding deliberately; ensure the effective inputs, not padding, differ.
    x[:, 8:] = 1
    assert torch.equal(circuit_labels(x), torch.tensor(expected).reshape(-1, 1))


def test_unique_relevant_assignments_and_rng_isolation() -> None:
    torch.manual_seed(845)
    before = torch.get_rng_state().clone()
    panels = make_circuit_inputs(128, 64, 64, seed=191601)
    assert torch.equal(before, torch.get_rng_state())
    whole = torch.cat(list(panels.values()))
    assert whole.shape == (256, 64) and whole.dtype == torch.float32
    assert torch.unique(whole[:, :8], dim=0).shape[0] == 256
    assert not bool(whole[:, 8:].any())
    for name, x in panels.items():
        assert torch.equal(x, make_circuit_inputs(128, 64, 64, seed=191601)[name])


@pytest.mark.parametrize("counts", [(128, 64, 65), (0, 1, 1), (True, 1, 1)])
def test_toy_counts_rejected(counts: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        make_circuit_inputs(*counts, seed=1)

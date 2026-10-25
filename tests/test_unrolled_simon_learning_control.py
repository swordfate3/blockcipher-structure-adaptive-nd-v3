"""Positive control for a public, artificial single-round Boolean target.

All 16 four-bit corners are training examples, so this checks fitting only;
100% accuracy is not evidence of generalization or cipher learning.
"""

from __future__ import annotations

from itertools import product

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from cipher_experiments.models.unrolled_simon import UnrolledSimonPredictor


@pytest.mark.parametrize("polarity", [0, 1])
def test_single_round_model_learns_artificial_boolean_table(polarity: int) -> None:
    """Fit y XOR (a AND b) XOR c XOR polarity on all public input corners."""
    previous_num_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            corners = list(product((0.0, 1.0), repeat=4))
            inputs = torch.zeros((len(corners), 64), dtype=torch.float32)
            targets = torch.tensor(
                [int(y) ^ (int(a) & int(b)) ^ int(c) ^ polarity for y, a, b, c in corners],
                dtype=torch.float32,
            ).unsqueeze(1)

            # Position zero reads (low[0], high[1], high[8], high[2]),
            # corresponding to public input coordinates (32, 1, 8, 2).
            for row, (y, a, b, c) in enumerate(corners):
                inputs[row, 32] = y
                inputs[row, 1] = a
                inputs[row, 8] = b
                inputs[row, 2] = c

            model = UnrolledSimonPredictor(32, rounds=1, target_bit=0)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            with torch.no_grad():
                initial_loss = float(
                    F.binary_cross_entropy_with_logits(model(inputs), targets)
                )

            first_cell_grad = 0.0
            first_condition_grad = 0.0
            for update in range(1000):
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(inputs), targets)
                loss.backward()  # type: ignore[no-untyped-call]

                if update == 0:
                    first_cell_grad = sum(
                        float(parameter.grad.detach().abs().sum())
                        for parameter in model.cell.parameters()
                        if parameter.grad is not None
                    )
                    assert model.conditions.grad is not None
                    first_condition_grad = float(model.conditions.grad.detach().abs().sum())

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            with torch.no_grad():
                final_logits: Tensor = model(inputs)
                final_loss = float(F.binary_cross_entropy_with_logits(final_logits, targets))
                correct = int(torch.count_nonzero((final_logits >= 0) == targets.bool()))

            assert first_cell_grad > 0.0
            assert first_condition_grad > 0.0
            assert final_loss < initial_loss
            assert final_loss < 0.05
            assert correct == 16
            print(
                f"artificial polarity={polarity}: initial_bce={initial_loss:.6f}, "
                f"final_bce={final_loss:.6f}, corners={correct}/16, "
                f"first_grad(cell={first_cell_grad:.6g}, "
                f"condition={first_condition_grad:.6g})"
            )
    finally:
        torch.set_num_threads(previous_num_threads)

"""Reject malformed inputs on the STE-specific DAG evaluation path."""
import pytest
import torch

from cipher_experiments.models.ste_simon import STESimonPredictor


@pytest.mark.parametrize("x,error", [
    (torch.zeros(1, 63), ValueError),
    (torch.zeros(1, 64, dtype=torch.float64), TypeError),
    (torch.full((1, 64), .5), ValueError),
    (torch.full((1, 64), float("nan")), ValueError),
    (torch.empty(0, 64), ValueError),
])
def test_ste_rejects_bad_plaintexts(x: torch.Tensor, error: type[Exception]) -> None:
    model = STESimonPredictor()
    with pytest.raises(error):
        model(x)
    with pytest.raises(error):
        model.hard_logits(x)


def test_ste_rejects_device_mismatch() -> None:
    model = STESimonPredictor().to(device="meta")
    with pytest.raises(ValueError, match="same device"):
        model(torch.zeros(1, 64))

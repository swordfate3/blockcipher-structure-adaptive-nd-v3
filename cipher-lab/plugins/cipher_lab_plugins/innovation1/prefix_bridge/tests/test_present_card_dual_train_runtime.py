from __future__ import annotations

import numpy as np
import pytest
import torch

from cipher_lab.execution import Meter


def test_average_probabilities_and_both_gradients() -> None:
    from cipher_lab_plugins.innovation1.present_card_dual_train.probability import (
        average_logits,
        average_logits_torch,
    )

    a = torch.tensor([[-3.], [2.], [7.]], requires_grad=True)
    b = torch.tensor([[4.], [-1.], [-2.]], requires_grad=True)
    actual = average_logits_torch(a, b)
    p, q = (1 + (-a.detach()).exp()).reciprocal(), (1 + (-b.detach()).exp()).reciprocal()
    mean = (p + q) / 2
    expected = mean.log() - (-mean).log1p()
    np.testing.assert_allclose(actual.detach().numpy(), expected.numpy(), rtol=1e-5)
    assert actual.requires_grad
    # The guard replaces model ops. Check sensitivity without importing Dynamo's
    # forward-mode machinery, which expects those ops to be unpatched.
    for index, probability in enumerate((p, q)):
        head = [a.detach(), b.detach()]
        tail = head.copy()
        head[index] = head[index] + .002
        tail[index] = tail[index] - .002
        tangent = (average_logits_torch(*head) - average_logits_torch(*tail)) / .004
        expected_derivative = probability * (1 - probability) / (2 * mean * (1 - mean))
        np.testing.assert_allclose(tangent.numpy(), expected_derivative.numpy(),
                                   rtol=2e-3, atol=2e-5)
    np.testing.assert_allclose(average_logits(a.detach().numpy(), b.detach().numpy()),
                               actual.detach().numpy())


def test_saturated_decisions() -> None:
    from cipher_lab_plugins.innovation1.present_card_dual_train.probability import (
        average_logits,
        average_logits_torch,
    )

    a = np.array([100., 100., 100., 1000., -1000., 1000., 1000., -1000.], dtype=np.float32)
    b = np.array([-101., -99., -100., -1001., 1001., -1000., 1000., -1000.], dtype=np.float32)
    signs = [-1, 1, 0, -1, 1, 0, 1, -1]
    for result in (average_logits(a, b), average_logits_torch(torch.from_numpy(a),
                                                           torch.from_numpy(b)).numpy()):
        assert result.dtype == np.float32
        np.testing.assert_array_equal(np.sign(result), signs)
        np.testing.assert_array_equal(result[-2:], a[-2:])


@pytest.mark.parametrize('slot', ['swap-50', 'dual-train'])
@pytest.mark.parametrize('train', [False, True])
def test_predict_with_legacy_outer_reservation(monkeypatch: pytest.MonkeyPatch,
                                               bridge_meter: Meter, slot: str,
                                               train: bool) -> None:
    from cipher_lab_plugins.direct_block.fit import predict_batch
    from cipher_lab_plugins.innovation1.present_card_dual_train.runtime import FitRuntime, Runtime

    inputs = np.array([1, 3], dtype=np.float32).reshape(1, 1, 2, 1)
    labels = np.ones((1, 1), dtype=np.uint8)
    seen: list[float] = []

    def predict(self: Runtime, x: np.ndarray) -> np.ndarray:
        assert not torch.is_grad_enabled()
        seen.append(float(x[0, 0, 0, 0]))
        return x[:, 0, 0, :]

    monkeypatch.setattr(Runtime, 'start', lambda self: None)
    monkeypatch.setattr(Runtime, 'predict_inputs', predict)
    fit = FitRuntime(slot, {}, train, source={}, run_id='toy', origin=0, meter=bridge_meter)
    actual = predict_batch(fit, inputs, labels, bridge_meter)
    assert seen == ([1., 3.] if train else [1.])
    for status in ('attempted', 'completed'):
        assert bridge_meter.snapshot()[f'{status}_forward_rows'] == len(seen)
    if train:
        mean = float(np.mean(1 / (1 + np.exp(-np.array([1., 3.])))))
        expected = np.log(mean / (1 - mean))
        assert actual.item() == pytest.approx(expected)
    else:
        assert actual.item() == 1.

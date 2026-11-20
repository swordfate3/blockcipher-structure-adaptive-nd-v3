"""Keep the pair-local head regression for the layout adapter's reuse."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from plugin_isolation import forbidden as _forbidden


def test_pair_heads_do_not_pool_container_slots() -> None:
    from cipher_lab_plugins.innovation1.single_pair import models as m

    # Fake linear and genuinely nonlinear heads use different hidden values for every row.
    ids = np.arange(1., 9.)
    bits = np.broadcast_to(ids[:, None, None, None], (8, 1, 2, 64))
    packed = m.computation_slots(bits)
    public = object()
    calls: list[str] = []

    def fake_repr(x: Any, card: object, *, rounds: int) -> Any:
        assert x is packed and card is public and rounds == 8
        assert x.shape == (2, 4, 2, 64)
        np.testing.assert_array_equal(x[:, :, 0, 0].reshape(-1), ids)
        return SimpleNamespace(canonical=x[:, :, :1], channel=x[:, :, 1:])

    def stem(raw: Any, channel: Any) -> Any:
        calls.append('stem')
        np.testing.assert_array_equal(raw, channel)
        return raw

    def route(card: object) -> Any:
        assert card is public
        return SimpleNamespace(indices=np.array([2, 0]), weights=[.25, .75])

    model = SimpleNamespace(representation=fake_repr, router=route,
        spec=SimpleNamespace(channels=1),
        stems={'nibble': stem, 'local': stem, 'word': _forbidden}, suffix=lambda x: x + 1,
        readout=SimpleNamespace(forward=_forbidden, base=lambda x: 3 * x[:, :1] + 5,
                                residual=lambda x: x[:, :1] ** 2 + 7))
    result = m.shared_logits(model, packed, public, rounds=8)
    h = ids[:, None] + 1
    np.testing.assert_array_equal(result, (3*h+5) + (h**2+7))
    fake_original_m = 3 * h.reshape(2, 4).mean(1) + 5 + (h**2+7).reshape(2, 4).mean(1)
    np.testing.assert_array_equal(result.reshape(2, 4).mean(1), fake_original_m)
    assert calls == ['stem', 'stem']

    for bad in (bits[:, 0], bits[:3], np.zeros((8, 4, 2, 64))):
        with pytest.raises(ValueError, match='SINGLE_PAIR_SHAPE'):
            m.computation_slots(bad)

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest
import torch
from cipher_lab_plugins.innovation1.checkpoint import encode
from cipher_lab_plugins.innovation1.prefix_bridge import config as old_config
from cipher_lab_plugins.innovation1.prefix_bridge import runtime as old_runtime
from cipher_lab_plugins.innovation1.prefix_bridge import weights as old_weights
from torch import nn

_SOURCE = {'fake/source.py': 'a' * 64}


def test_constructor_same_parent_architecture_and_old_default_untouched() -> None:
    from cipher_lab_plugins.innovation1.s2_window import config, runtime, weights

    new = runtime.FitRuntime(config.SLOT, {}, False, source=_SOURCE, run_id='run')
    old = old_runtime.FitRuntime(config.SLOT, {}, False, source=_SOURCE, run_id='run')
    assert old._metadata({'epoch': 3, 'bce': .5}) == old_weights.identity(
        config.SLOT, _SOURCE, 'run', 3, .5)
    assert old._metadata({'epoch': 3, 'bce': .5}) == {
        'schema': 1, 'method': old_config.METHOD, 'slot': 'r7-cipher', 'round': 7, 'mask': 56,
        's2_boundary': old_config.S2_BOUNDARY, 'recipe': old_config.recipe(), 'source': _SOURCE,
        'run_id': 'run', 'epoch': 3, 'bce': .5,
    }
    meta = new._metadata({'epoch': 3, 'bce': .5})
    assert meta == weights.identity(config.SLOT, _SOURCE, 'run', 3, .5)
    assert meta['recipe'] == config.recipe() and meta['method'] != old_config.METHOD
    assert meta['round'] == 7 and meta['mask'] == 56
    assert meta['step'] == 3 * 256
    a = new._runtime._fresh_cpu()
    b = old._runtime._fresh_cpu()
    assert type(a) is type(b) and new.parameters == old.parameters == 1_095_285
    assert new.initial_sha == old.initial_sha
    assert new._runtime._mask == old._runtime._mask == 56
    x = np.zeros((1, 4, 6, 64), dtype=np.float32)
    x[:, :, 3:] = 1
    new._runtime._validate(x)
    x[:, :, 1] = 1
    with pytest.raises(ValueError, match='INPUT_MASK'):
        new._runtime._validate(x)


@pytest.mark.parametrize('wrong', ['', 'buffer', 'dtype', 'old', 'run', 'source', 'recipe', 'step',
                                   'slot', 'epoch', 'extra', 'metadata'])
def test_strict_new_checkpoint_rejects_another_domain_or_state(
    wrong: str, bridge_manual_only: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.innovation1.s2_window import config, runtime, weights

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer('layout', torch.tensor([1, 2]))
            self.weight = nn.Parameter(torch.tensor([0.]))

    engine = runtime.Runtime(config.SLOT)
    fresh = Toy()
    monkeypatch.setattr(engine, '_fresh_cpu', lambda: fresh)
    expected = weights.identity(config.SLOT, _SOURCE, 'fake', 1, .25)
    bound = dict(expected)
    if wrong == 'old':
        bound = old_weights.identity(config.SLOT, _SOURCE, 'fake', 1, .25)
    elif wrong in ('run', 'source', 'recipe', 'step', 'slot', 'epoch', 'extra', 'metadata'):
        key, value = {
            'run': ('run_id', 'other'), 'source': ('source', {'fake/b.py': 'b'*64}),
            'recipe': ('recipe', {}), 'step': ('step', 1), 'slot': ('slot', 'best'),
            'epoch': ('epoch', 31), 'extra': ('extra', True), 'metadata': ('round', 2),
        }[wrong]
        bound[key] = value
    state = {'layout': torch.tensor([1, 3 if wrong == 'buffer' else 2]),
             'weight': torch.tensor([3.25], dtype=torch.float64 if wrong == 'dtype' else
                                    torch.float32)}
    raw = encode(bound, state)
    guard, _ = bridge_manual_only
    loads: list[bool] = []

    def load(stream: Any, *, map_location: str, weights_only: bool) -> Any:
        assert type(stream) is io.BytesIO and stream.getvalue() == raw
        assert map_location == 'cpu' and weights_only
        loads.append(True)
        return {'metadata': bound, 'state': state}

    guard.setattr(torch, 'load', load)
    if wrong:
        with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
            engine.restore_bound(raw, expected)
        assert not engine.models and not engine.optimizers
    else:
        engine.restore_bound(raw, expected)
        assert engine.models[config.SLOT] is fresh and not engine.optimizers
        assert fresh.weight.detach().tolist() == [3.25]
    assert loads == [True]
    with pytest.raises(ValueError, match='CHECKPOINT_IDENTITY'):
        old_weights.require_identity(config.SLOT, expected)

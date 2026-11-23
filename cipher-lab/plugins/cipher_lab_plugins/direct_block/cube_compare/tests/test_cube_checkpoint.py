from __future__ import annotations

from io import BytesIO

import pytest
import torch
from torch import nn


class ManualState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.learned = nn.Parameter(torch.tensor([2., 3.]))
        self.register_buffer('indices', torch.tensor([4, 8]))
        self.register_buffer('card', torch.tensor([.25]))


def test_cube_checkpoint_model_only_roundtrip_and_size() -> None:
    from cipher_lab_plugins.direct_block.cube_compare import checkpoint

    first, second = ManualState(), ManualState()
    with torch.no_grad():
        first.learned.fill_(5.)
    raw = checkpoint.dump(first, identity='test-model')
    assert 0 < len(raw) <= 16 * 2**20
    obj = torch.load(BytesIO(raw), weights_only=True, map_location='cpu')
    assert set(obj) == {'identity', 'state'}
    assert set(obj['state']) == {'learned', 'indices', 'card'}
    checkpoint.restore(second, raw, identity='test-model')
    assert torch.equal(first.learned, second.learned)
    with pytest.raises(ValueError, match='size'):
        checkpoint.restore(second, b'0' * (16 * 2**20 + 1), identity='test-model')


@pytest.mark.parametrize('failure', ['identity', 'keys', 'shape', 'dtype', 'nan', 'fixed', 'card'])
def test_cube_checkpoint_rejects_corruption_before_loading(failure: str) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import checkpoint

    model = ManualState()
    state = dict(model.state_dict())
    state['learned'] = torch.tensor([8., 9.])
    identity = 'other' if failure == 'identity' else 'expected'
    if failure == 'keys':
        state.pop('indices')
    if failure == 'shape':
        state['learned'] = state['learned'][1:]
    if failure == 'dtype':
        state['learned'] = state['learned'].double()
    if failure == 'nan':
        state['learned'][0] = torch.nan
    if failure == 'fixed':
        state['indices'] = torch.tensor([8, 4])
    if failure == 'card':
        state['card'] = torch.tensor([.5])
    stream = BytesIO()
    torch.save({'identity': identity, 'state': state}, stream)
    with pytest.raises(ValueError, match='checkpoint'):
        checkpoint.restore(model, stream.getvalue(), identity='expected')
    assert model.learned.tolist() == [2., 3.]

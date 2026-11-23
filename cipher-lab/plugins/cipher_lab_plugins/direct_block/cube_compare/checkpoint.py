"""Size-, identity- and fixed-buffer-checked CPU model-only archives."""
from __future__ import annotations

from io import BytesIO

import torch
from torch import Tensor, nn

from .config import CHECKPOINT_BYTES


def _size(raw: bytes) -> None:
    if type(raw) is not bytes or not 0 < len(raw) <= CHECKPOINT_BYTES:
        raise ValueError('invalid checkpoint size')


def _validated(model: nn.Module, state: object) -> dict[str, Tensor]:
    expected = model.state_dict()
    fixed = dict(model.named_buffers())
    if type(state) is not dict or state.keys() != expected.keys():
        raise ValueError('invalid checkpoint keys')
    out: dict[str, Tensor] = {}
    for name, target in expected.items():
        value = state[name]
        if (type(value) is not Tensor or value.device.type != 'cpu' or value.layout != torch.strided
                or value.dtype != target.dtype or value.shape != target.shape
                or not bool(value.isfinite().all())):
            raise ValueError('invalid checkpoint tensor')
        if name in fixed and not torch.equal(value, fixed[name].detach().cpu()):
            raise ValueError('invalid checkpoint fixed buffer')
        out[name] = value
    return out


def dump(model: nn.Module, *, identity: str) -> bytes:
    state = {name: value.detach().cpu().contiguous().clone()
             for name, value in model.state_dict().items()}
    buffer = BytesIO()
    torch.save({'identity': identity, 'state': _validated(model, state)}, buffer)
    raw = buffer.getvalue()
    _size(raw)
    return raw


def restore(model: nn.Module, raw: bytes, *, identity: str) -> None:
    _size(raw)
    try:
        record = torch.load(BytesIO(raw), weights_only=True, map_location='cpu')
    except Exception:
        raise ValueError('invalid checkpoint archive') from None
    if (type(record) is not dict or record.keys() != {'identity', 'state'}
            or type(record['identity']) is not str or record['identity'] != identity):
        raise ValueError('invalid checkpoint identity')
    model.load_state_dict(_validated(model, record['state']), strict=True)

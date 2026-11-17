"""Owner checkpoint identity and CPU state codec; no trainer/data generation imports."""
from __future__ import annotations

import io
import json
import math
import zipfile
from collections.abc import Mapping
from typing import Any

import torch

from .config import Recipe, _decode, validate
from .contracts import MAX_CHECKPOINT, SLOTS

__all__ = ['SLOTS', 'MAX_CHECKPOINT', 'identity', 'encode', 'decode', 'canonical', 'preflight']


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(recipe: Recipe, source: Mapping[str, str], backend: str, slot: str,
             epoch: int, score: float) -> dict[str, Any]:
    if (slot not in SLOTS or backend not in ('demo', 'ciphers') or type(epoch) is not int
            or not 1 <= epoch <= recipe.training.epochs or type(score) not in (int, float)
            or not math.isfinite(score) or score < 0):
        raise ValueError('CHECKPOINT_IDENTITY')
    shared = slot == 'M-best'
    task = None if shared else next(t for t in recipe.tasks if f'D-{t.cipher}-best' == slot)
    return dict(schema=1, method=recipe.method, family='shared-M' if shared else 'dedicated',
                slot=slot, task=None if task is None else vars(task), recipe=recipe.to_dict(),
                source=dict(source), backend=backend, synthetic=backend == 'demo',
                runtime='manual-v1' if backend == 'demo' else 'torch-fp32-v1',
                model=recipe.to_dict()['model'] if shared else recipe.model.dedicated[
                    recipe.tasks.index(task)], init_seed=recipe.seeds.init,
                selection=recipe.selection.shared if shared else recipe.selection.dedicated,
                epoch=epoch, score=float(score))


def encode(metadata: dict[str, Any], state: Mapping[str, torch.Tensor]) -> bytes:
    stream = io.BytesIO()
    torch.save({'metadata': metadata,
                'state': {k: v.detach().cpu().clone() for k, v in state.items()}}, stream)
    raw = stream.getvalue()
    if len(raw) > MAX_CHECKPOINT:
        raise ValueError('CHECKPOINT_SIZE')
    return raw


def preflight(raw: bytes, maximum: int) -> None:
    if type(raw) is not bytes or not 0 < len(raw) <= maximum:
        raise ValueError('ARCHIVE_SIZE')
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            if (len(infos) > 4096 or len({i.filename for i in infos}) != len(infos)
                    or sum(i.file_size for i in infos) > maximum
                    or any(i.flag_bits & 1 or i.file_size < 0 for i in infos)):
                raise ValueError
    except Exception:
        raise ValueError('ARCHIVE_SCHEMA') from None


def decode(raw: bytes, expected: dict[str, Any],
           template: Mapping[str, torch.Tensor], *,
           fixed_buffers: Mapping[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    preflight(raw, MAX_CHECKPOINT)
    try:
        if type(expected) is not dict:
            raise ValueError
        recipe = _decode(expected['recipe'], Recipe(), 'recipe')
        validate(recipe)
        source = expected['source']
        if (type(source) is not dict or not source or any(
                type(k) is not str or type(v) is not str or len(v) != 64
                or any(c not in '0123456789abcdef' for c in v) for k, v in source.items())):
            raise ValueError
        rebuilt = identity(recipe, source, expected['backend'], expected['slot'],
                           expected['epoch'], expected['score'])
        if canonical(expected) != canonical(rebuilt):
            raise ValueError
        value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if (type(value) is not dict or set(value) != {'metadata', 'state'}
                or canonical(value['metadata']) != canonical(expected)
                or type(value['state']) is not dict or set(value['state']) != set(template)):
            raise ValueError
        # Only the production M has fixed public geometry. Dynamic BN buffers
        # and learned parameters are deliberately NOT compared with initial state.
        names = ({f'stems.{stem}.indices' for stem in ('local', 'word', 'nibble')}
                 if expected['backend'] == 'ciphers' and expected['family'] == 'shared-M'
                 else set())
        fixed = {} if fixed_buffers is None else fixed_buffers
        if set(fixed) != names or not names <= set(template):
            raise ValueError
        for name in names:
            reference = fixed[name]
            if (type(reference) is not torch.Tensor or reference.device.type != 'cpu'
                    or reference.dtype != template[name].dtype
                    or reference.shape != template[name].shape
                    or not torch.equal(reference, template[name])):
                raise ValueError
        state = value['state']
        for name, reference in template.items():
            tensor = state[name]
            if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu'
                    or tensor.layout != torch.strided or tensor.shape != reference.shape
                    or tensor.dtype != reference.dtype
                    or not bool(torch.isfinite(tensor).all())):
                raise ValueError
        if any(not torch.equal(state[name], fixed[name]) for name in names):
            raise ValueError
        return dict(state)
    except Exception:
        raise ValueError('CHECKPOINT_INVALID') from None

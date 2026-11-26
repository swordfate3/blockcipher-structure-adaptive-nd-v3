"""One-slot FP32 lifecycle over the existing innovation1 synchronized update."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from cipher_lab.execution import Meter

from ..checkpoint import encode
from ..config import Optimizer, Recipe, Task
from ..models import TorchRuntime
from . import config, weights
from .data import Floats, Labels
from .model import build, initial_sha256

_OPT_CONFIG = Recipe(optimizer=Optimizer(lr_max=config.LR, lr_min=config.LR,
                                       betas=config.BETAS, eps=config.EPS,
                                       weight_decay=config.WEIGHT_DECAY, clip_norm=config.CLIP))


def configure() -> dict[str, object]:
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return {'dtype': str(torch.get_default_dtype()), 'amp': False,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark}


class Runtime(TorchRuntime):
    _slot_input = staticmethod(config.slot_input)
    _require_identity = staticmethod(weights.require_identity)

    def __init__(self, slot: str) -> None:
        r, self._mask = self._slot_input(slot)
        super().__init__(_OPT_CONFIG, config.GPU_DEVICE, inference_only=True)
        self.slot = slot
        # For the S2 arm the Task required by the parent carries only the fixed
        # S2 boundary (2), not a supplied final round. No Task enters the model.
        self._task = Task('PRESENT', 2 if r is None else r, config.DIFFERENCE)
        self._initial = ''
        self._parameters = 0
        self._state = 'EMPTY'

    def _fresh_cpu(self) -> nn.Module:
        model = build(config.INIT_SEED)
        self._initial = initial_sha256(model)
        self._parameters = sum(t.numel() for t in model.parameters())
        return model

    def start(self) -> None:
        if self._state != 'EMPTY':
            raise ValueError('INVALID_RUNTIME_STATE')
        self.models[self.slot] = self._fresh_cpu().to(device=self.device, dtype=torch.float32)
        self.add_adam(self.slot)
        self._state = 'TRAINING'

    def _on_device(self, value: Floats | Labels) -> Tensor:
        return torch.from_numpy(value).to(device=self.device, dtype=torch.float32)

    def _validate(self, x: Floats) -> None:
        if x.dtype != np.float32 or x.ndim != 4 or x.shape[1:] != (4, 6, 64) or not len(x):
            raise ValueError('INPUT_SHAPE')
        if ((self._mask == 7 and np.any(x[:, :, 3:]))
                or (self._mask == 56 and np.any(x[:, :, :3]))):
            raise ValueError('INPUT_MASK')

    def _forward(self, slot: str, task: Task, x: Any) -> Tensor:
        if slot != self.slot or task != self._task or set(self.models) != {slot}:
            raise ValueError('INVALID_RUNTIME_DOMAIN')
        return cast(Tensor, self.models[slot](x))

    def batch_update(self, inputs: Floats, labels: Labels, meter: Meter) -> float:
        self._validate(inputs)
        if (self._state != 'TRAINING' or set(self.models) != {self.slot}
                or set(self.optimizers) != {self.slot}):
            raise ValueError('INVALID_RUNTIME_STATE')
        if labels.dtype != np.uint8 or labels.shape != (len(inputs), 1) or not (labels <= 1).all():
            raise ValueError('INVALID_LABELS')
        x, y = self._on_device(inputs), self._on_device(labels)
        with torch.autocast(device_type='cuda', enabled=False):
            return super().update(self.slot, self._task, x, y, config.LR, meter)

    def predict_inputs(self, inputs: Floats) -> Floats:
        self._validate(inputs)
        if self._state not in ('TRAINING', 'SCORING'):
            raise ValueError('INVALID_RUNTIME_STATE')
        model = self.models[self.slot]
        training = model.training
        model.eval()
        try:
            with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
                z = self._forward(self.slot, self._task, self._on_device(inputs))
                if z.dtype != torch.float32 or z.shape != (len(inputs), 1):
                    raise ValueError('INVALID_LOGITS')
                return cast(Floats, z.detach().cpu().numpy())
        finally:
            model.train(training)

    def dump_bound(self, metadata: dict[str, Any]) -> bytes:
        if self._state != 'TRAINING':
            raise ValueError('INVALID_RUNTIME_STATE')
        self._require_identity(self.slot, metadata)
        raw = encode(metadata, self.models[self.slot].state_dict())
        if not 0 < len(raw) <= config.CHECKPOINT_BYTES:
            raise ValueError('CHECKPOINT_SIZE')
        return raw

    def restore_bound(self, raw: bytes, metadata: dict[str, Any]) -> None:
        if self._state != 'EMPTY' or self.models or self.optimizers:
            raise ValueError('INVALID_RUNTIME_STATE')
        if type(raw) is not bytes or not 0 < len(raw) <= config.CHECKPOINT_BYTES:
            raise ValueError('CHECKPOINT_SIZE')
        self._require_identity(self.slot, metadata)
        model = self._fresh_cpu()
        fixed = tuple(name for name, _ in model.named_buffers())
        state = weights.decode(raw, metadata, model.state_dict(), fixed)
        model.load_state_dict(state, strict=True)
        self.models[self.slot] = model.eval()
        self._state = 'RESTORED_CPU'

    def activate(self) -> None:
        if self._state != 'RESTORED_CPU' or self.optimizers:
            raise ValueError('INVALID_RUNTIME_STATE')
        self.models[self.slot].to(device=self.device, dtype=torch.float32)
        self._state = 'SCORING'

    def release(self) -> None:
        # Drop live tensors/optimizer state. Keep the allocator's cache reusable
        # within this process; no CUDA probe or cache operation on a failure path.
        self._state = 'RELEASED'
        self.optimizers.clear()
        self.models.clear()


class FitRuntime:
    """Bind one public arm, epoch identity and CPU restore to the shared fit seam."""

    _runtime_type: type[Runtime] = Runtime
    _identity = staticmethod(weights.identity)

    def __init__(self, slot: str, arm: dict[str, object], train: bool, *,
                 source: dict[str, str], run_id: str) -> None:
        self._runtime = self._runtime_type(slot)
        self._slot, self._arm, self._source, self._run_id = slot, arm, source, run_id
        if train:
            self._runtime.start()

    @property
    def initial_sha(self) -> str:
        return self._runtime._initial

    @property
    def parameters(self) -> int:
        return self._runtime._parameters

    def _metadata(self, point: object) -> dict[str, Any]:
        if not isinstance(point, dict):
            raise ValueError('MISSING_DEV_SELECTION')
        return self._identity(self._slot, self._source, self._run_id,
                              point['epoch'], point['bce'])

    def dump(self) -> bytes:
        curve = self._arm.get('dev')
        if not isinstance(curve, list) or not curve:
            raise ValueError('MISSING_DEV_SELECTION')
        return self._runtime.dump_bound(self._metadata(curve[-1]))

    def restore(self, raw: bytes) -> None:
        self._runtime.restore_bound(raw, self._metadata(self._arm.get('selected')))
        if (self.initial_sha != self._arm.get('initial_sha256')
                or self.parameters != self._arm.get('parameters')):
            raise ValueError('MODEL_INITIAL_IDENTITY')

    def activate(self) -> None:
        self._runtime.activate()

    def release(self) -> None:
        self._runtime.release()

    def predict(self, inputs: Floats) -> Floats:
        return self._runtime.predict_inputs(inputs)

    def batch_update(self, inputs: Floats, labels: Labels, meter: Meter) -> float:
        return self._runtime.batch_update(inputs, labels, meter)

    def train_forward(self, inputs: Floats) -> Callable[[Labels], float | None]:
        raise ValueError('REQUIRES_BATCH_UPDATE')

    def update(self) -> None:
        raise ValueError('REQUIRES_BATCH_UPDATE')

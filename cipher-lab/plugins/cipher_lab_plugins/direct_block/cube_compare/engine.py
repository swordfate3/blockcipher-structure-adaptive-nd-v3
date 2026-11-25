"""FP32 device runtime for fixed cube models; never constructs labels."""
from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cipher_lab_plugins.direct_block.model import DirectBlockMLP

from . import checkpoint
from .config import (
    BETAS,
    CLIP,
    CPU_THREADS,
    DIRECT_METHOD,
    EPS,
    GPU_DEVICE,
    INIT_SEED,
    INTEROP_THREADS,
    LOW_METHOD,
    LOW_ROUNDS,
    LR,
    LSTM_METHOD,
    METHOD,
    WEIGHT_DECAY,
    WIDTH,
)
from .data import Floats, Labels
from .lstm import CubeLSTM
from .lstm64 import DirectLSTM64
from .model import CubeSharedM


def configure_runtime() -> dict[str, object]:
    """Call once before preflight or fit; report actual settings, not bitwise promises."""
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(INTEROP_THREADS)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return {
        'amp': False, 'dtype': str(torch.get_default_dtype()),
        'matmul_precision': torch.get_float32_matmul_precision(),
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cpu_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
    }


def make_engine(arm: Literal['mlp', 'sharedm']) -> TorchEngine:
    if arm not in ('mlp', 'sharedm'):
        raise ValueError('invalid cube arm')
    # CPU initialization with an isolated public seed, followed by transfer.
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.default_generator.manual_seed(INIT_SEED)
        model: nn.Module
        if arm == 'mlp':
            model = DirectBlockMLP(input_width=WIDTH, output_width=WIDTH, device='cpu')
        else:
            model = CubeSharedM()
    return TorchEngine(model, torch.device(GPU_DEVICE), identity=f'{METHOD}/{arm}')


def make_lstm_engine() -> TorchEngine:
    return _lstm_engine(f'{LSTM_METHOD}/lstm')


def make_low_engine(rounds: int) -> TorchEngine:
    if type(rounds) is not int or rounds not in LOW_ROUNDS:
        raise ValueError('invalid low round')
    return _lstm_engine(f'{LOW_METHOD}/r{rounds}')


def make_direct_engine(rounds: int) -> TorchEngine:
    if type(rounds) is not int or rounds not in LOW_ROUNDS:
        raise ValueError('invalid direct64 round')
    return _lstm_engine(f'{DIRECT_METHOD}/r{rounds}', direct=True)


def _lstm_engine(identity: str, *, direct: bool = False) -> TorchEngine:
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.default_generator.manual_seed(INIT_SEED)
        model = DirectLSTM64() if direct else CubeLSTM()
    if direct:
        return TorchEngine(model, torch.device(GPU_DEVICE), identity=identity, width=64)
    return TorchEngine(model, torch.device(GPU_DEVICE), identity=identity)


class TorchEngine:
    def __init__(self, model: nn.Module, device: torch.device, *, identity: str,
                 width: int = WIDTH) -> None:
        if type(width) is not int or width not in (64, WIDTH):
            raise ValueError('invalid cube engine width')
        self._width = width
        self._device = device
        self._model = model.to(device=device, dtype=torch.float32)
        self._identity = identity
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=LR, betas=BETAS,
                                           eps=EPS, weight_decay=WEIGHT_DECAY,
                                           foreach=False, fused=False)

    def _input(self, inputs: Floats) -> Tensor:
        if inputs.dtype != np.float32 or inputs.ndim != 2 or inputs.shape[1] != self._width:
            raise ValueError('invalid FP32 plaintext')
        return torch.from_numpy(inputs).to(self._device)

    def train_forward(self, inputs: Floats) -> Callable[[Labels], float]:
        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self._device.type, enabled=False):
            logits = cast(Tensor, self._model(self._input(inputs)))

        def backward(labels: Labels) -> float:
            if (labels.dtype != np.uint8 or labels.shape != tuple(logits.shape)
                    or (labels > 1).any()):
                raise ValueError('invalid training labels')
            targets = torch.from_numpy(labels).to(self._device, dtype=torch.float32)
            with torch.autocast(device_type=self._device.type, enabled=False):
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            value = float(loss.detach().item())
            if not bool(loss.isfinite()):
                raise ValueError('invalid training loss')
            loss.backward()  # type: ignore[no-untyped-call]  # PyTorch stub
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), CLIP, error_if_nonfinite=True)
            return value

        return backward

    def update(self) -> None:
        self._optimizer.step()
        # Do not credit completion before any asynchronous device failure surfaces.
        if self._device.type == 'cuda':
            torch.cuda.synchronize(self._device)

    def predict(self, inputs: Floats) -> Floats:
        self._model.eval()
        with torch.no_grad(), torch.autocast(device_type=self._device.type, enabled=False):
            logits = cast(Tensor, self._model(self._input(inputs)))
            return cast(Floats, logits.cpu().numpy())

    def dump(self) -> bytes:
        return checkpoint.dump(self._model, identity=self._identity)

    def restore(self, raw: bytes) -> None:
        checkpoint.restore(self._model, raw, identity=self._identity)
        if self._device.type == 'cuda':
            torch.cuda.synchronize(self._device)

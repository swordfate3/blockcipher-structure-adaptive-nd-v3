"""The fixed two-logit MLP and public-seeded FP32 engine composition."""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from . import config

if TYPE_CHECKING:
    from ..cube_compare.engine import TorchEngine


class SplitMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(32, 128, device='cpu', dtype=torch.float32), nn.ReLU(),
            nn.Linear(128, 128, device='cpu', dtype=torch.float32), nn.ReLU(),
            nn.Linear(128, 2, device='cpu', dtype=torch.float32),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if (not isinstance(inputs, Tensor) or inputs.layout != torch.strided
                or inputs.ndim != 2 or not inputs.shape[0] or inputs.shape[1] != 32):
            raise ValueError('expected nonempty plaintext shape [N,32]')
        p = next(self.parameters(), None)
        if inputs.dtype != torch.float32 or (p is not None and inputs.device != p.device):
            raise ValueError('expected same-device FP32 plaintext')
        if not bool(((inputs == 0) | (inputs == 1)).all()):
            raise ValueError('expected binary plaintext')
        logits = cast(Tensor, self.layers(inputs))
        if (logits.shape != (len(inputs), 2) or logits.dtype != inputs.dtype
                or logits.device != inputs.device or not bool(logits.isfinite().all())):
            raise ValueError('invalid two-bit logits')
        return logits


def make_engine(rounds: int, chunk: int) -> TorchEngine:
    from ..cube_compare import engine
    from ..cube_compare.config import GPU_DEVICE

    if (type(rounds) is not int or rounds not in config.ROUNDS
            or type(chunk) is not int or chunk not in config.CHUNKS):
        raise ValueError('invalid split8 model identity')
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(config.INIT_SEED + chunk)
        model = SplitMLP()
    return engine.TorchEngine(model, torch.device(GPU_DEVICE),
                              identity=f'{config.METHOD}/r{rounds}/b{chunk}', width=32)

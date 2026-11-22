"""Production M+4D lifecycle, isolated initialization and FP32 Adam updates.

No cipher, recipe file reads, report logic or dedicated feature derivation.
Only explicit ciphers assembly may instantiate this numerical runtime.
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from cipher_lab.models.dedicated import FeistelSEExpert, PresentExpert, SkinnyExpert
from cipher_lab.models.shared_m import SharedM
from cipher_lab.models.shared_m.spec import SharedMSpec
from cipher_lab.representations.tasks import public_features

from .batch import Batch
from .checkpoint import SLOTS, decode, encode
from .config import Recipe, Task


def construct(recipe: Recipe, slot: str) -> nn.Module:
    # CPU-only initialization must not alter global CPU/CUDA RNG streams.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(recipe.seeds.init)
        if slot == 'M-best':
            m = recipe.model
            return SharedM(SharedMSpec(m.stem_width, m.channels, m.norm_groups,
                                      m.suffix_kernels, m.head_hidden, m.router_hidden))
        cipher = slot.removeprefix('D-').removesuffix('-best')
        if cipher in ('SIMECK', 'SIMON'):
            return FeistelSEExpert({'SIMECK': 'simeck64', 'SIMON': 'simon64'}[cipher])
        if cipher == 'PRESENT':
            return PresentExpert()
        if cipher == 'SKINNY':
            return SkinnyExpert(rounds=next(t.rounds for t in recipe.tasks if t.cipher == cipher))
        raise ValueError('MODEL_SLOT')


class TorchRuntime:
    backend = 'ciphers'

    def __init__(self, recipe: Recipe, device: str, *, inference_only: bool = False) -> None:
        if device != 'cuda:0':
            raise ValueError('DEVICE_UNSUPPORTED')
        self.recipe, self.device = recipe, torch.device(device)
        self.models: dict[str, nn.Module] = {}
        self.optimizers: dict[str, Any] = {}
        if not inference_only:
            for slot in SLOTS:
                model = construct(recipe, slot).to(device=self.device, dtype=torch.float32)
                self.models[slot] = model
                self.add_adam(slot)

    def add_adam(self, slot: str) -> None:
        o = self.recipe.optimizer
        self.optimizers[slot] = torch.optim.Adam(self.models[slot].parameters(), lr=o.lr_max,
                    betas=o.betas, eps=o.eps, weight_decay=o.weight_decay, amsgrad=o.amsgrad)

    def metadata(self) -> dict[str, Any]:
        return {slot: dict(parameters=sum(p.numel() for p in model.parameters()),
                           buffers=sum(b.numel() for b in model.buffers()),
                           state_bytes=sum(t.numel() * t.element_size()
                                           for t in model.state_dict().values()),
                           manual_state_only=False) for slot, model in self.models.items()}

    def prepare(self, batch: Batch) -> tuple[Any, Any]:
        x, y = batch.tensors()
        return x.to(self.device), y.to(self.device)

    def _forward(self, slot: str, task: Task, x: Any) -> Any:
        model = self.models[slot]
        if slot == 'M-best':
            public = torch.tensor(public_features(task.cipher), device=x.device, dtype=x.dtype)
            return model(x, public, rounds=task.rounds)
        return model(x)

    def _loss(self, logits: Tensor, y: Tensor) -> Tensor:
        return nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='mean')

    def update(self, slot: str, task: Task, x: Any, y: Any, lr: float,
               resources: Any) -> float:
        model, optimizer = self.models[slot], self.optimizers[slot]
        model.train()
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        try:
            resources.reserve('forward_rows', int(y.shape[0]))
            logits = self._forward(slot, task, x)
            if logits.shape != y.shape or not bool(torch.isfinite(logits).all()):
                raise ValueError('FORWARD_INVALID')
            resources.complete('forward_rows', int(y.shape[0]))
            loss = self._loss(logits, y)
            if not bool(torch.isfinite(loss)):
                raise ValueError('LOSS_INVALID')
            resources.check()
            cast(Any, loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.recipe.optimizer.clip_norm,
                                     error_if_nonfinite=True)
            resources.check()
            optimizer.step()
            # Synchronize so asynchronous numerical/device failures cannot be completed.
            torch.cuda.synchronize(self.device)
            return float(loss.detach().cpu())
        finally:
            optimizer.zero_grad(set_to_none=True)

    def predict(self, slot: str, task: Task, x: Any) -> Any:
        model = self.models[slot]
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                value = self._forward(slot, task, x)
                return value.detach().cpu().numpy().astype(np.float64)
        finally:
            model.train(was_training)

    def checkpoint(self, slot: str, metadata: dict[str, Any]) -> bytes:
        return encode(metadata, self.models[slot].state_dict())

    def release(self) -> None:
        self.optimizers.clear()
        self.models.clear()
        torch.cuda.empty_cache()

    def restore(self, slot: str, raw: bytes, metadata: dict[str, Any]) -> None:
        model = construct(self.recipe, slot)
        template = model.state_dict()
        fixed = ({f'stems.{stem}.indices': template[f'stems.{stem}.indices']
                  for stem in ('local', 'word', 'nibble')} if slot == 'M-best' else {})
        state = decode(raw, metadata, template, fixed_buffers=fixed)
        model.load_state_dict(state, strict=True)
        self.models[slot] = model.to(self.device, dtype=torch.float32).eval()

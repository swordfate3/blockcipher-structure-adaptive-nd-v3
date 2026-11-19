"""Reuse pair-local components of exactly the two restored production models."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cipher_lab.models.shared_m.spec import STEM_NAMES

from ..config import Recipe, Task


def computation_slots(bits: Any) -> Any:
    """Shape-only container, in row order; no repetition or group semantics."""
    if (bits.ndim != 4 or bits.shape[1:] != (1, 2, 64)
            or not bits.shape[0] or bits.shape[0] % 4):
        raise ValueError('SINGLE_PAIR_SHAPE')
    return bits.reshape(-1, 4, 2, 64)


def shared_logits(model: Any, slots: Any, public: Any, *, rounds: int) -> Any:
    represented = model.representation(slots, public, rounds=rounds)
    route = model.router(public)
    # Restored modules and inputs are FP32. Stems and suffix use per-row GN.
    a, b = (model.stems[STEM_NAMES[index]](represented.canonical, represented.channel)
            for index in route.indices.tolist())
    mixed = route.weights[0] * a + route.weights[1] * b
    channels = model.spec.channels
    hidden = model.suffix(mixed.reshape(-1, channels, 64)).reshape(-1, channels * 64)
    # Affine base commutes with the old mean; nonlinear residual stays per pair.
    return model.readout.base(hidden) + model.readout.residual(hidden)


def dedicated_logits(model: Any, slots: Any, layout: Callable[..., Any]) -> Any:
    # With BN in eval mode, both this layout and the encoder are pair-local.
    hidden = model.encoder(layout(slots, rounds=model.rounds))
    return model.head(hidden)


class SinglePairRuntime:
    def __init__(self, runtime: Any, task: Task) -> None:
        import torch

        from cipher_lab.representations.tasks import public_features

        self.runtime = runtime
        self.task = task
        self.public = torch.tensor(public_features(task.cipher), device=runtime.device,
                                   dtype=torch.float32)

    def prepare(self, bits: Any) -> Any:
        import torch

        with torch.inference_mode():
            return torch.from_numpy(bits).to(self.runtime.device)

    def predict(self, slot: str, bits: Any) -> Any:
        import torch

        from cipher_lab.representations.dedicated import author_pair_layout

        with torch.inference_mode():
            packed = computation_slots(bits)
            model = self.runtime.models[slot]
            if slot == 'M-best':
                logits = shared_logits(model, packed, self.public, rounds=self.task.rounds)
            elif slot == 'D-SKINNY-best':
                logits = dedicated_logits(model, packed, author_pair_layout)
            else:
                raise ValueError('MODEL_SLOT')
            # Blocking transfer surfaces asynchronous forward errors before completion.
            return logits.detach().cpu().numpy()


def restore(recipe: Recipe, task: Task, raw: dict[str, bytes],
            expected: dict[str, dict[str, Any]], check: Callable[[], None]) -> SinglePairRuntime:
    from ..models import TorchRuntime

    runtime = TorchRuntime(recipe, 'cuda:0', inference_only=True)
    # Use original strict weights-only decoder, including its fixed M buffers.
    for slot, blob in raw.items():
        check()
        runtime.restore(slot, blob, expected[slot])
        runtime.models[slot].requires_grad_(False)
        check()
    return SinglePairRuntime(runtime, task)

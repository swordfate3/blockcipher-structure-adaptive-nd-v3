"""Pair-local shared components used by the layout adapter."""
from __future__ import annotations

from typing import Any

from cipher_lab.models.shared_m.spec import STEM_NAMES


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

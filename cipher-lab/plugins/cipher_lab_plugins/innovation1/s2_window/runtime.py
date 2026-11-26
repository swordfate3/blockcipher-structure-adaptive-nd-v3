"""New checkpoint domain over the existing C7-only FP32/Adam lifecycle."""
from __future__ import annotations

from ..prefix_bridge import runtime as parent
from . import config, weights


class Runtime(parent.Runtime):
    _require_identity = staticmethod(weights.require_identity)

    def __init__(self, slot: str) -> None:
        if slot != config.SLOT:
            raise ValueError('INVALID_SLOT')
        super().__init__(slot)


class FitRuntime(parent.FitRuntime):
    _runtime_type = Runtime
    _identity = staticmethod(weights.identity)

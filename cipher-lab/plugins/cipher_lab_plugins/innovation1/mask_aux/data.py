"""Only the canonical C7 panel; the parent owns the private split closure."""
from __future__ import annotations

from cipher_lab.execution import Meter

from ..prefix_bridge import data as bridge
from . import config

Panels = bridge.Panels


def make_panels(meter: Meter, draw: bridge.Draw, encrypt: bridge.Encrypt,
                *, sample: bridge.Sample = bridge._sample) -> Panels:
    return bridge.make_panels(meter, draw, encrypt, sample=sample, rounds=config.ROUNDS)


def production_panels(meter: Meter) -> Panels:
    from secrets import randbits

    from cipher_lab.ciphers.present_prefix import encrypt_prefixes

    return make_panels(meter, randbits, encrypt_prefixes)

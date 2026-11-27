"""Fixed public MSB0 position gathers."""
from hashlib import sha256

from ..present_card_fresh.layout import Q

BASE = 'swap-50'
SHUFFLED = 'shuffled-fusion'
P_LAYER = 'p-layer-fusion'
SLOTS = (BASE, SHUFFLED, P_LAYER)


def _tables() -> tuple[tuple[int, ...], tuple[int, ...]]:
    positions = range(64)
    q = Q[:64]
    if sorted(q) != list(positions):
        raise ValueError('FUSION_WIRING')
    gather = tuple(q.index(dest) for dest in positions)
    renaming = sorted(positions, key=lambda i: sha256(
        b'present-p-layer-fusion-shuffle-v1/' + bytes([i])).digest())
    shuffled = [0] * 64
    for dest, source in enumerate(gather):
        shuffled[renaming[dest]] = renaming[source]
    if sorted(shuffled) != list(positions) or tuple(shuffled) == gather:
        raise ValueError('FUSION_WIRING')
    return gather, tuple(shuffled)


_GATHER, _SHUFFLED = _tables()
_WIRING = {BASE: (), SHUFFLED: _SHUFFLED, P_LAYER: _GATHER}


def gather_for(slot: str) -> tuple[int, ...]:
    try:
        return _WIRING[slot]
    except KeyError:
        raise ValueError('INVALID_SLOT') from None


def wiring_sha256(slot: str) -> str:
    return sha256(bytes(gather_for(slot))).hexdigest()

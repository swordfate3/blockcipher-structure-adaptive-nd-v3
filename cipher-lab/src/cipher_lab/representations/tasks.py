"""Public13 structural constants, with the supported families owned by Task1.

Card order: ARX, Feistel, SPN, block/128, key/128, word/32, branches/16,
Sbox/8, three rotations/32, permutation, MixColumns. Never task rounds/diff.
"""
from __future__ import annotations

import torch
from torch import Tensor

from cipher_lab.training_config import TASK_FAMILIES

# word, branches, ARX, Feistel, SPN, Sbox, rotations, permutation, MixColumns.
# This table owns only structure, NOT a second task declaration/rounds list.
_STRUCTURE = {
    "SIMECK": (32, 2, 0, 1, 0, 0, (0, 5, 1), 0, 0),
    "PRESENT": (4, 16, 0, 0, 1, 4, (0, 0, 0), 1, 0),
    "SIMON": (32, 2, 0, 1, 0, 0, (1, 8, 2), 0, 0),
    "SKINNY": (4, 16, 0, 0, 1, 4, (0, 0, 0), 0, 1),
}


def public_features(cipher: str) -> tuple[float, ...]:
    """Exactly one reviewed structural card; accepts no sample/label metadata."""
    family = next((family for family in TASK_FAMILIES if family.cipher == cipher), None)
    if family is None:
        raise ValueError("unsupported public family")
    word, branches, arx, feistel, spn, sbox, rotations, permutation, mix = _STRUCTURE[cipher]
    return (float(arx), float(feistel), float(spn), family.block_bits / 128,
            family.key_bits / 128, word / 32, branches / 16, sbox / 8,
            *(rotation / 32 for rotation in rotations), float(permutation), float(mix))


def validate_public(public: Tensor) -> None:
    """Reject batch cards, nonfinite values and even near-cards (no rounding)."""
    if (not isinstance(public, Tensor) or public.layout != torch.strided
            or public.shape != (13,) or not public.is_floating_point()
            or not bool(torch.isfinite(public).all())):
        raise ValueError("one finite floating public structure vector [13] required")
    cards = torch.tensor([public_features(f.cipher) for f in TASK_FAMILIES],
                         dtype=torch.float64, device=public.device)
    if not bool((cards == public.to(torch.float64)).all(dim=1).any()):
        raise ValueError("unreviewed public card")


def validate_rounds(public: Tensor, rounds: int) -> None:
    """Public rounds are representation context, never a fourteenth router input."""
    validate_public(public)
    family = next(f for f in TASK_FAMILIES if bool(torch.equal(
        public.double(), torch.tensor(public_features(f.cipher), dtype=torch.float64,
                                      device=public.device))))
    if type(rounds) is not int or not 1 <= rounds <= family.max_rounds:
        raise ValueError("unsupported public rounds for family")

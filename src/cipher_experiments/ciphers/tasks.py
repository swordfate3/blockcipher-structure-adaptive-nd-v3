# Reviewed extraction: src/blockcipher_nd_v3/expansion/config.py
# Source commit: d7a8698f4c474e1899f848b5de7a5abb570a13ea
# Source SHA-256: 7004b3d05e5810ffdb8445bea7d0cf376df5e2c0c9b33c0930229ac05038b965
"""Four explicit tasks and ciphertext-independent public structure features."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# block, key, full rounds, word, branches, ARX, Feistel, SPN, Sbox,
# three rotation amounts, permutation, mix-column
_PARAMETERS = {
    "simeck64": (64, 128, 44, 32, 2, 0, 1, 0, 0, 0, 5, 1, 0, 0),
    "present80": (64, 80, 31, 4, 16, 0, 0, 1, 4, 0, 0, 0, 1, 0),
    "simon64": (64, 128, 44, 32, 2, 0, 1, 0, 0, 1, 8, 2, 0, 0),
    "skinny64": (64, 64, 32, 4, 16, 0, 0, 1, 4, 0, 0, 0, 0, 1),
}


def public_features(cipher: str) -> tuple[float, ...]:
    """Return the reviewed 13 public features; no sample information is accepted."""
    p = _PARAMETERS[cipher]
    return (
        float(p[5]),
        float(p[6]),
        float(p[7]),
        p[0] / 128,
        p[1] / 128,
        p[3] / 32,
        p[4] / 16,
        p[8] / 8,
        p[9] / 32,
        p[10] / 32,
        p[11] / 32,
        float(p[12]),
        float(p[13]),
    )


def validate_public(public: torch.Tensor) -> None:
    """Validate the reviewed structure cards without any ciphertext adapter dependency."""
    if public.shape != (13,) or not bool(torch.isfinite(public).all()):
        raise ValueError("one finite public_structure_features vector [13] required")
    # Do not round to the caller dtype: half precision must not admit near-cards.
    cards = torch.tensor(
        [public_features(task) for task in _PARAMETERS], dtype=torch.float64, device=public.device
    )
    if not bool((cards == public.to(torch.float64)).all(dim=1).any()):
        raise ValueError("unreviewed public card outside the four-task 64-bit contract")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    rounds: int
    input_difference: int

    def __post_init__(self) -> None:
        if self.name not in _PARAMETERS:
            raise ValueError("unknown cipher")
        if type(self.rounds) is not int or not 1 <= self.rounds <= _PARAMETERS[self.name][2]:
            raise ValueError("invalid round count")
        if (
            type(self.input_difference) is not int
            or not 0 < self.input_difference < 1 << self.block_bits
        ):
            raise ValueError("invalid nonzero input difference")

    @property
    def cipher(self) -> str:
        return self.name

    @property
    def block_bits(self) -> int:
        return _PARAMETERS[self.name][0]

    @property
    def key_bits(self) -> int:
        return _PARAMETERS[self.name][1]


DEFAULT_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("simeck64", 16, 0x40),
    TaskSpec("present80", 7, 0xD00000),
    TaskSpec("simon64", 13, 0x40),
    TaskSpec("skinny64", 8, 0x2000),
)

# Name lookup remains convenient for data callers; the ordered tuple is the
# public model-facing task list.
TASKS: dict[str, TaskSpec] = {task.name: task for task in DEFAULT_TASKS}

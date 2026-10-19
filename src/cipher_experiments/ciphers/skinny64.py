# Reviewed extraction: src/blockcipher_nd_v3/ciphers/skinny64.py
# Source commit: d7a8698f4c474e1899f848b5de7a5abb570a13ea
# Source SHA-256: c6d517a1a2872ea82716cddac6ebf37ddde8d2ce692e76868163ee6bf462cd64

"""SKINNY-64/64, adapted from the reviewed v2 spn/skinny.py.

Source commit: 979b734e35da2704390db6f375dfffdafc84c45f.
Upstream byte hashes are recorded in the project extraction manifest.
The integer state is a row-major, most-significant-nibble-first 4x4 matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

SKINNY64_SBOX = (0xC, 0x6, 0x9, 0x0, 0x1, 0xA, 0x2, 0xB, 0x3, 0x8, 0x5, 0xD, 0x4, 0xE, 0x7, 0xF)
SKINNY64_TK_PERMUTATION = (9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7)
SKINNY64_SHIFT_ROWS = (0, 1, 2, 3, 7, 4, 5, 6, 10, 11, 8, 9, 13, 14, 15, 12)


@dataclass(frozen=True)
class Skinny64_64:
    rounds: int = 32
    key: int = 0

    block_bits: ClassVar[int] = 64
    key_bits: ClassVar[int] = 64

    def __post_init__(self) -> None:
        if not isinstance(self.rounds, int):
            raise TypeError("SKINNY-64/64 rounds must be an integer")
        # v2 deliberately allows the zero-round identity transform.
        if not 0 <= self.rounds <= 32:
            raise ValueError("SKINNY-64/64 rounds must be between 0 and 32")
        if not isinstance(self.key, int):
            raise TypeError("SKINNY-64/64 key must be an integer")
        if not 0 <= self.key < (1 << 64):
            raise ValueError("SKINNY-64/64 key must fit in 64 bits")

    def encrypt(self, plaintext: int) -> int:
        if not isinstance(plaintext, int):
            raise TypeError("SKINNY-64/64 plaintext must be an integer")
        if not 0 <= plaintext < (1 << 64):
            raise ValueError("SKINNY-64/64 plaintext must fit in 64 bits")
        state = [(plaintext >> shift) & 0xF for shift in range(60, -1, -4)]
        tweakey = [(self.key >> shift) & 0xF for shift in range(60, -1, -4)]
        constant = 1
        for _ in range(self.rounds):
            state = [SKINNY64_SBOX[cell] for cell in state]
            state[0] ^= constant & 0xF
            state[4] ^= (constant >> 4) & 3
            state[8] ^= 2
            for index in range(8):
                state[index] ^= tweakey[index]
            shifted = [state[index] for index in SKINNY64_SHIFT_ROWS]
            for column in range(4):
                s0, s1, s2, s3 = (shifted[column + 4 * row] for row in range(4))
                state[column] = s0 ^ s2 ^ s3
                state[4 + column] = s0
                state[8 + column] = s1 ^ s2
                state[12 + column] = s0 ^ s2
            tweakey = [tweakey[index] for index in SKINNY64_TK_PERMUTATION]
            constant = ((constant << 1) & 0x3F) | (((constant >> 5) ^ (constant >> 4) ^ 1) & 1)
        output = 0
        for cell in state:
            output = (output << 4) | cell
        return output

# Reviewed extraction: src/blockcipher_nd_v3/ciphers/simeck64.py
# Source commit: d7a8698f4c474e1899f848b5de7a5abb570a13ea
# Source SHA-256: b49596e517722d18ce9b814da04b60dc93d20d88d9f2fea00141bff3f9e11b80

"""SIMECK64/128, adapted from reviewed v2 feistel/simeck.py.

Source commit: 979b734e35da2704390db6f375dfffdafc84c45f.
The 63-bit Z1 stream belongs to the variant, never to the requested round count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .common import rol

SIMECK_Z1 = tuple(
    int(bit) for bit in "111111000001000011000101001111010001110010010110111011001101010"
)
_MASK32 = (1 << 32) - 1


@dataclass(frozen=True)
class Simeck64_128:
    rounds: int = 44
    key: int = 0

    block_bits: ClassVar[int] = 64
    key_bits: ClassVar[int] = 128

    def __post_init__(self) -> None:
        if not isinstance(self.rounds, int):
            raise TypeError("SIMECK64/128 rounds must be an integer")
        if not 1 <= self.rounds <= 44:
            raise ValueError("SIMECK64/128 rounds must be between 1 and 44")
        if not isinstance(self.key, int):
            raise TypeError("SIMECK64/128 key must be an integer")
        if not 0 <= self.key < (1 << 128):
            raise ValueError("SIMECK64/128 key must fit in 128 bits")

    def _round_keys(self) -> list[int]:
        words = [(self.key >> (32 * index)) & _MASK32 for index in range(4)]
        keys = [words[0]]
        for index in range(self.rounds - 1):
            feedback = (rol(words[1], 5, 32) & words[1]) ^ rol(words[1], 1, 32)
            next_word = words[0] ^ feedback ^ (_MASK32 ^ 3) ^ SIMECK_Z1[index % 63]
            words = [words[1], words[2], words[3], next_word & _MASK32]
            keys.append(words[0])
        return keys

    def encrypt(self, plaintext: int) -> int:
        if not isinstance(plaintext, int):
            raise TypeError("SIMECK64/128 plaintext must be an integer")
        if not 0 <= plaintext < (1 << 64):
            raise ValueError("SIMECK64/128 plaintext must fit in 64 bits")
        left, right = plaintext >> 32, plaintext & _MASK32
        for key in self._round_keys():
            feedback = (rol(left, 5, 32) & left) ^ rol(left, 1, 32)
            left, right = (right ^ feedback ^ key) & _MASK32, left
        return (left << 32) | right

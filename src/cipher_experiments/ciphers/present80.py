# Reviewed extraction: src/blockcipher_nd_v3/ciphers/present80.py
# Source commit: d7a8698f4c474e1899f848b5de7a5abb570a13ea
# Source SHA-256: c4162ab6152c58dbfc9993532a9c4f29c1d23e884beacbf8637099e43719bfca

from __future__ import annotations

from dataclasses import dataclass

from .common import rol

PRESENT_SBOX = (
    0xC,
    0x5,
    0x6,
    0xB,
    0x9,
    0x0,
    0xA,
    0xD,
    0x3,
    0xE,
    0xF,
    0x8,
    0x4,
    0x7,
    0x1,
    0x2,
)


@dataclass(frozen=True)
class Present80:
    rounds: int
    key: int

    block_bits: int = 64
    key_bits: int = 80

    def __post_init__(self) -> None:
        if not 1 <= self.rounds <= 31:
            raise ValueError("PRESENT-80 rounds must be between 1 and 31")
        if not 0 <= self.key < (1 << self.key_bits):
            raise ValueError("PRESENT-80 key must fit in 80 bits")

    def encrypt(self, plaintext: int) -> int:
        if not 0 <= plaintext < (1 << self.block_bits):
            raise ValueError("PRESENT-80 plaintext must fit in 64 bits")
        state = plaintext
        key_register = self.key
        for round_counter in range(1, self.rounds + 1):
            state ^= key_register >> 16
            state = self._sbox_layer(state)
            state = self._permutation_layer(state)
            key_register = self._update_key(key_register, round_counter)
        return state ^ (key_register >> 16)

    @staticmethod
    def _sbox_layer(state: int) -> int:
        output = 0
        for nibble_index in range(16):
            nibble = (state >> (4 * nibble_index)) & 0xF
            output |= PRESENT_SBOX[nibble] << (4 * nibble_index)
        return output

    @staticmethod
    def _permutation_layer(state: int) -> int:
        output = ((state >> 63) & 1) << 63
        for bit_index in range(63):
            output |= ((state >> bit_index) & 1) << ((16 * bit_index) % 63)
        return output

    @staticmethod
    def _update_key(key_register: int, round_counter: int) -> int:
        key_register = rol(key_register, 61, 80)
        top_nibble = (key_register >> 76) & 0xF
        key_register &= ~(0xF << 76)
        key_register |= PRESENT_SBOX[top_nibble] << 76
        key_register ^= (round_counter & 0x1F) << 15
        return key_register & ((1 << 80) - 1)

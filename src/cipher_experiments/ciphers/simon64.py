# Reviewed extraction: src/blockcipher_nd_v3/ciphers/simon64.py
# Source commit: d7a8698f4c474e1899f848b5de7a5abb570a13ea
# Source SHA-256: f85ddc48142671cf7bb6a5e360d6a2dcb0f4cb7cea82c394f706846bf5926b76

from __future__ import annotations

from dataclasses import dataclass

from .common import rol, ror

SIMON_Z3 = "11011011101011000110010111100000010010001010011100110100001111"


@dataclass(frozen=True)
class Simon64_128:
    rounds: int
    key: int

    block_bits: int = 64
    key_bits: int = 128

    def __post_init__(self) -> None:
        if not 1 <= self.rounds <= 44:
            raise ValueError("SIMON64/128 rounds must be between 1 and 44")
        if not 0 <= self.key < (1 << self.key_bits):
            raise ValueError("SIMON64/128 key must fit in 128 bits")

    def encrypt(self, plaintext: int) -> int:
        if not 0 <= plaintext < (1 << self.block_bits):
            raise ValueError("SIMON64/128 plaintext must fit in 64 bits")
        mask = 0xFFFFFFFF
        x = (plaintext >> 32) & mask
        y = plaintext & mask
        for round_key in self._round_keys():
            x, y = (y ^ (rol(x, 1, 32) & rol(x, 8, 32)) ^ rol(x, 2, 32) ^ round_key) & mask, x
        return (x << 32) | y

    def _round_keys(self) -> list[int]:
        mask = 0xFFFFFFFF
        c = mask ^ 0x3
        round_keys = [(self.key >> (32 * index)) & mask for index in range(4)]
        for index in range(4, self.rounds):
            temporary = ror(round_keys[index - 1], 3, 32) ^ round_keys[index - 3]
            temporary ^= ror(temporary, 1, 32)
            z_bit = int(SIMON_Z3[(index - 4) % 62])
            round_keys.append((c ^ z_bit ^ round_keys[index - 4] ^ temporary) & mask)
        return round_keys[: self.rounds]

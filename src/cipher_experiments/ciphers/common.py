# Reviewed extraction: src/blockcipher_nd_v3/ciphers/common.py
# Source commit: d7a8698f4c474e1899f848b5de7a5abb570a13ea
# Source SHA-256: 1110e61feeab3d5465d7119fcb47ac0ffc4f21f38b91318f5c22032713fe1d2a

from __future__ import annotations


def rol(value: int, shift: int, width: int) -> int:
    if width <= 0:
        raise ValueError("width must be positive")
    mask = (1 << width) - 1
    normalized = shift % width
    return ((value << normalized) & mask) | ((value & mask) >> (width - normalized))


def ror(value: int, shift: int, width: int) -> int:
    if width <= 0:
        raise ValueError("width must be positive")
    mask = (1 << width) - 1
    normalized = shift % width
    return ((value & mask) >> normalized) | ((value << (width - normalized)) & mask)

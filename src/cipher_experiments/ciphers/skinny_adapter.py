# Reviewed extraction: src/blockcipher_nd_v3/expansion/skinny.py
# Source commit: d7a8698f4c474e1899f848b5de7a5abb570a13ea
# Source SHA-256: 12eb570f01ee2ba770d6c0d4a91d6be9de89d7dd790306e485e43f2b8a52a707
"""Key-free last-round transform for the fixed eight-round SKINNY task."""

from __future__ import annotations

import torch


def public_backstep(bits: torch.Tensor) -> torch.Tensor:
    """Map [B,4,2,64] MSB-first binary input to the same shape/dtype/device.

    Undo inverse MixColumns, inverse ShiftRows and round-eight constants.
    Inverse S-boxes apply ONLY to the lower half, where no key was injected.
    This is not full round decryption, and accepts no secret/sample metadata.
    """
    if bits.ndim != 4 or tuple(bits.shape[1:]) != (4, 2, 64):
        raise ValueError("expected [B,4,2,64] SKINNY bits")
    weights = torch.tensor([8, 4, 2, 1], device=bits.device)
    cells = (bits.reshape(bits.shape[0], 4, 2, 16, 4).long() * weights).sum(-1)
    c = cells.long()
    # Invert MC: output rows were (a^c^d, a, b^c, a^c).
    a = c[..., 4:8]
    third = c[..., 12:16] ^ a
    b = c[..., 8:12] ^ third
    d = c[..., :4] ^ c[..., 12:16]
    before_sr = torch.cat((a, b, third, d), dim=-1)
    s = before_sr[..., [0, 1, 2, 3, 5, 6, 7, 4, 10, 11, 8, 9, 15, 12, 13, 14]].clone()
    constant = 1
    for _ in range(8 - 1):
        constant = ((constant << 1) & 63) ^ ((constant >> 5) & 1) ^ ((constant >> 4) & 1) ^ 1
    s[..., 0] ^= constant & 15
    s[..., 4] ^= constant >> 4
    s[..., 8] ^= 2
    inverse = torch.tensor([3, 4, 6, 8, 12, 10, 1, 14, 9, 2, 5, 7, 0, 11, 13, 15], device=s.device)
    # Last-round key injection affects only the top half. Do NOT undo its S-boxes.
    s[..., 8:] = inverse[s[..., 8:]]
    shifts = torch.arange(3, -1, -1, device=bits.device)
    return (((s[..., None] >> shifts) & 1).reshape(bits.shape)).to(bits.dtype)

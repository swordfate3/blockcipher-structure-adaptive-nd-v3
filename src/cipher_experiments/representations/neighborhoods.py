"""Fixed local/word/nibble neighborhoods with a zero-padding sentinel."""

from __future__ import annotations

import torch


def neighborhood_indices(kind: str) -> torch.Tensor:
    rows: list[list[int]] = []
    for i in range(64):
        if kind == "local":
            rows.append([j if 0 <= j < 64 else 64 for j in range(i - 4, i + 5)])
        elif kind == "word":
            base, b = (i // 32) * 32, i % 32
            rows.append(
                [base + (b + d) % 32 for d in (-8, -5, -2, -1, 0, 1, 2, 5)] + [(32 - base) + b]
            )
        elif kind == "nibble":
            n, b = divmod(i, 4)
            rows.append(
                [
                    (n + dn) * 4 + b + db if 0 <= n + dn < 16 and 0 <= b + db < 4 else 64
                    for dn in (-1, 0, 1)
                    for db in (-1, 0, 1)
                ]
            )
        else:
            raise ValueError("unknown neighborhood")
    return torch.tensor(rows, dtype=torch.long)


def _gather(raw: torch.Tensor, indices: torch.Tensor, channels: int) -> torch.Tensor:
    if raw.ndim != 4 or raw.shape[1:] != (4, channels, 64):
        raise ValueError(f"expected aligned [B,4,{channels},64] input")
    if (
        indices.shape != (64, 9)
        or indices.dtype != torch.long
        or indices.device != raw.device
        or not bool(((indices >= 0) & (indices <= 64)).all())
    ):
        raise ValueError("expected same-device 64x9 integer neighborhood table in [0,64]")
    padded = torch.cat((raw, raw.new_zeros((*raw.shape[:-1], 1))), dim=-1)
    return padded[..., indices].permute(0, 1, 2, 4, 3).reshape(-1, channels * 9, 64)


def gather_neighborhood(raw: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return _gather(raw, indices, 3)


def gather_relation(channel: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return _gather(channel, indices, 1)

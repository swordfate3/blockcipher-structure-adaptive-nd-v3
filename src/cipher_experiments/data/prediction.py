"""Unlabeled prediction inputs and caller-controlled label generation.

Inputs are generated independently of labels, so callers can build every split
before fitting. This API does not enforce checkpoint/TEST access ordering: a
caller must request TEST labels only after choosing and strictly loading its
checkpoint. TEST is pre-generated data, not a sealed confirmation set.
"""

from __future__ import annotations

import hashlib
import random

import torch
from torch import Tensor

from cipher_experiments.ciphers.simon import encrypt as _encrypt
from cipher_experiments.data.synthetic import (
    _DRAWS_PER_SAMPLE,
    _binary_rows,
    _counts,
    _seed,
    _split,
    _unique_blocks,
)

_SPLITS = ("train", "dev", "test")
_MAX_ROUNDS = 44
_MAX_ROWS = 2_000_000


def make_inputs(
    train: int, dev: int, test: int, *, seed: int, train_pool_rows: int | None = None,
    plaintext_space: str = "full",
) -> dict[str, Tensor]:
    """Generate globally unique CPU-FP32 plaintext rows without making labels.

    With ``train_pool_rows``, TRAIN is the pool prefix while DEV starts after
    the full pool, so different TRAIN prefix lengths share identical heldout
    panels. Unused pool rows are generated only as unlabeled inputs.
    Non-full spaces sample the same unique 32-bit sequence before placing it in
    the active half; the other half is fixed to zero. No key or label is used.
    """
    if type(plaintext_space) is not str or plaintext_space not in (
            "full", "high32-zero", "low32-zero"):
        raise ValueError("plaintext_space must be full, high32-zero or low32-zero")
    if plaintext_space != "full":
        _counts(train, dev, test)
        pool = train if train_pool_rows is None else train_pool_rows
        if type(pool) is not int:
            raise TypeError("train_pool_rows must be a built-in int or None")
        if not train <= pool <= 1_000_000:
            raise ValueError("train_pool_rows must be between train and 1,000,000")
        total = _counts(pool, dev, test)
        _seed(seed)
        if total > (1 << 32):
            raise ValueError("requested rows exceed the 32-bit plaintext space capacity")
        rng = random.Random(seed)
        blocks: list[int] = []
        seen: set[int] = set()
        for _ in range(total * _DRAWS_PER_SAMPLE):
            value = rng.getrandbits(32)
            if value in seen:
                continue
            seen.add(value)
            blocks.append(value if plaintext_space == "high32-zero" else value << 32)
            if len(blocks) == total:
                break
        else:
            raise RuntimeError("unique uint32 sampling exhausted its bounded draw budget")
        inputs = _binary_rows(blocks)
        panels = {name: pair[0] for name, pair in _split(
            inputs, inputs[:, :0], pool, dev
        ).items()}
        panels["train"] = panels["train"][:train].contiguous()
        return panels

    if train_pool_rows is None:
        total = _counts(train, dev, test)
        _seed(seed)
        inputs = _binary_rows(_unique_blocks(total, seed))
        # Preserve the original split implementation and byte identities.
        return {name: pair[0] for name, pair in _split(
            inputs, inputs[:, :0], train, dev
        ).items()}

    _counts(train, dev, test)
    if type(train_pool_rows) is not int:
        raise TypeError("train_pool_rows must be a built-in int or None")
    if not train <= train_pool_rows <= 1_000_000:
        raise ValueError("train_pool_rows must be between train and 1,000,000")
    total = _counts(train_pool_rows, dev, test)
    _seed(seed)
    inputs = _binary_rows(_unique_blocks(total, seed))
    return {
        "train": inputs[:train].contiguous(),
        "dev": inputs[train_pool_rows : train_pool_rows + dev].contiguous(),
        "test": inputs[train_pool_rows + dev :].contiguous(),
    }


def _validate_inputs(inputs: Tensor) -> int:
    if not isinstance(inputs, Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    if inputs.device.type != "cpu":
        raise ValueError("inputs must be on CPU")
    if inputs.dtype != torch.float32:
        raise TypeError("inputs must have dtype torch.float32")
    if inputs.requires_grad:
        raise ValueError("inputs must be detached")
    if inputs.ndim != 2 or inputs.shape[1] != 64:
        raise ValueError("inputs must have shape [N,64]")
    rows = inputs.shape[0]
    if not 1 <= rows <= _MAX_ROWS:
        raise ValueError(f"inputs row count must be between 1 and {_MAX_ROWS}")
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("inputs must contain only finite values")
    if not bool(((inputs == 0) | (inputs == 1)).all()):
        raise ValueError("inputs must contain only binary values 0 and 1")
    return rows


def _plaintexts(inputs: Tensor) -> list[int]:
    """Pack FP32 bit rows into unsigned 64-bit Python values in bounded chunks."""
    result: list[int] = []
    for start in range(0, inputs.shape[0], 8192):
        bits = inputs[start : start + 8192].to(torch.int64)
        packed = torch.zeros(bits.shape[0], dtype=torch.int64)
        for column in range(64):
            packed = (packed << 1) | bits[:, column]
        result.extend(value if value >= 0 else value + (1 << 64) for value in packed.tolist())
    return result


def _null_rng(null_seed: int, split: str) -> random.Random:
    # Fixed domain separator and fixed-width seed encoding avoid ambiguous text
    # concatenation while keeping each split's stream independent.
    material = b"cipher-experiments/prediction-null/v1\0"
    material += null_seed.to_bytes(4, "big") + b"\0" + split.encode("ascii")
    stream_seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
    return random.Random(stream_seed)


def make_labels(
    inputs: Tensor,
    *,
    dataset: str,
    rounds: int,
    target_bit: int,
    key_seed: int,
    label_mode: str = "cipher",
    null_seed: int = 191302,
    split: str,
    label_expression: str = "bit",
    label_bits: tuple[int, ...] = (),
) -> Tensor:
    """Create one detached CPU-FP32 label per input row.

    ``synthetic`` labels are the first input bit and never invoke a cipher.
    ``simon`` labels are the selected MSB-first ciphertext bit under a key
    generated locally from ``key_seed``; the key is never returned. ``null``
    labels use a deterministic local RNG stream independently domain-separated
    by the explicit train/dev/test split. Non-bit expressions select distinct
    MSB-first bits from that same complete ciphertext, never additional encryptions.
    Null validates the declaration but does not compute a ciphertext expression.
    """
    rows = _validate_inputs(inputs)
    if dataset not in ("synthetic", "simon"):
        raise ValueError("dataset must be synthetic or simon")
    if label_mode not in ("cipher", "null"):
        raise ValueError("label_mode must be cipher or null")
    if split not in _SPLITS:
        raise ValueError("split must be train, dev or test")
    _seed(key_seed, "key_seed")
    _seed(null_seed, "null_seed")
    if type(rounds) is not int:
        raise TypeError("rounds must be a built-in int")
    if not 1 <= rounds <= _MAX_ROUNDS:
        raise ValueError(f"rounds must be between 1 and {_MAX_ROUNDS}")
    if type(target_bit) is not int:
        raise TypeError("target_bit must be a built-in int")
    if not 0 <= target_bit < 64:
        raise ValueError("target_bit must be between 0 and 63")
    if label_mode == "null" and dataset != "simon":
        raise ValueError("null labels require dataset=simon")
    if type(label_expression) is not str or label_expression not in (
            "bit", "xor2", "xor3", "xor-and"):
        raise ValueError("label_expression must be bit, xor2, xor3 or xor-and")
    if type(label_bits) not in (tuple, list):
        raise TypeError("label_bits must be a tuple or JSON array")
    if any(type(bit) is not int or not 0 <= bit <= 63 for bit in label_bits):
        raise ValueError("label_bits must contain built-in integers in [0, 63]")
    bits = tuple(label_bits)
    if label_expression == "bit":
        if bits:
            raise ValueError("bit requires empty label_bits; use target_bit")
    else:
        count = 2 if label_expression == "xor2" else 3
        if len(bits) != count or len(set(bits)) != count:
            raise ValueError(f"{label_expression} requires {count} distinct label_bits")
        if bits[0] != target_bit:
            raise ValueError("label_bits[0] must equal target_bit")
        ordered = bits[1:] if label_expression == "xor-and" else bits
        if ordered != tuple(sorted(ordered)):
            raise ValueError("XOR bits, or the AND operand bits, must be increasing")
        if dataset != "simon":
            raise ValueError("non-bit expressions require dataset=simon")

    if label_mode == "null":
        rng = _null_rng(null_seed, split)
        return torch.tensor([[float(rng.getrandbits(1))] for _ in range(rows)], dtype=torch.float32)
    if dataset == "synthetic":
        return inputs[:, :1].clone().contiguous()

    key = random.Random(key_seed).getrandbits(128)
    if label_expression != "bit":
        labels = torch.empty((rows, 1), dtype=torch.float32)
        for start in range(0, rows, 8192):
            part = _plaintexts(inputs[start : start + 8192])
            values: list[float] = []
            for plaintext in part:
                ciphertext = _encrypt(plaintext, key, rounds)
                selected = [(ciphertext >> (63 - bit)) & 1 for bit in bits]
                if label_expression == "xor2":
                    value = selected[0] ^ selected[1]
                elif label_expression == "xor3":
                    value = selected[0] ^ selected[1] ^ selected[2]
                else:
                    value = selected[0] ^ (selected[1] & selected[2])
                values.append(float(value))
            labels[start : start + len(part), 0] = torch.tensor(values, dtype=torch.float32)
        return labels

    # Keep the historical default bit path byte-for-byte equivalent.
    mask = 1 << (63 - target_bit)
    labels = torch.empty((rows, 1), dtype=torch.float32)
    for start in range(0, rows, 8192):
        part = _plaintexts(inputs[start : start + 8192])
        labels[start : start + len(part), 0] = torch.tensor(
            [float(bool(_encrypt(plaintext, key, rounds) & mask)) for plaintext in part],
            dtype=torch.float32,
        )
    return labels

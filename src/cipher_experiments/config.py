"""Strict, portable experiment configuration; no machine paths or approvals."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: str = "synthetic"
    experiment: str = "prediction"
    train_rows: int = 64
    dev_rows: int = 32
    test_rows: int = 32
    data_seed: int = 1
    key_seed: int = 2
    init_seed: int = 3
    rounds: int = 3
    target_bit: int = 0
    hidden_width: int = 32
    model_kind: str = "mlp"
    model_features: str = "full"
    prediction_mode: str = "direct"
    label_mode: str = "cipher"
    null_seed: int = 191302
    batch_size: int = 16
    epochs: int = 2
    learning_rate: float = 0.001
    device: str = "cpu"
    max_seconds: int = 60
    train_pool_rows: int | None = None
    dev_every_epochs: int = 1
    label_expression: str = "bit"
    label_bits: tuple[int, ...] = ()
    plaintext_space: str = "full"
    plaintext_fixed_value: int = 0

    def __post_init__(self) -> None:
        if type(self.plaintext_space) is not str or self.plaintext_space not in (
                "full", "high32-zero", "low32-zero"):
            raise ValueError("plaintext_space must be full, high32-zero or low32-zero")
        if (type(self.plaintext_fixed_value) is not int
                or not 0 <= self.plaintext_fixed_value < (1 << 32)):
            raise ValueError("plaintext_fixed_value must be a built-in integer in [0, 4294967295]")
        if self.plaintext_space == "full" and self.plaintext_fixed_value != 0:
            raise ValueError("full plaintext_space requires plaintext_fixed_value=0")
        if self.plaintext_space != "full" and (
                self.experiment != "prediction" or self.dataset != "simon"
                or self.model_kind != "mlp" or self.model_features != "full"
                or self.prediction_mode != "direct"):
            raise ValueError("non-full plaintext_space requires prediction/simon/mlp/full/direct")
        if type(self.name) is not str or not self.name.strip() or len(self.name) > 100:
            raise ValueError("name must be a nonempty string of at most 100 characters")
        if type(self.label_expression) is not str or self.label_expression not in (
                "bit", "xor2", "xor3", "xor-and"):
            raise ValueError("label_expression must be bit, xor2, xor3 or xor-and")
        if type(self.label_bits) not in (tuple, list):
            raise ValueError("label_bits must be a tuple or JSON array")
        if any(type(bit) is not int or not 0 <= bit <= 63 for bit in self.label_bits):
            raise ValueError("label_bits must contain built-in integers in [0, 63]")
        bits = tuple(self.label_bits)
        object.__setattr__(self, "label_bits", bits)
        if self.label_expression == "bit":
            if bits:
                raise ValueError("bit requires empty label_bits; use target_bit")
        else:
            count = 2 if self.label_expression == "xor2" else 3
            if len(bits) != count or len(set(bits)) != count:
                raise ValueError(f"{self.label_expression} requires {count} distinct label_bits")
            if type(self.target_bit) is not int or bits[0] != self.target_bit:
                raise ValueError("label_bits[0] must equal built-in integer target_bit")
            ordered = bits[1:] if self.label_expression == "xor-and" else bits
            if ordered != tuple(sorted(ordered)):
                raise ValueError("XOR bits, or the AND operand bits, must be increasing")
            if (self.experiment != "prediction" or self.dataset != "simon"
                    or self.model_kind != "mlp" or self.model_features != "full"
                    or self.prediction_mode != "direct"):
                raise ValueError("non-bit expressions require prediction/simon/mlp/full/direct")
        if self.experiment not in ("prediction", "innovation1"):
            raise ValueError("experiment must be prediction or innovation1")
        if self.train_pool_rows is not None:
            if type(self.train_pool_rows) is not int:
                raise ValueError("train_pool_rows must be an integer or null")
            if self.experiment != "prediction" or self.dataset != "simon":
                raise ValueError("train_pool_rows requires experiment=prediction and dataset=simon")
        if self.experiment == "innovation1":
            if self.dataset != "differential":
                raise ValueError("innovation1 requires dataset=differential")
            if any(type(n) is not int or n < 2 or n % 2 for n in (
                    self.train_rows, self.dev_rows, self.test_rows, self.batch_size)):
                raise ValueError("innovation1 split groups and batch_size must be positive even")
            if self.batch_size > self.train_rows or self.train_rows % self.batch_size:
                raise ValueError("innovation1 batch_size must divide train_rows")
            if self.dev_every_epochs != 1:
                raise ValueError("innovation1 requires dev_every_epochs=1")
            if max(self.train_rows, self.dev_rows, self.test_rows) > 65536:
                raise ValueError("innovation1 groups per task must not exceed 65536")
        if self.dataset not in ("synthetic", "circuit-synthetic", "simon", "differential"):
            raise ValueError("dataset must be synthetic, circuit-synthetic, simon or differential")
        if self.experiment == "prediction" and self.dataset == "differential":
            raise ValueError("differential requires experiment=innovation1")
        if self.model_kind not in ("mlp", "logic16", "circuit-aligned", "circuit-random",
                                   "simon-structure", "simon-unrolled", "simon-r2-table",
                                   "simon-r2-local", "simon-r2-vector", "simon-cnn",
                                   "simon-rotation-cnn", "simon-vector"):
            raise ValueError("unknown prediction model_kind")
        if self.model_kind == "simon-vector" and (
                self.experiment != "prediction" or self.dataset != "simon"
                or self.model_features != "full" or self.prediction_mode != "direct"
                or type(self.hidden_width) is not int or self.hidden_width != 32
                or type(self.rounds) is not int or self.rounds not in (5, 6)
                or type(self.target_bit) is not int or self.target_bit != 0):
            raise ValueError("simon-vector requires prediction/simon/full/direct, "
                             "rounds5or6/bit0/width32")
        if self.model_kind in ("simon-cnn", "simon-rotation-cnn") and (
                self.experiment != "prediction" or self.dataset != "simon"
                or self.model_features != "full" or self.prediction_mode != "direct"
                or type(self.hidden_width) is not int or not 8 <= self.hidden_width <= 256
                or type(self.rounds) is not int or self.rounds != 6):
            raise ValueError("simon CNN models require prediction/simon/full/direct, "
                             "rounds=6 and hidden_width=8..256")
        if self.model_kind in ("simon-r2-table", "simon-r2-local", "simon-r2-vector") and (
                self.experiment != "prediction" or self.dataset != "simon"
                or self.hidden_width != 16 or self.model_features != "full"
                or self.prediction_mode != "direct" or type(self.rounds) is not int
                or self.rounds != 2 or type(self.target_bit) is not int
                or not 0 <= self.target_bit <= 31):
            raise ValueError("simon-r2 models require prediction/simon/full/direct/hidden_width16, "
                             "rounds=2 and target_bit=0..31")
        if self.model_kind == "simon-unrolled" and (
                self.dataset != "simon" or self.model_features != "full"
                or self.prediction_mode not in (
                    "direct", "plaintext-xor", "plaintext-xor-no-final-low")
                or (self.prediction_mode == "direct" and (
                    type(self.rounds) is not int or not 1 <= self.rounds <= 8
                    or (self.rounds == 1 and type(self.target_bit) is int
                        and self.target_bit >= 32)))
                or (self.prediction_mode in (
                    "plaintext-xor", "plaintext-xor-no-final-low") and (
                    type(self.rounds) is not int or self.rounds != 2
                    or type(self.target_bit) is not int or self.target_bit >= 32))):
            raise ValueError("simon-unrolled needs simon/full and either direct rounds1..8 "
                             "with a learned target, or a plaintext-xor mode at "
                             "rounds=2/high-half")
        if self.model_kind == "simon-structure" and (
                self.dataset != "simon" or self.hidden_width != 64
                or self.model_features != "full" or self.prediction_mode != "direct"
                or type(self.rounds) is not int or not 1 <= self.rounds <= 6
                or (self.rounds == 1 and type(self.target_bit) is int and self.target_bit >= 32)):
            raise ValueError("simon-structure needs simon/full/direct/hidden_width=64, "
                             "rounds1..6 and a target depending on unknown round constants")
        if self.model_kind.startswith("circuit-") and (
                self.dataset != "circuit-synthetic" or self.hidden_width != 64
                or self.model_features != "full" or self.prediction_mode != "direct"):
            raise ValueError("circuit models need circuit-synthetic, hidden_width=64, full/direct")
        if self.dataset == "circuit-synthetic" and (
                self.experiment != "prediction" or self.label_mode != "cipher"):
            raise ValueError("circuit-synthetic requires prediction and deterministic toy labels")
        if self.experiment == "innovation1" and self.model_kind != "mlp":
            raise ValueError("model_kind applies only to prediction")
        if self.model_features not in ("full", "support"):
            raise ValueError("model_features must be full or support")
        if self.prediction_mode not in (
                "direct", "public-residual", "public-coordinate", "plaintext-xor",
                "plaintext-xor-no-final-low"):
            raise ValueError("unknown prediction_mode")
        if self.prediction_mode in (
                "plaintext-xor", "plaintext-xor-no-final-low") and (
                    self.model_kind != "simon-unrolled"):
            raise ValueError("plaintext-xor modes require model_kind=simon-unrolled")
        if self.label_mode not in ("cipher", "null"):
            raise ValueError("label_mode must be cipher or null")
        if self.experiment == "innovation1" and (
                self.model_features != "full" or self.prediction_mode != "direct"
                or self.label_mode != "cipher"):
            raise ValueError("prediction controls do not apply to innovation1")
        if (self.prediction_mode == "public-residual"
                and (self.dataset != "simon" or type(self.rounds) is not int
                     or not 1 <= self.rounds <= 4)):
            raise ValueError("public-residual requires simon rounds 1..4")
        if self.prediction_mode == "public-coordinate" and (
                self.experiment != "prediction" or self.model_kind != "mlp"
                or self.dataset != "simon" or self.model_features != "full"
                or type(self.rounds) is not int or not 2 <= self.rounds <= 44):
            raise ValueError(
                "public-coordinate requires prediction/mlp/simon/full and rounds 2..44"
            )
        if self.model_features == "support" and self.dataset != "simon":
            raise ValueError("support features require simon")
        if self.label_mode == "null" and self.dataset != "simon":
            raise ValueError("null labels require simon")
        bounds = {"train_rows": (1, 1_000_000), "dev_rows": (1, 1_000_000),
                  "test_rows": (1, 1_000_000), "rounds": (1, 44), "target_bit": (0, 63),
                  "hidden_width": (1, 4096), "batch_size": (1, 65536),
                  "epochs": (1, 10000), "dev_every_epochs": (1, 10000),
                  "max_seconds": (1, 86400),
                  "data_seed": (0, 2**32 - 1), "key_seed": (0, 2**32 - 1),
                  "init_seed": (0, 2**32 - 1), "null_seed": (0, 2**32 - 1)}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.dev_every_epochs > self.epochs:
            raise ValueError("dev_every_epochs must not exceed epochs")
        if self.model_kind == "logic16" and self.hidden_width < 64:
            raise ValueError("logic16 requires hidden_width >= 64 for feature coverage")
        if self.dataset == "circuit-synthetic" and (
                self.train_rows + self.dev_rows + self.test_rows > 256):
            raise ValueError("circuit-synthetic has only 256 distinct relevant input assignments")
        if self.train_rows + self.dev_rows + self.test_rows > 2_000_000:
            raise ValueError("total rows exceed 2,000,000")
        if self.train_pool_rows is not None:
            if not self.train_rows <= self.train_pool_rows <= 1_000_000:
                raise ValueError("train_pool_rows must be between train_rows and 1,000,000")
            if self.train_pool_rows + self.dev_rows + self.test_rows > 2_000_000:
                raise ValueError("train pool, DEV and TEST rows exceed 2,000,000")
        if (type(self.learning_rate) not in (int, float)
                or not 0 < self.learning_rate <= 1):
            raise ValueError("learning_rate must be finite in (0, 1]")
        if self.device not in ("cpu", "cuda:0", "cuda:1"):
            raise ValueError("device must be cpu, cuda:0 or cuda:1; no automatic fallback")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.plaintext_fixed_value == 0:
            del result["plaintext_fixed_value"]
        if self.plaintext_space == "full":
            del result["plaintext_space"]
        if self.label_expression == "bit" and not self.label_bits:
            # Additive defaults must not change historical config/checkpoint identities.
            del result["label_expression"]
            del result["label_bits"]
        return result

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                         allow_nan=False).encode()).hexdigest()


def load_config(path: Path) -> ExperimentConfig:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate config field: {key}")
            result[key] = value
        return result
    if path.stat().st_size > 65536:
        raise ValueError("configuration exceeds 64 KiB")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                       parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
    allowed = {field.name for field in fields(ExperimentConfig)}
    if type(value) is not dict or not value.keys() <= allowed:
        raise ValueError("config must be an object with known fields only")
    if "name" not in value:
        raise ValueError("config requires name")
    return ExperimentConfig(**value)

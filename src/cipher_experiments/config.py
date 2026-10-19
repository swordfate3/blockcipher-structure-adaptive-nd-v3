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
    model_features: str = "full"
    prediction_mode: str = "direct"
    label_mode: str = "cipher"
    null_seed: int = 191302
    batch_size: int = 16
    epochs: int = 2
    learning_rate: float = 0.001
    device: str = "cpu"
    max_seconds: int = 60

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip() or len(self.name) > 100:
            raise ValueError("name must be a nonempty string of at most 100 characters")
        if self.experiment not in ("prediction", "innovation1"):
            raise ValueError("experiment must be prediction or innovation1")
        if self.experiment == "innovation1":
            if self.dataset != "differential":
                raise ValueError("innovation1 requires dataset=differential")
            if any(type(n) is not int or n < 2 or n % 2 for n in (
                    self.train_rows, self.dev_rows, self.test_rows, self.batch_size)):
                raise ValueError("innovation1 split groups and batch_size must be positive even")
            if self.batch_size > self.train_rows or self.train_rows % self.batch_size:
                raise ValueError("innovation1 batch_size must divide train_rows")
            if max(self.train_rows, self.dev_rows, self.test_rows) > 65536:
                raise ValueError("innovation1 groups per task must not exceed 65536")
        if self.dataset not in ("synthetic", "simon", "differential"):
            raise ValueError("dataset must be synthetic, simon or differential")
        if self.experiment == "prediction" and self.dataset == "differential":
            raise ValueError("differential requires experiment=innovation1")
        if self.model_features not in ("full", "support"):
            raise ValueError("model_features must be full or support")
        if self.prediction_mode not in ("direct", "public-residual"):
            raise ValueError("prediction_mode must be direct or public-residual")
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
        if self.model_features == "support" and self.dataset != "simon":
            raise ValueError("support features require simon")
        if self.label_mode == "null" and self.dataset != "simon":
            raise ValueError("null labels require simon")
        bounds = {"train_rows": (1, 1_000_000), "dev_rows": (1, 1_000_000),
                  "test_rows": (1, 1_000_000), "rounds": (1, 44), "target_bit": (0, 63),
                  "hidden_width": (1, 4096), "batch_size": (1, 65536),
                  "epochs": (1, 10000), "max_seconds": (1, 86400),
                  "data_seed": (0, 2**32 - 1), "key_seed": (0, 2**32 - 1),
                  "init_seed": (0, 2**32 - 1), "null_seed": (0, 2**32 - 1)}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.train_rows + self.dev_rows + self.test_rows > 2_000_000:
            raise ValueError("total rows exceed 2,000,000")
        if (type(self.learning_rate) not in (int, float)
                or not 0 < self.learning_rate <= 1):
            raise ValueError("learning_rate must be finite in (0, 1]")
        if self.device not in ("cpu", "cuda:0", "cuda:1"):
            raise ValueError("device must be cpu, cuda:0 or cuda:1; no automatic fallback")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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

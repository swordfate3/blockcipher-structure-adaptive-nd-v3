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
    train_rows: int = 64
    dev_rows: int = 32
    test_rows: int = 32
    data_seed: int = 1
    key_seed: int = 2
    init_seed: int = 3
    rounds: int = 3
    target_bit: int = 0
    hidden_width: int = 32
    batch_size: int = 16
    epochs: int = 2
    learning_rate: float = 0.001
    device: str = "cpu"
    max_seconds: int = 60

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip() or len(self.name) > 100:
            raise ValueError("name must be a nonempty string of at most 100 characters")
        if self.dataset not in ("synthetic", "simon"):
            raise ValueError("dataset must be synthetic or simon")
        bounds = {"train_rows": (1, 1_000_000), "dev_rows": (1, 1_000_000),
                  "test_rows": (1, 1_000_000), "rounds": (1, 44), "target_bit": (0, 63),
                  "hidden_width": (1, 4096), "batch_size": (1, 65536),
                  "epochs": (1, 10000), "max_seconds": (1, 86400),
                  "data_seed": (0, 2**32 - 1), "key_seed": (0, 2**32 - 1),
                  "init_seed": (0, 2**32 - 1)}
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

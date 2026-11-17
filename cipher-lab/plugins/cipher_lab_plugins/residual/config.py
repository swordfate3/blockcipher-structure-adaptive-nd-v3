"""Public experiment specifications; stdlib only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from cipher_lab.training_config import (
    MLPSpec,
    PredictionRecipe,
    TrainSpec,
    _object,
    _training,
)


@dataclass(frozen=True, slots=True)
class ResidualPredictionRecipe:
    """Fixed two-arm experiment; public seeds never specify a private key."""

    schema: int
    method: str
    reference: str
    model: MLPSpec
    training: TrainSpec
    rounds: int
    target_bit: int
    label_mode: str

    def __post_init__(self) -> None:
        if self.method != "bit-residual-paired-v1":
            raise ValueError("method: unsupported residual method")
        if self.reference != "simon64-zero-master-v1":
            raise ValueError("reference: unsupported public reference")
        PredictionRecipe(self.schema, "bit-prediction-v1", self.model, self.training,
                         self.rounds, self.target_bit, self.label_mode)

    def to_dict(self) -> dict[str, object]:
        node = PredictionRecipe(self.schema, "bit-prediction-v1", self.model, self.training,
                                self.rounds, self.target_bit, self.label_mode).to_dict()
        return {**node, "method": self.method, "reference": self.reference}


def parse_residual_prediction(value: object) -> ResidualPredictionRecipe:
    node = _object(value, {"schema", "method", "reference", "model", "training", "rounds",
                           "target_bit", "label_mode"}, "residual prediction recipe")
    model = _object(node["model"], {"hidden_width"}, "MLP model")
    return ResidualPredictionRecipe(
        schema=cast(int, node["schema"]), method=cast(str, node["method"]),
        reference=cast(str, node["reference"]), model=MLPSpec(cast(int, model["hidden_width"])),
        training=_training(node["training"]), rounds=cast(int, node["rounds"]),
        target_bit=cast(int, node["target_bit"]), label_mode=cast(str, node["label_mode"]),
    )


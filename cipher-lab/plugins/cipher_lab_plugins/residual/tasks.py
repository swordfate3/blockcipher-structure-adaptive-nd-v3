from __future__ import annotations

from dataclasses import dataclass

from cipher_lab.data.bundles import PredictionTask


@dataclass(frozen=True, slots=True)
class ResidualPredictionTask:
    """Absolute target identity plus the complete public reference algorithm."""

    cipher: str
    rounds: int
    target_bit: int
    label_mode: str
    reference: str

    def __post_init__(self) -> None:
        PredictionTask(self.cipher, self.rounds, self.target_bit, self.label_mode)
        if self.reference != "simon64-zero-master-v1":
            raise ValueError("invalid public reference")

    def to_dict(self) -> dict[str, object]:
        return {**PredictionTask(self.cipher, self.rounds, self.target_bit,
                                 self.label_mode).to_dict(), "reference": self.reference}


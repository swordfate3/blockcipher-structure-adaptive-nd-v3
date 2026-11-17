from __future__ import annotations

from dataclasses import dataclass

from cipher_lab_plugins.feature_probe.config import FEATURE_REPRESENTATION
from cipher_lab_plugins.residual.tasks import ResidualPredictionTask


@dataclass(frozen=True, slots=True)
class FeatureProbeTask:
    cipher: str
    rounds: int
    target_bit: int
    label_mode: str
    reference: str
    representation: str

    def __post_init__(self) -> None:
        ResidualPredictionTask(self.cipher, self.rounds, self.target_bit, self.label_mode,
                               self.reference)
        if (self.rounds != 3 or self.target_bit != 0
                or self.representation != FEATURE_REPRESENTATION):
            raise ValueError('invalid feature probe task')

    def to_dict(self) -> dict[str, object]:
        return {**ResidualPredictionTask(self.cipher, self.rounds, self.target_bit,
                self.label_mode, self.reference).to_dict(), 'representation': self.representation}


from __future__ import annotations

from dataclasses import dataclass

from cipher_lab_plugins.frontier.config import validate_frontier_identity
from cipher_lab_plugins.frontier.simon_frontier import FeatureIdentity
from cipher_lab_plugins.residual.tasks import ResidualPredictionTask


@dataclass(frozen=True, slots=True)
class FrontierTask:
    cipher: str
    rounds: int
    target_bit: int
    label_mode: str
    reference: str
    identity: FeatureIdentity

    def __post_init__(self) -> None:
        ResidualPredictionTask(self.cipher, self.rounds, self.target_bit, self.label_mode,
                               self.reference)
        if any(type(value) is not str for value in (
                self.cipher, self.label_mode, self.reference)):
            raise ValueError('invalid frontier task types')
        validate_frontier_identity(self.identity)
        if (self.rounds, self.target_bit) != (self.identity.rounds, self.identity.target_bit):
            raise ValueError('frontier task/identity mismatch')

    def to_dict(self) -> dict[str, object]:
        return {**ResidualPredictionTask(self.cipher, self.rounds, self.target_bit,
                self.label_mode, self.reference).to_dict(), 'identity': self.identity.to_dict()}


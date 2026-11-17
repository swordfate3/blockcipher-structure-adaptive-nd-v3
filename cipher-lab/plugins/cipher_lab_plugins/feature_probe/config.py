"""Public experiment specifications; stdlib only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from cipher_lab.training_config import (
    MLPSpec,
    PredictionRecipe,
    TrainSpec,
    _integer,
    _object,
    _training,
)

FEATURE_REPRESENTATION = 'simon-r3-residual13-v1'


@dataclass(frozen=True, slots=True)
class FeatureMLPSpec:
    hidden_width: int
    representation: str

    def __post_init__(self) -> None:
        _integer(self.hidden_width, 'hidden_width', 1, 4096)
        if self.representation != FEATURE_REPRESENTATION:
            raise ValueError('unsupported feature representation')

    def to_dict(self) -> dict[str, object]:
        return {'hidden_width': self.hidden_width, 'representation': self.representation}


@dataclass(frozen=True, slots=True)
class FeatureProbeRecipe:
    schema: int
    method: str
    reference: str
    representation: str
    model: MLPSpec
    training: TrainSpec
    rounds: int
    target_bit: int
    label_mode: str

    def __post_init__(self) -> None:
        if self.method != 'bit-public-feature-probe-v1':
            raise ValueError('unsupported feature probe method')
        if self.reference != 'simon64-zero-master-v1':
            raise ValueError('unsupported public reference')
        if self.representation != FEATURE_REPRESENTATION:
            raise ValueError('unsupported feature representation')
        PredictionRecipe(self.schema, 'bit-prediction-v1', self.model, self.training,
                         self.rounds, self.target_bit, self.label_mode)
        _integer(self.rounds, 'rounds', 3, 3)
        _integer(self.target_bit, 'target_bit', 0, 0)

    def to_dict(self) -> dict[str, object]:
        node = PredictionRecipe(self.schema, 'bit-prediction-v1', self.model, self.training,
                                self.rounds, self.target_bit, self.label_mode).to_dict()
        return {**node, 'method': self.method, 'reference': self.reference,
                'representation': self.representation}


def parse_feature_probe(value: object) -> FeatureProbeRecipe:
    node = _object(value, {'schema', 'method', 'reference', 'representation', 'model',
                           'training', 'rounds', 'target_bit', 'label_mode'}, 'feature probe')
    model = _object(node['model'], {'hidden_width'}, 'MLP model')
    return FeatureProbeRecipe(cast(int, node['schema']), cast(str, node['method']),
        cast(str, node['reference']), cast(str, node['representation']),
        MLPSpec(cast(int, model['hidden_width'])), _training(node['training']),
        cast(int, node['rounds']), cast(int, node['target_bit']), cast(str, node['label_mode']))


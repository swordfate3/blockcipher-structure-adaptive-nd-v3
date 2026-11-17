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
from cipher_lab_plugins.frontier.simon_frontier import (
    FeatureIdentity,
    PublicFeaturePlan,
    decode_plan,
    encode_plan,
)

FRONTIER_METHOD = 'bit-public-feature-frontier-v1'


_FRONTIER_AUDITED = {
    4: ('simon-r4-residual89-v1', 89,
        'bb87a12a789b9713f0d79875ac85b560b05fe938817550182ec6d90b5bae2968',
        '21038a42d331559d5f3b5252489849f1ae10b05571541fcc328c942108754dff'),
    5: ('simon-r5-factored-v1', 8098,
        '43f03e294b5b7849ae23b6d592a01ca89c9f609f652d5dbe50be1e0139f9b409',
        '3554d0ea9fa0a38ba2811bbc69dcbbf8019ba1a7f9ac0e72e41ad9c2d1dd4841'),
}


def validate_frontier_identity(identity: FeatureIdentity, *, audited: bool = False) -> None:
    """Strict at every trust entry; FeatureIdentity itself is only a container.

    A loader can inspect another structurally valid public plan; only recipes
    claim admission to this frozen experiment. No r6 identity was audited.
    """
    import re
    if type(identity) is not FeatureIdentity:
        raise ValueError('expected FeatureIdentity')
    _integer(identity.rounds, 'identity.rounds', 4, 6)
    _integer(identity.target_bit, 'identity.target_bit', 0, 0)
    _integer(identity.dimension, 'identity.dimension', 1, 8192)
    expected = {4: 'simon-r4-residual89-v1', 5: 'simon-r5-factored-v1',
                6: 'simon-r6-factored-v1'}[identity.rounds]
    if (type(identity.representation) is not str or identity.representation != expected
            or (identity.rounds == 4 and identity.dimension != 89)):
        raise ValueError('invalid frontier representation/dimension')
    for digest in (identity.order_sha256, identity.plan_sha256):
        if type(digest) is not str or re.fullmatch('[0-9a-f]{64}', digest) is None:
            raise ValueError('invalid frontier SHA')
    if audited and _FRONTIER_AUDITED.get(identity.rounds) != (
            identity.representation, identity.dimension, identity.order_sha256,
            identity.plan_sha256):
        raise ValueError('frontier identity was not admitted by frozen audit')


def parse_frontier_identity(value: object) -> FeatureIdentity:
    node = _object(value, {'rounds', 'target_bit', 'representation', 'dimension',
                           'order_sha256', 'plan_sha256'}, 'frontier identity')
    identity = FeatureIdentity(cast(int, node['rounds']), cast(int, node['target_bit']),
        cast(str, node['representation']), cast(int, node['dimension']),
        cast(str, node['order_sha256']), cast(str, node['plan_sha256']))
    validate_frontier_identity(identity)
    return identity


@dataclass(frozen=True, slots=True)
class FrontierMLPSpec:
    hidden_width: int
    identity: FeatureIdentity
    plan: PublicFeaturePlan

    def __post_init__(self) -> None:
        _integer(self.hidden_width, 'hidden_width', 1, 4096)
        validate_frontier_identity(self.identity)
        if type(self.plan) is not PublicFeaturePlan:
            raise ValueError('expected validated public plan')
        # Reconstruct to reject mutated frozen objects and stale _identity.
        validated = decode_plan(encode_plan(self.plan))
        if validated.identity != self.identity:
            raise ValueError('frontier plan identity mismatch')

    def to_dict(self) -> dict[str, object]:
        return {'hidden_width': self.hidden_width, 'identity': self.identity.to_dict()}


@dataclass(frozen=True, slots=True)
class FrontierRecipe:
    schema: int
    method: str
    reference: str
    identity: FeatureIdentity
    model: MLPSpec
    training: TrainSpec
    rounds: int
    target_bit: int
    label_mode: str

    def __post_init__(self) -> None:
        if (type(self.method) is not str or type(self.reference) is not str
                or self.method != FRONTIER_METHOD
                or self.reference != 'simon64-zero-master-v1'):
            raise ValueError('invalid frontier method/reference')
        PredictionRecipe(self.schema, 'bit-prediction-v1', self.model, self.training,
                         self.rounds, self.target_bit, self.label_mode)
        if (type(self.model) is not MLPSpec or type(self.training) is not TrainSpec
                or type(self.label_mode) is not str):
            raise ValueError('invalid frontier model/training/label types')
        self.model.__post_init__()
        self.training.__post_init__()
        validate_frontier_identity(self.identity, audited=True)
        if (self.rounds, self.target_bit) != (self.identity.rounds, self.identity.target_bit):
            raise ValueError('frontier recipe/identity mismatch')

    @property
    def representation(self) -> str:
        return self.identity.representation

    def to_dict(self) -> dict[str, object]:
        node = PredictionRecipe(self.schema, 'bit-prediction-v1', self.model, self.training,
                                self.rounds, self.target_bit, self.label_mode).to_dict()
        return {**node, 'method': self.method, 'reference': self.reference,
                'identity': self.identity.to_dict()}


def parse_frontier(value: object) -> FrontierRecipe:
    node = _object(value, {'schema', 'method', 'reference', 'identity', 'model', 'training',
                           'rounds', 'target_bit', 'label_mode'}, 'frontier recipe')
    model = _object(node['model'], {'hidden_width'}, 'MLP model')
    return FrontierRecipe(cast(int, node['schema']), cast(str, node['method']),
        cast(str, node['reference']), parse_frontier_identity(node['identity']),
        MLPSpec(cast(int, model['hidden_width'])), _training(node['training']),
        cast(int, node['rounds']), cast(int, node['target_bit']), cast(str, node['label_mode']))


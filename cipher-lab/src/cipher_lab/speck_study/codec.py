"""Bounded strict JSON model, entirely stdlib; external identity is mandatory.

No torch.load, arbitrary object deserialization, cipher calls or label access.
Masks are full uint64 MSB-first; B32 occupies the high half of the integer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction

from cipher_lab.speck_study_config import SpeckStudyRecipe, strict_json

MAX_MODEL_BYTES = 256 * 1024


def _integer(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


@dataclass(frozen=True, slots=True)
class StudyTerm:
    mask: int
    signed_sum: int


@dataclass(frozen=True, slots=True)
class StudyPredictor:
    kind: str
    task: str
    source_sha256: str
    recipe: SpeckStudyRecipe
    terms: tuple[StudyTerm, ...]
    count: int
    intercept: int = 0
    tie_bit: int = 0

    @property
    def identity(self) -> dict[str, object]:
        return {'kind': self.kind, 'task': self.task, 'source_sha256': self.source_sha256,
                'method': self.recipe.method, 'synthetic': self.recipe.synthetic}

    def score_numerator(self, point: int) -> int:
        validate_predictor(self)
        bits = self.recipe.r_bits if self.kind == 'R' else self.recipe.b_bits
        if not _integer(point, 0, 2**64-1) or point & ((1 << (64-bits))-1):
            raise ValueError('point outside encoded domain')
        return sum(term.signed_sum * (-1 if (point & term.mask).bit_count() % 2 else 1)
                   for term in self.terms)

    def score(self, point: int) -> float:
        return self.score_numerator(point) / self.count

    def predict_bit(self, point: int) -> int:
        """Main protocol: negative score =>1, zero score =>0 (not official tie1)."""
        return int(self.score_numerator(point) < 0)


def validate_predictor(model: StudyPredictor) -> None:
    if type(model) is not StudyPredictor or type(model.recipe) is not SpeckStudyRecipe:
        raise ValueError('invalid model/recipe type')
    if type(model.kind) is not str or model.kind not in ('R', 'B'):
        raise ValueError('invalid kind')
    task = 'teacher-reproduction' if model.kind == 'R' else 'absolute-bit-prediction'
    if model.task != task or type(model.task) is not str:
        raise ValueError('kind/task mismatch')
    if (type(model.source_sha256) is not str
            or not re.fullmatch('[0-9a-f]{64}', model.source_sha256)):
        raise ValueError('invalid source SHA256')
    count = model.recipe.r_coef_size if model.kind == 'R' else model.recipe.b_coef_size
    if not _integer(model.count, 1, 10_000_000) or model.count != count:
        raise ValueError('coefficient sample count mismatch')
    if not _integer(model.intercept, 0, 0) or not _integer(model.tie_bit, 0, 0):
        raise ValueError('intercept/tie must be zero')
    cap = model.recipe.max_prefixes if model.kind == 'R' else model.recipe.b_max_terms
    if type(model.terms) is not tuple or not 1 <= len(model.terms) <= cap:
        raise ValueError('invalid term count')
    bits = model.recipe.r_bits if model.kind == 'R' else model.recipe.b_bits
    seen = set()
    threshold = Fraction(str(model.recipe.coefficient_threshold))
    for term in model.terms:
        if type(term) is not StudyTerm or not _integer(term.mask, 1, 2**64-1):
            raise ValueError('invalid/zero mask')
        if term.mask & ((1 << (64-bits))-1) or term.mask in seen:
            raise ValueError('duplicate/out-of-domain mask')
        seen.add(term.mask)
        if not _integer(term.signed_sum, -count, count) or (term.signed_sum-count) % 2:
            raise ValueError('integer sum/count mismatch')
        if abs(term.signed_sum) * threshold.denominator <= count * threshold.numerator:
            raise ValueError('term does not strictly exceed coefficient threshold')
    ordered = tuple(sorted(model.terms, key=(lambda t: t.mask) if model.kind == 'R'
                           else (lambda t: (-abs(t.signed_sum), t.mask))))
    if model.terms != ordered:
        raise ValueError('noncanonical term order')


def encode_predictor(model: StudyPredictor) -> bytes:
    validate_predictor(model)
    value = {'schema': 1, 'identity': model.identity, 'recipe': model.recipe.to_dict(),
             'count': model.count, 'intercept': model.intercept, 'tie_bit': model.tie_bit,
             'terms': [{'mask': t.mask, 'signed_sum': t.signed_sum} for t in model.terms]}
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(raw) > MAX_MODEL_BYTES:
        raise ValueError('model exceeds byte cap')
    return raw


def decode_predictor(raw: bytes, *, expected_identity: dict[str, object],
                     expected_recipe: SpeckStudyRecipe) -> StudyPredictor:
    if type(raw) is not bytes or len(raw) > MAX_MODEL_BYTES:
        raise ValueError('expected bounded JSON bytes')
    if type(expected_recipe) is not SpeckStudyRecipe or type(expected_identity) is not dict:
        raise ValueError('external expected identity/recipe required')
    value = strict_json(raw)
    keys = {'schema', 'identity', 'recipe', 'count', 'intercept', 'tie_bit', 'terms'}
    if type(value) is not dict or set(value) != keys or not _integer(value['schema'], 1, 1):
        raise ValueError('invalid model schema')
    recipe = SpeckStudyRecipe.parse(value['recipe'])
    identity = value['identity']
    identity_keys = {'kind', 'task', 'source_sha256', 'method', 'synthetic'}
    if (type(identity) is not dict or set(identity) != identity_keys
            or set(expected_identity) != identity_keys or identity != expected_identity
            or any(type(identity[k]) is not type(expected_identity[k]) for k in identity_keys)
            or recipe != expected_recipe):
        raise ValueError('external identity/recipe mismatch')
    if (identity['method'] != recipe.method or type(identity['synthetic']) is not bool
            or identity['synthetic'] != recipe.synthetic):
        raise ValueError('recipe/identity mismatch')
    terms = value['terms']
    if type(terms) is not list or not 1 <= len(terms) <= 400:
        raise ValueError('invalid term list')
    for term in terms:
        if type(term) is not dict or set(term) != {'mask', 'signed_sum'}:
            raise ValueError('invalid term object')
    model = StudyPredictor(kind=identity['kind'], task=identity['task'],
                           source_sha256=identity['source_sha256'], recipe=recipe,
                           terms=tuple(StudyTerm(t['mask'], t['signed_sum']) for t in terms),
                           count=value['count'], intercept=value['intercept'],
                           tie_bit=value['tie_bit'])
    validate_predictor(model)
    return model

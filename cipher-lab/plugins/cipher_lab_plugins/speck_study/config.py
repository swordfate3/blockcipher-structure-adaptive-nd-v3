"""Frozen, stdlib-only SPECK study protocol; a recipe is never permission to run."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Any

METHOD = 'speck-fourier-paper-prediction-v1'
SYNTHETIC_METHOD = 'speck-fourier-synthetic-v1'


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON field')
        result[key] = value
    return result


def reject_constant(value: str) -> Any:
    raise ValueError('nonfinite JSON number')


def strict_json(text: str | bytes) -> Any:
    try:
        return json.loads(text, object_pairs_hook=strict_object, parse_constant=reject_constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError('invalid JSON') from exc


@dataclass(frozen=True, slots=True)
class SpeckStudyRecipe:
    method: str = METHOD
    synthetic: bool = False
    rounds: int = 8
    r_bits: int = 64
    b_bits: int = 32
    initial_depth: int = 7
    pairs_per_layer: int = 10_000_000
    r_classification_size: int = 10_000_000
    r_coef_size: int = 1_000_000
    r_agreement_size: int = 10_000_000
    b_coef_size: int = 1_000_000
    b_dev_size: int = 10_000_000
    b_test_size: int = 10_000_000
    r0_max_blocks: int = 512
    tau: float = .1
    bucket_threshold: float = .005
    coefficient_threshold: float = .05
    threshold_rule: str = 'strict-greater'
    max_prefixes: int = 400
    r_term_rule: str = 'all-above-threshold'
    b_max_terms: int = 12
    b_term_rule: str = 'abs-sum-desc-mask-asc'
    intercept: int = 0
    main_tie_bit: int = 0
    official_tie_bit: int = 1
    teacher_threshold: float = .5
    teacher_threshold_rule: str = 'strict-greater'
    delta_left: int = 0x0040
    delta_right: int = 0
    target_bit_msb0: int = 0
    point_encoding: str = 'uint64-msb-first-b32-high-half'
    pair_rule: str = 'shared-low-suffix'
    sampler: str = 'PCG64-with-replacement'
    seed_derivation: str = 'sha256-stage-depth-v1'
    public_r_classification_seed: int = 19691112
    public_r_search_seed: int = 29691112
    public_r_coef_seed: int = 39691112
    public_r_agreement_seed: int = 49691112
    public_b_search_seed: int = 59691112
    public_b_coef_seed: int = 69691112
    public_b_dev_seed: int = 79691112
    public_b_test_seed: int = 89691112
    private_key_bytes: int = 8
    r_teacher_accuracy_min: float = .510
    r_teacher_accuracy_max: float = .518
    r_eq4_accuracy_reference: float = .5087755
    r_eq4_accuracy_tolerance: float = .002
    r_terms_min: int = 10
    r_terms_max: int = 14
    r_mask_intersection_min: int = 11
    r_coefficient_error_max: float = .01
    r_agreement_reference: float = .8027425
    r_agreement_tolerance: float = .006
    r_classification_accuracy_reference: float = .5087755
    r_classification_accuracy_tolerance: float = .002
    b_dev_accuracy_min: float = .501
    b_dev_delta_min_exclusive: float = 0.
    b_test_accuracy_min: float = .501
    b_test_lower_min_exclusive: float = 0.
    b_test_alpha: float = .01
    b_test_bound: str = 'delta-sqrt(2ln(1/alpha)/N)'
    baseline: str = 'constant0'
    retry_count: int = 0
    optimizer_updates: int = 0
    null_words: int = 0
    chunk_rows: int = 50_000
    batch_fallback_rows: tuple[int, ...] = (50_000, 25_000, 12_500)
    parity_tile_rows: int = 4096
    physical_gpu_index: int = 1
    gpu_binding: str = 'UUID-and-PCI-required-no-fallback'
    gpu_memory_bytes: int = 20 * 2**30
    torch_allocator_bytes: int = 8 * 2**30
    tf_allocator_bytes: int = 8 * 2**30
    gpu_free_min_bytes: int = 24 * 2**30
    gpu_utilization_max_percent: int = 30
    host_rss_bytes: int = 32 * 2**30
    output_bytes: int = 4 * 2**30
    terminal_reserve_bytes: int = 64 * 2**20
    sample_interval_seconds: float = .5
    wall_seconds: int = 28_800
    science_seconds: int = 28_200
    close_reserve_seconds: int = 600
    cpu_rule: str = 'record-process-tree-only'
    missing_resource_rule: str = 'stop-unverified'
    oom_rule: str = 'stop-no-replay'
    policy_real_blocks: int = 542_000_000
    policy_teacher_rows: int = 1_161_000_000

    def __post_init__(self) -> None:
        defaults = {field.name: field.default for field in fields(self)}
        scalable = {'rounds', 'r_bits', 'b_bits', 'initial_depth', 'pairs_per_layer',
                    'r_classification_size', 'r_coef_size', 'r_agreement_size', 'b_coef_size',
                    'b_dev_size', 'b_test_size', 'r0_max_blocks'}
        for field in fields(self):
            value, default = getattr(self, field.name), defaults[field.name]
            if type(value) is not type(default):
                raise ValueError(f'{field.name}: wrong type')
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'{field.name}: nonfinite')
            if field.name == 'method':
                expected = SYNTHETIC_METHOD if self.synthetic else METHOD
                if value != expected:
                    raise ValueError('method/synthetic identity mismatch')
            elif field.name != 'synthetic' and not (self.synthetic and field.name in scalable):
                if value != default:
                    raise ValueError(f'{field.name}: fixed study protocol')
        if any(type(n) is not int for n in self.batch_fallback_rows):
            raise ValueError('batch_fallback_rows: expected exact integer entries')
        if not 1 <= self.initial_depth < min(self.r_bits, self.b_bits) <= 64:
            raise ValueError('invalid search dimensions')
        if max(self.r_bits, self.b_bits) > 64 or self.initial_depth > 7:
            raise ValueError('invalid search dimensions')
        if self.synthetic and (self.rounds != 0 or self.r0_max_blocks != 0):
            raise ValueError('synthetic must not request real encryption')
        for name in scalable - {'rounds', 'r0_max_blocks', 'initial_depth', 'r_bits', 'b_bits'}:
            if not 1 <= getattr(self, name) <= defaults[name]:
                raise ValueError(f'{name}: invalid panel size')

    @classmethod
    def synthetic_recipe(cls, **sizes: int) -> SpeckStudyRecipe:
        data: dict[str, Any] = dict(sizes)
        return cls(method=SYNTHETIC_METHOD, synthetic=True, rounds=0, r0_max_blocks=0, **data)

    @classmethod
    def parse(cls, value: object) -> SpeckStudyRecipe:
        if isinstance(value, (str, bytes)):
            value = strict_json(value)
        if type(value) is not dict or set(value) != {field.name for field in fields(cls)}:
            raise ValueError('recipe requires complete exact fields')
        data = dict(value)
        if type(data['batch_fallback_rows']) is list:
            data['batch_fallback_rows'] = tuple(data['batch_fallback_rows'])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['batch_fallback_rows'] = list(self.batch_fallback_rows)
        return data


def estimate_study(recipe: SpeckStudyRecipe) -> dict[str, int]:
    if type(recipe) is not SpeckStudyRecipe:
        raise ValueError('expected SpeckStudyRecipe')
    r_layers = recipe.r_bits - recipe.initial_depth
    b_layers = recipe.b_bits - recipe.initial_depth
    r_search = 2 * recipe.pairs_per_layer * r_layers
    b_search = 2 * recipe.pairs_per_layer * b_layers
    teacher = (r_search + recipe.r_classification_size + recipe.r_coef_size
               + recipe.r_agreement_size)
    r_real = 0 if recipe.synthetic else 2 * recipe.r_classification_size
    b_real = 0 if recipe.synthetic else (b_search + recipe.b_coef_size + recipe.b_dev_size
                                        + recipe.b_test_size)
    candidates = 2 * recipe.max_prefixes
    return {
        'r0_real_blocks': recipe.r0_max_blocks,
        'r_classification_real_blocks': r_real,
        'r_search_teacher_rows': r_search,
        'r_classification_teacher_rows': recipe.r_classification_size,
        'r_coef_teacher_rows': recipe.r_coef_size,
        'r_agreement_teacher_rows': recipe.r_agreement_size,
        'teacher_rows': teacher,
        'b_search_real_blocks': 0 if recipe.synthetic else b_search,
        'b_coef_real_blocks': 0 if recipe.synthetic else recipe.b_coef_size,
        'b_dev_real_blocks': 0 if recipe.synthetic else recipe.b_dev_size,
        'b_test_real_blocks': 0 if recipe.synthetic else recipe.b_test_size,
        'b_real_blocks': b_real,
        'real_blocks': recipe.r0_max_blocks + r_real + b_real,
        'r_search_sample_bucket_products': r_layers * recipe.pairs_per_layer * candidates,
        'b_search_sample_bucket_products': b_layers * recipe.pairs_per_layer * candidates,
        'search_sample_bucket_products': ((r_layers + b_layers) * recipe.pairs_per_layer
                                          * candidates),
        'coefficient_parity_products': recipe.max_prefixes * (recipe.r_coef_size
                                                            + recipe.b_coef_size),
        'r_agreement_parity_products': recipe.max_prefixes * recipe.r_agreement_size,
        'r_classification_parity_products': ((recipe.max_prefixes + 12)
                                              * recipe.r_classification_size),
        'b_scoring_parity_products': recipe.b_max_terms * (recipe.b_dev_size + recipe.b_test_size),
        'optimizer_updates': 0,
        'null_words': 0,
    }

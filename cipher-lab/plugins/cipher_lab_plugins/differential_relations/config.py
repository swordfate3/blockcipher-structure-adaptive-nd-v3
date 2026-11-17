"""Pure-stdlib frozen scientific recipe, not a configurable search space."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

MASKS = (0x0000000008040201, 0x0000000080402010, 0x0804020100000000, 0x8040201000000000)
ENCODING = 'orbit-u64-msb0-v1'
INPUT_ADAPTER = 'plaintext-pair-msb0-v1'
METHOD = 'shared-m-differential-relations-v1'


@dataclass(frozen=True, slots=True)
class Recipe:
    method: str = METHOD
    cipher: str = 'SIMON64/128'
    rounds: int = 6
    alpha: int = 64
    masks: tuple[int, ...] = MASKS
    encoding: str = ENCODING
    input_adapter: str = INPUT_ADAPTER
    key_protocol: str = 'single-private-independent-128-bit-v1'
    train_small: int = 262144
    train_large: int = 1048576
    dev: int = 65536
    test: int = 262144
    sample_seed: int = 20261116
    shuffle_seed: int = 20261117
    init_a: int = 31
    init_b: int = 37
    null_train_seed: int = 41
    null_dev_seed: int = 43
    learning_rates: tuple[float, ...] = (.001, .0003)
    optimizer: str = 'Adam'
    betas: tuple[float, ...] = (.9, .999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    clip_norm: float = 1.0
    precision: str = 'float32'
    amp: bool = False
    dropout: float = 0.0
    schedule: str = 'none'
    batch_groups: int = 256
    updates: int = 32768
    dev_every: int = 1024
    checkpoint_rule: str = 'minimum-dev-bce-earliest-tie-v1'
    wall_seconds: int = 28800
    fit_wall_seconds: int = 5400
    real_blocks: int = 2752768
    gate_blocks: int = 256
    optimizer_updates: int = 163840
    forward_pairs: int = 183566336
    gpu_bytes: int = 20 * 1024**3
    rss_bytes: int = 32 * 1024**3
    artifact_bytes: int = 2 * 1024**3
    retries: int = 0

    def __post_init__(self) -> None:
        # No alternate Recipe(...) may silently reduce the scientific protocol.
        for field in self.__dataclass_fields__.values():
            value = getattr(self, field.name)
            if type(value) is not type(field.default) or value != field.default:
                raise ValueError('FIXED_RECIPE_REQUIRED')

    def to_dict(self) -> dict[str, object]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('DUPLICATE_CONFIG_KEY')
        result[name] = value
    return result


def load_config(path: Path) -> Recipe:
    try:
        actual = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError('INVALID_CONFIG') from None
    expected = Recipe().to_dict()
    # Canonical JSON comparison is type sensitive (True is not 1, 1 is not 1.0).
    if not isinstance(actual, dict) or json.dumps(actual, sort_keys=True) != json.dumps(
            expected, sort_keys=True):
        raise ValueError('FIXED_RECIPE_REQUIRED')
    return Recipe()

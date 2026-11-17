"""Pure fixed Fourier recipe and worst-case counters (no execution or approval)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import cast


def _integer(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}: expected bounded integer")
    return value


@dataclass(frozen=True, slots=True)
class FourierRecipe:
    """Formal six-round protocol, or explicitly separate tiny synthetic protocol.

    Synthetic inputs occupy the *highest* input_bits of uint64 words, with all
    remaining bits zero. Synthetic rounds=0 means no cipher, never low-round SIMON.
    """

    schema: int = 1
    method: str = "bit-fourier-search-v1"
    synthetic: bool = False
    rounds: int = 6
    target_bit: int = 0
    input_bits: int = 64
    pairs_per_layer: int = 65536
    beam_width: int = 32
    tau: float = 0.1
    coef_size: int = 65536
    dev_size: int = 65536
    test_size: int = 65536
    search_seed: int = 19691111
    coef_seed: int = 29691111
    dev_seed: int = 39691111
    test_seed: int = 49691111

    def __post_init__(self) -> None:
        _integer(self.schema, "schema", 1, 1)
        if type(self.method) is not str or self.method != "bit-fourier-search-v1":
            raise ValueError("method: unsupported Fourier method")
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic: expected boolean")
        if type(self.tau) is not float or not math.isfinite(self.tau) or not 0 < self.tau <= 1:
            raise ValueError("tau: expected finite float in (0, 1]")
        _integer(self.target_bit, "target_bit", 0, 0)
        _integer(self.beam_width, "beam_width", 1, 32)
        for name in ("search_seed", "coef_seed", "dev_seed", "test_seed"):
            _integer(getattr(self, name), name, 0, 2**32 - 1)
        if self.synthetic:
            _integer(self.rounds, "synthetic rounds", 0, 0)
            _integer(self.input_bits, "synthetic input_bits", 1, 8)
            for name in ("pairs_per_layer", "coef_size", "dev_size", "test_size"):
                _integer(getattr(self, name), name, 1, 4096)
        else:
            fixed = {"rounds": 6, "input_bits": 64, "pairs_per_layer": 65536,
                     "beam_width": 32, "coef_size": 65536, "dev_size": 65536,
                     "test_size": 65536, "search_seed": 19691111, "coef_seed": 29691111,
                     "dev_seed": 39691111, "test_seed": 49691111}
            for name, value in fixed.items():
                _integer(getattr(self, name), name, value, value)
            if self.tau != 0.1:
                raise ValueError("tau: formal protocol fixes 0.1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_fourier(value: object) -> FourierRecipe:
    """Reject unknown/missing fields, bool-as-integer and non-finite values."""
    names = {field.name for field in fields(FourierRecipe)}
    if type(value) is not dict or set(value) != names:
        raise ValueError("Fourier recipe: missing or unknown fields")
    return FourierRecipe(**cast("dict[str, object]", value))  # type: ignore[arg-type]


def estimate_fourier(recipe: FourierRecipe) -> dict[str, object]:
    """Worst-case counts, including both real/null bucket arithmetic, no sampling."""
    search = 2 * recipe.input_bits * recipe.pairs_per_layer
    total = search + recipe.coef_size + recipe.dev_size + recipe.test_size
    return {"method": recipe.method, "synthetic": recipe.synthetic,
            "search_real_blocks": search, "max_real_blocks": total,
            "max_null_words": total,
            "max_sample_bucket_products": (
                2 * recipe.input_bits * (2 * recipe.beam_width) * recipe.pairs_per_layer
            ), "updates": 0, "nn_forward_rows": 0}


def load_experiment(path: Path) -> FourierRecipe:
    """Load only this plugin's strict recipe; no core method dispatch."""
    from cipher_lab.config import _load

    return parse_fourier(_load(path))

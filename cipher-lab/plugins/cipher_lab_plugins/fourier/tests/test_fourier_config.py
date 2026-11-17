"""Pure configuration checks: no scientific imports or execution."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
from cipher_lab_plugins.fourier.config import FourierRecipe, estimate_fourier, parse_fourier


def test_formal_recipe_and_exact_counts() -> None:
    recipe = FourierRecipe()
    assert parse_fourier(recipe.to_dict()) == recipe
    assert (recipe.rounds, recipe.input_bits, recipe.target_bit) == (6, 64, 0)
    assert (recipe.search_seed, recipe.coef_seed, recipe.dev_seed, recipe.test_seed) == (
        19691111, 29691111, 39691111, 49691111,
    )
    estimate = estimate_fourier(recipe)
    assert estimate["search_real_blocks"] == 8388608
    assert estimate["max_real_blocks"] == estimate["max_null_words"] == 8585216
    assert estimate["max_sample_bucket_products"] == 536870912
    assert estimate["updates"] == estimate["nn_forward_rows"] == 0
    with pytest.raises(FrozenInstanceError):
        recipe.tau = 0.2  # type: ignore[misc]


@pytest.mark.parametrize("field,value", [
    ("rounds", 5), ("input_bits", 63), ("target_bit", 1), ("tau", .2),
    ("pairs_per_layer", 16), ("beam_width", 31), ("coef_size", 16),
    ("dev_size", 16), ("test_size", 16), ("search_seed", 1),
    ("coef_seed", 1), ("dev_seed", 1), ("test_seed", 1),
    ("method", "other"), ("schema", 2), ("synthetic", 1),
    ("rounds", True), ("tau", True), ("tau", float("nan")),
    ("tau", float("inf")), ("tau", 10**500), ("input_bits", 64.0),
])
def test_formal_rejects_changes_and_bad_types(field: str, value: object) -> None:
    obj = FourierRecipe().to_dict()
    obj[field] = value
    with pytest.raises(ValueError):
        parse_fourier(obj)


@pytest.mark.parametrize("value", [None, [], {}, {"unknown": 1}])
def test_bad_objects(value: object) -> None:
    with pytest.raises(ValueError):
        parse_fourier(value)


def test_synthetic_is_explicit_bounded_and_independent() -> None:
    demo = FourierRecipe(synthetic=True, input_bits=3, rounds=0, pairs_per_layer=16,
                         coef_size=8, dev_size=8, test_size=8, beam_width=4)
    assert parse_fourier(demo.to_dict()) == demo
    assert estimate_fourier(demo)["synthetic"] is True
    for changes in ({"synthetic": False}, {"input_bits": 9}, {"rounds": 6},
                    {"pairs_per_layer": 65536}, {"coef_size": 0}, {"tau": 0.0},
                    {"beam_width": 33}):
        with pytest.raises(ValueError):
            replace(demo, **changes)


def test_missing_and_extra_fields_rejected() -> None:
    obj = FourierRecipe().to_dict()
    obj.pop("synthetic")
    with pytest.raises(ValueError):
        parse_fourier(obj)
    obj = FourierRecipe().to_dict() | {"key": "not permitted"}
    with pytest.raises(ValueError):
        parse_fourier(obj)

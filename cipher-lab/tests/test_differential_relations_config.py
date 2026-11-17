import json
from pathlib import Path

import pytest
from cipher_lab_plugins.differential_relations.config import MASKS, Recipe, load_config


def test_fixed_recipe_and_file() -> None:
    recipe = load_config(Path('configs/differential-relations-simon6.json'))
    assert recipe == Recipe()
    assert recipe.train_small == 262144
    assert recipe.learning_rates == (.001, .0003)
    assert recipe.real_blocks == 2752768
    assert MASKS == (0x8040201, 0x80402010, 0x0804020100000000, 0x8040201000000000)


@pytest.mark.parametrize('change', [{'train_large': 8}, {'updates': True}, {'extra': 1},
                                   {'learning_rates': [.0003, .001]}, {'alpha': 1}])
def test_no_recipe_drift(tmp_path: Path, change: dict[str, object]) -> None:
    payload = Recipe().to_dict() | change
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_and_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / 'config.json'
    path.write_text('{}')
    with pytest.raises(ValueError):
        load_config(path)
    original = json.dumps(Recipe().to_dict())
    path.write_text(original[:-1] + ', "alpha": 64}')
    with pytest.raises(ValueError):
        load_config(path)

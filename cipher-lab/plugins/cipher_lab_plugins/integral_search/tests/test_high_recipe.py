from pathlib import Path

from cipher_lab_plugins.integral_search.config import load_experiment

PROJECT = Path(__file__).resolve().parents[4]

def test_high_fixed_recipe_exists() -> None:
    recipe = load_experiment(PROJECT / "configs/integral-search-k18k20.json")
    assert recipe.to_dict() == {
        "schema": 1, "method": "integral-search-v1", "rounds": [11],
        "patterns": [list(range(20)), list(range(32, 52))],
        "discovery_active": 18, "expanded_active": 20,
        "groups": {"discovery": 2112, "screen": 320, "confirm": 256},
        "chunk_points": 65536,
    }

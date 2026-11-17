"""Explicit frontier entry; check imports stdlib declarations only."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from cipher_lab.config import _load
from cipher_lab_plugins.frontier.config import FrontierRecipe, parse_frontier
from cipher_lab_plugins.prediction_common.cli import main as prediction_main


def load_experiment(path: Path) -> FrontierRecipe:
    return parse_frontier(_load(path))


def main(argv: Sequence[str]) -> int:
    return prediction_main(argv, method='bit-public-feature-frontier-v1', parse=parse_frontier)

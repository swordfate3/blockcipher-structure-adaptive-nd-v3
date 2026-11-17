"""Explicit residual entry; check imports stdlib declarations only."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from cipher_lab.config import _load
from cipher_lab_plugins.prediction_common.cli import main as prediction_main
from cipher_lab_plugins.residual.config import ResidualPredictionRecipe, parse_residual_prediction


def load_experiment(path: Path) -> ResidualPredictionRecipe:
    return parse_residual_prediction(_load(path))


def main(argv: Sequence[str]) -> int:
    return prediction_main(argv, method='bit-residual-paired-v1', parse=parse_residual_prediction)

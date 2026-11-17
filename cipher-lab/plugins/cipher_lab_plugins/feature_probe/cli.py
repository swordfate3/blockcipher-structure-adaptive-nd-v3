"""Explicit feature_probe entry; check imports stdlib declarations only."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from cipher_lab.config import _load
from cipher_lab_plugins.feature_probe.config import FeatureProbeRecipe, parse_feature_probe
from cipher_lab_plugins.prediction_common.cli import main as prediction_main


def load_experiment(path: Path) -> FeatureProbeRecipe:
    return parse_feature_probe(_load(path))


def main(argv: Sequence[str]) -> int:
    return prediction_main(argv, method='bit-public-feature-probe-v1', parse=parse_feature_probe)

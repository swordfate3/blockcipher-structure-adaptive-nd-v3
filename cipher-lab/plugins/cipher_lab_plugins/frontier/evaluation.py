"""Independent saved-file evaluation for frontier; no training or generation."""
from __future__ import annotations

from pathlib import Path

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter
from cipher_lab_plugins.prediction_common.evaluation import run_evaluation as evaluate_saved


def run_evaluation(checkpoint: Path, checkpoint_sha: str, dataset: Path, dataset_sha: str,
                   meter: Meter, store: RunStore) -> dict[str, object]:
    return evaluate_saved(checkpoint, checkpoint_sha, dataset, dataset_sha, meter, store,
                          expected_method='bit-public-feature-frontier-v1')

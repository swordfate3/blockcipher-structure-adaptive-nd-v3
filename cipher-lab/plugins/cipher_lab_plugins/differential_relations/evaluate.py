"""Saved-only evaluation and numerical panels. No generation or training imports."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np

from .checkpoint import Identity, load_checkpoint, load_saved_dataset
from .config import Recipe
from .metrics import (
    accuracy_comparisons,
    fit_baselines,
    logloss,
    paired_bce_comparison,
    relation_metrics,
)
from .report import render_report


def encode_saved(orbits: Any) -> Any:
    pairs = np.stack((orbits, orbits ^ np.uint64(64)), axis=1)
    return (
        ((pairs[..., None] >> np.arange(63, -1, -1, dtype=np.uint64)) & np.uint64(1))
        .astype(np.float32)
        .reshape(-1, 4, 2, 64)
    )


def panel(logits: Any, labels: Any) -> dict[str, object]:
    return {
        "relations": relation_metrics(logits, labels),
        "mean_bce": float(logloss(logits, labels).mean(dtype=np.float64)),
    }


def baseline_panels(baselines: Any, orbits: Any, labels: Any) -> dict[str, object]:
    constant, stump = baselines.logits(orbits)
    return {"constant": panel(constant, labels), "stump": panel(stump, labels)}


def comparisons(logits: Any, labels: Any, baselines: Any, orbits: Any) -> dict[str, object]:
    constant, stump = baselines.logits(orbits)
    return {
        "paired_bce": paired_bce_comparison(
            logloss(logits, labels), logloss(constant, labels), logloss(stump, labels)
        ),
        "accuracy": accuracy_comparisons(logits, constant, stump, labels),
    }


class SavedPredictor:
    """Independent inference implementation: never imports optimizer/trainer."""

    def __init__(self, device: str = "cpu") -> None:
        import torch

        from cipher_lab.representations.tasks import public_features

        self.torch: Any = torch
        self.device = device
        self.public = torch.tensor(public_features("SIMON"), dtype=torch.float32, device=device)

    def model(self) -> Any:
        from cipher_lab.models.shared_m_relations import SharedMRelations

        # Constructor RNG is restored; every actual parameter is strictly replaced.
        with self.torch.random.fork_rng(devices=[]):
            self.torch.random.default_generator.manual_seed(0)
            return SharedMRelations().to(self.device)

    def predict(self, model: Any, orbits: Any) -> Any:
        model.eval()
        with self.torch.no_grad():
            return (
                model(self.torch.from_numpy(encode_saved(orbits)).to(self.device), self.public)
                .reshape(-1, 4)
                .detach()
                .cpu()
                .numpy()
            )


def _array_hash(array: Any) -> str:
    array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    digest = sha256(
        json.dumps([array.dtype.str, list(array.shape)], separators=(",", ":")).encode("ascii")
    )
    digest.update(array.tobytes())
    return digest.hexdigest()


def _output_path(output: Path) -> Path:
    output = Path(os.path.abspath(output))
    for path in (*reversed(output.parents), output):
        if path.is_symlink() or (
            path.exists() and getattr(path.stat(), "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("UNSAFE_OUTPUT_PATH")
    if not output.parent.is_dir():
        raise ValueError("OUTPUT_PARENT")
    return output


def write_evaluation(output: Path, saved: Mapping[str, Any]) -> None:
    """Exclusive new directory; an interrupted output remains evidence, not overwritten."""
    output = _output_path(output)
    output.mkdir(exist_ok=False)
    raw = json.dumps(saved, sort_keys=True, allow_nan=False).encode("utf-8")
    (output / "evaluation.json").write_bytes(raw)
    (output / "report.html").write_bytes(render_report(saved))


def evaluate_saved(
    *,
    run_record: Path,
    checkpoint: Path,
    dataset: Path,
    output: Path,
    predictor_factory: Callable[[], Any] = SavedPredictor,
) -> dict[str, Any]:
    from cipher_lab.artifacts import read_binary, read_saved

    output = _output_path(output)
    if output.exists():
        raise FileExistsError("OCCUPIED")
    # Only canonical original RunStore layouts are accepted, never lookalike JSON.
    root = run_record.parents[2]
    experiment_id = run_record.parent.name
    if run_record != root / "data" / experiment_id / "run.json":
        raise ValueError("RUN_RECORD_PATH")
    saved: Any = read_saved(root, experiment_id)
    recipe = dict(saved["recipe"])
    plugin_source = recipe.pop("plugin_source")
    if not isinstance(plugin_source, dict):
        raise ValueError("PLUGIN_SOURCE_IDENTITY")
    expected_recipe = json.loads(json.dumps(Recipe().to_dict()))
    if recipe != expected_recipe:
        raise ValueError("RECIPE_IDENTITY")
    summary = saved["summary"]
    if summary["completion"] != "COMPLETE" or summary["experiment_id"] != experiment_id:
        raise ValueError("INCOMPLETE_OR_WRONG_EXPERIMENT")
    fits = summary["fits"]
    if (
        len(fits) != 5
        or {f["fit_id"] for f in fits} != {f"F{i}" for i in range(1, 6)}
        or set(summary["selected_sha256"]) != {f"F{i}" for i in range(1, 6)}
    ):
        raise ValueError("FROZEN_MATRIX_IDENTITY")
    for fit in fits:
        expected_ref = f"model/{fit['fit_id']}-best.pt"
        if (
            fit["checkpoint_ref"] != expected_ref
            or fit["checkpoint_sha256"] != summary["selected_sha256"][fit["fit_id"]]
            or saved["artifacts"][expected_ref]["sha256"] != fit["checkpoint_sha256"]
            or fit["identity"]["experiment_id"] != experiment_id
        ):
            raise ValueError("FROZEN_MATRIX_IDENTITY")
    dataset_ref = summary["dataset_ref"]
    if dataset != root / "data" / experiment_id / dataset_ref:
        raise ValueError("DATASET_REFERENCE")
    digest = saved["artifacts"][dataset_ref]["sha256"]
    raw = read_binary(root, experiment_id, dataset_ref, digest, Recipe().artifact_bytes)
    # Decode exactly verified immutable bytes, not a newly reopened caller path.
    with TemporaryDirectory(prefix="relations-evaluate-", dir=output.parent) as temporary:
        temp = Path(temporary) / "dataset.npz"
        temp.write_bytes(raw)
        decoded = load_saved_dataset(temp, experiment_id=experiment_id)
    metadata = decoded.metadata
    fixed = Recipe()
    if (
        len(decoded.train) != fixed.train_large
        or len(decoded.dev) != fixed.dev
        or len(decoded.test) != fixed.test
        or metadata["train_small"] != fixed.train_small
    ):
        raise ValueError("RECIPE_DATASET_SIZES")
    if (
        metadata["test_hash"] != summary["test_hash"]
        or metadata["selected_sha256"] != summary["selected_sha256"]
        or metadata["train_hash"] != summary["train_hash"]
        or metadata["dev_hash"] != summary["dev_hash"]
    ):
        raise ValueError("DATASET_IDENTITY")
    entries = [
        entry
        for entry in summary["fits"]
        if checkpoint == root / "models" / experiment_id / Path(entry["checkpoint_ref"]).name
    ]
    if len(entries) != 1:
        raise ValueError("CHECKPOINT_REFERENCE")
    entry = entries[0]
    fit_id = entry["fit_id"]
    selected_sha = summary["selected_sha256"][fit_id]
    ref = entry["checkpoint_ref"]
    if (
        saved["artifacts"][ref]["sha256"] != selected_sha
        or entry["checkpoint_sha256"] != selected_sha
    ):
        raise ValueError("SELECTED_IDENTITY")
    small = fit_id in ("F1", "F2")
    n = metadata["train_small"] if small else len(decoded.train)
    train = decoded.train[:n]
    null = fit_id == "F5"
    train_labels = decoded.null_train if null else decoded.train_labels[:n]
    dev_labels = decoded.null_dev if null else decoded.dev_labels
    if train_labels is None or dev_labels is None or decoded.test_labels is None:
        raise ValueError("MISSING_TARGETS")
    # Content hash uses the same explicit public canonical serialization as decoder.
    train_hash = sha256(
        (_array_hash(train) + _array_hash(decoded.train_labels[:n])).encode("ascii")
    ).hexdigest()
    identity = Identity(
        experiment_id,
        fit_id,
        train_hash,
        metadata["dev_hash"],
        _array_hash(train_labels),
        _array_hash(dev_labels),
        "row-permutation-v1" if null else "real-v1",
    )
    if identity.to_dict() != entry["identity"]:
        raise ValueError("TARGET_IDENTITY")
    predictor = predictor_factory()
    model = predictor.model()
    model_raw = read_binary(root, experiment_id, ref, selected_sha, Recipe().artifact_bytes)
    with TemporaryDirectory(prefix="relations-selected-", dir=output.parent) as temporary:
        temp = Path(temporary) / "selected.pt"
        temp.write_bytes(model_raw)
        load_checkpoint(temp, model, identity, expected_sha256=selected_sha)
    metrics: dict[str, object] = {}
    test_logits = None
    for name, orbits, labels in (
        ("train", train, train_labels),
        ("dev", decoded.dev, dev_labels),
        ("test", decoded.test, decoded.test_labels),
    ):
        outputs = []
        for start in range(0, len(orbits), 1024):
            values = np.asarray(
                predictor.predict(model, orbits[start : start + 1024]), dtype=np.float64
            )
            if values.shape != (min(1024, len(orbits) - start), 4) or not np.isfinite(values).all():
                raise ValueError("INVALID_FORWARD")
            outputs.append(values)
        logits = np.concatenate(outputs)
        metrics[name] = panel(logits, labels)
        if name == "test":
            test_logits = logits
    baselines = fit_baselines(train, decoded.train_labels[:n])
    comparison = comparisons(test_logits, decoded.test_labels, baselines, decoded.test)
    paired = dict(cast(dict[str, Any], comparison["paired_bce"]))
    if fit_id != "F3":
        paired.pop("main_signal", None)
        paired["role"] = "descriptive-only"
    result: dict[str, Any] = {
        "run_id": experiment_id,
        "status": "SUCCEEDED",
        "run_ref": saved["run_ref"],
        "run_sha256": saved["run_sha256"],
        "recipe": saved["recipe"],
        "counts": {},
        "summary": {
            "completion": "COMPLETE",
            "operation": "independent-saved-evaluation",
            "checkpoint_sha256": selected_sha,
            "dataset_sha256": digest,
            "test_hash": metadata["test_hash"],
            "fits": [{"fit_id": fit_id, "final_metrics": metrics}],
            ("main_comparisons" if fit_id == "F3" else "selected_comparisons"): paired,
            (
                "accuracy_comparisons" if fit_id == "F3" else "selected_accuracy_comparisons"
            ): comparison["accuracy"],
            "baseline_size": "small" if small else "large",
            "baselines": {
                "small" if small else "large": {
                    kind: {"test": value}
                    for kind, value in baseline_panels(
                        baselines, decoded.test, decoded.test_labels
                    ).items()
                }
            },
            "targets": {
                "train": "row-permutation-v1" if null else "real-v1",
                "dev": "row-permutation-v1" if null else "real-v1",
                "test": "real-v1",
            },
        },
    }
    write_evaluation(output, result)
    return result

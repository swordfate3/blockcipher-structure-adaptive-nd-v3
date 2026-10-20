"""CPU TRAIN-consistency fitting for one public SIMON Boolean output cone."""
from __future__ import annotations

import hashlib
import time
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from .artifacts import file_sha256, git_identity, source_identity, write_json
from .config import ExperimentConfig
from .evaluation import evaluate_model, load_checkpoint
from .fitting.hard_simon import HardSimonFitResult, fit_hard_simon

if TYPE_CHECKING:
    from .fitting.sat_simon import SatSimonFitResult
from .models.structured_simon import HardSimonPredictor
from .training import (
    _save_checkpoint,
    baseline_parameters,
    input_panels,
    labels_for,
    reference_panels,
)


def _plaintexts(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(sum(int(bit) << (63 - i) for i, bit in enumerate(row)) for row in x.tolist())


def train(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    if config.model_kind not in ("simon-hard-search", "simon-hard-sat") or config.device != "cpu":
        raise ValueError("hard fitting requires simon-hard-search/sat on cpu")
    sat = config.model_kind == "simon-hard-sat"
    method = ("Glucose4 CNF hard TRAIN consistency" if sat
              else "integer-bitparallel TRAIN consistency")
    criterion = ("one SAT model satisfying all TRAIN rows; no model enumeration" if sat
                 else "first ascending TRAIN-zero-error assignment")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "manifest.json", {
        "schema": "cipher-experiments-hard-fit-v1", "started_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config.sha256(), "source_sha256": source_identity(),
        "git": git_identity(Path(__file__).resolve().parents[2]),
        "device": "cpu", "method": method,
        "selection": criterion + "; no DEV candidate selection",
        "search_order": config.search_order or "ascending",
        "test_policy": "TEST labels only after strict selected-checkpoint loading",
        "timeout_policy": "cooperative; not a hard process limit",
        "key_schedule_constraint": False, "gradient_updates": 0})
    panel_sha: dict[str, Any] = {}

    def deadline() -> float:
        remaining = config.max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("cooperative hard-fitting budget exceeded")
        return remaining

    try:
        if sat:
            from .fitting.sat_simon import check_sat_backend
            write_json(output / "backend.json", check_sat_backend())
        inputs = input_panels(config)
        panels = {s: (inputs[s], labels_for(config, inputs[s], s)) for s in ("train", "dev")}
        for split, x in inputs.items():
            panel_sha[split] = {"rows": len(x), "x_sha256": hashlib.sha256(
                x.contiguous().numpy().tobytes()).hexdigest()}
            if split in panels:
                panel_sha[split]["y_sha256"] = hashlib.sha256(
                    panels[split][1].contiguous().numpy().tobytes()).hexdigest()
        write_json(output / "data.json", panel_sha)
        x, y = panels["train"]
        # The fitting API receives only plaintext integers, TRAIN bits and public structure.
        fit: HardSimonFitResult | SatSimonFitResult
        if sat:
            from .fitting.sat_simon import fit_sat_simon
            fit = fit_sat_simon(_plaintexts(x), tuple(int(v) for v in y.flatten().tolist()),
                                rounds=config.rounds, target_bit=config.target_bit,
                                max_seconds=deadline())
        else:
            fit = fit_hard_simon(_plaintexts(x), tuple(int(v) for v in y.flatten().tolist()),
                                 rounds=config.rounds, target_bit=config.target_bit,
                                 max_seconds=deadline(),
                                 order=cast("Literal['ascending','descending']",
                                             config.search_order or "ascending"))
        write_json(output / "fit.json", asdict(fit))
        if fit.assignment is None:
            result: dict[str, Any] = {
                "status": fit.status, "method": method,
                "config_sha256": config.sha256(), "gradient_updates": 0,
                "test_labels_generated": False, "prediction_metrics": None,
                "elapsed_seconds": time.monotonic() - started}
        else:
            model = HardSimonPredictor(rounds=config.rounds, target_bit=config.target_bit)
            model.set_assignment(fit.assignment)
            model.validate_assignment()
            majority, probability = baseline_parameters(y, None)
            refs = reference_panels(config, inputs)
            residual_bit, residual_p = baseline_parameters(y, refs.get("train"))
            _save_checkpoint(output / "best.pt", model, config, 1, 0, majority, probability)
            sha = file_sha256(output / "best.pt")
            selected, _, _, _ = load_checkpoint(output / "best.pt", sha,
                                                device=torch.device("cpu"))
            write_json(output / "selection.json", {
                "assignment": fit.assignment, "checkpoint_sha256": sha,
                "criterion": criterion,
                "dev_used_for_selection": False, "test_labels_generated": False})
            deadline()
            train_check = evaluate_model(
                selected, x, y, device=torch.device("cpu"), batch_size=config.batch_size,
                majority=majority, probability=probability)
            if train_check["accuracy"] != 1.:
                raise ValueError("fitted assignment failed independent TRAIN circuit validation")
            # Explicit freeze/load boundary before requesting any TEST answer.
            panels["test"] = (inputs["test"], labels_for(config, inputs["test"], "test"))
            panel_sha["test"]["y_sha256"] = hashlib.sha256(
                panels["test"][1].contiguous().numpy().tobytes()).hexdigest()
            write_json(output / "data.json", panel_sha)
            metrics = {s: evaluate_model(
                selected, *panels[s], device=torch.device("cpu"), batch_size=config.batch_size,
                majority=majority, probability=probability, reference=refs.get(s),
                residual_majority=residual_bit, residual_probability=residual_p)
                for s in ("train", "dev", "test")}
            deadline()
            result = {"status": "completed", "method": method,
                      "config_sha256": config.sha256(), "gradient_updates": 0,
                      "assignment": fit.assignment, "parameters": len(model.key_coordinates),
                      "metrics": metrics, "best_checkpoint_sha256": sha,
                      "elapsed_seconds": time.monotonic() - started}
        write_json(output / "results.json", result)
        write_json(output / "exit.json", {
            "status": result["status"], "exit_code": 0, "gradient_updates": 0,
            "elapsed_seconds": time.monotonic() - started})
        return result
    except BaseException as error:
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(output / "exit.json", {
            "status": "failed", "exit_code": 1, "gradient_updates": 0,
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.monotonic() - started})
        raise

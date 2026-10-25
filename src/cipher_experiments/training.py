"""One local-process experiment, independent of Git sync, SSH and old run cards."""
from __future__ import annotations

import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .artifacts import append_progress, file_sha256, git_identity, source_identity, write_json
from .config import ExperimentConfig
from .evaluation import CHECKPOINT_SCHEMA, evaluate_model, load_checkpoint
from .models.prediction import build_predictor, public_reference
from .models.simon_two_round import (
    TwoRoundLocalPredictor,
    TwoRoundTablePredictor,
    TwoRoundVectorPredictor,
)
from .models.structured_simon import StructuredSimonPredictor
from .models.unrolled_simon import UnrolledSimonPredictor
from .two_round_diagnostics import two_round_local_diagnostics


def device_for(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available() or device.index is None:
            raise ValueError(f"requested {name}, but CUDA is unavailable; no CPU fallback")
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"requested {name}, but only {torch.cuda.device_count()} GPUs visible")
    return device


def datasets(config: ExperimentConfig) -> dict[str, tuple[Tensor, Tensor]]:
    # Explicit evaluate capability: this helper materializes all labels on request.
    inputs = input_panels(config)
    return {split: (x, labels_for(config, x, split)) for split, x in inputs.items()}


def input_panels(config: ExperimentConfig) -> dict[str, Tensor]:
    if config.dataset == "circuit-synthetic":
        from .data.circuit import make_circuit_inputs
        return make_circuit_inputs(config.train_rows, config.dev_rows, config.test_rows,
                                   seed=config.data_seed)
    from .data.prediction import make_inputs
    return make_inputs(config.train_rows, config.dev_rows, config.test_rows,
                       seed=config.data_seed, train_pool_rows=config.train_pool_rows)


def labels_for(config: ExperimentConfig, x: Tensor, split: str) -> Tensor:
    if config.dataset == "circuit-synthetic":
        from .data.circuit import circuit_labels
        return circuit_labels(x)
    from .data.prediction import make_labels
    return make_labels(x, dataset=config.dataset, rounds=config.rounds,
                       target_bit=config.target_bit,
                       key_seed=config.key_seed, label_mode=config.label_mode,
                       null_seed=config.null_seed, split=split)


def reference_panels(config: ExperimentConfig, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    if config.dataset == "simon" and config.rounds <= 4:
        return {s: public_reference(x, config.rounds, config.target_bit) for s, x in inputs.items()}
    return {}


def baseline_parameters(labels: Tensor, reference: Tensor | None) -> tuple[int, float]:
    residual = labels if reference is None else (labels != reference).float()
    ones = int(residual.sum())
    return int(2 * ones > len(labels)), (ones + .5) / (len(labels) + 1)


def _save_checkpoint(path: Path, model: nn.Module, config: ExperimentConfig,
                     epoch: int, updates: int, majority: int, probability: float) -> None:
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    payload = {"schema": CHECKPOINT_SCHEMA, "config": config.to_dict(),
               "config_sha256": config.sha256(), "state_dict": state,
               "epoch": epoch, "updates": updates,
               "majority": majority, "probability": probability}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    if config.experiment != "prediction":
        raise ValueError("prediction train requires experiment=prediction")
    # Validate environment before claiming a new run directory.
    device = device_for(config.device)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    updates = 0
    presentations = 0
    current_epoch = 0
    write_json(output / "config.json", config.to_dict())
    write_json(output / "manifest.json", {
        "schema": "cipher-experiments-run-v1", "name": config.name,
        "started_utc": datetime.now(UTC).isoformat(), "config_sha256": config.sha256(),
        "source_sha256": source_identity(),
        "git": git_identity(Path(__file__).resolve().parents[2]),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "device": str(device), "cuda": torch.version.cuda},
        "timeout_policy": "cooperative between batches; not a hard process/GPU deadline",
        "test_policy": "TEST labels after selected load; no access-enforced sealed confirm",
        "label_mode": config.label_mode, "model_features": config.model_features,
        "model_kind": config.model_kind,
        "prediction_mode": config.prediction_mode,
        "train_pool_rows": config.train_pool_rows,
        "experiment_kind": "synthetic-engineering" if config.dataset in (
            "synthetic", "circuit-synthetic")
                           else "simon-structure-unknown-constants" if (
                               config.model_kind == "simon-structure")
                           else "simon-unrolled-final-bit-supervision" if (
                               config.model_kind == "simon-unrolled")
                           else "simon-r2-table-positive-control" if (
                               config.model_kind == "simon-r2-table")
                           else "simon-r2-local-neural-candidate" if (
                               config.model_kind == "simon-r2-local")
                           else "simon-r2-vector-neural-candidate" if (
                               config.model_kind == "simon-r2-vector")
                           else "simon-output-bit-baseline"})

    def deadline() -> None:
        if time.monotonic() - started > config.max_seconds:
            raise TimeoutError("configured cooperative training time budget exceeded")

    try:
        torch.manual_seed(config.init_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.init_seed)
        torch.set_num_threads(1)
        inputs = input_panels(config)
        panels = {split: (inputs[split], labels_for(config, inputs[split], split))
                  for split in ("train", "dev")}
        references = reference_panels(config, inputs)
        deadline()
        panel_sha: dict[str, Any] = {}
        import hashlib
        for split, x in inputs.items():
            panel_sha[split] = {"rows": len(x), "x_sha256": hashlib.sha256(
                x.contiguous().numpy().tobytes()).hexdigest()}
        write_json(output / "data.json", panel_sha)
        x_train, y_train = panels["train"]
        ones = int(y_train.sum())
        majority = int(2 * ones > len(y_train))
        probability = (ones + .5) / (len(y_train) + 1)
        residual_majority, residual_probability = baseline_parameters(
            y_train, references.get("train"))
        model = build_predictor(config.hidden_width, model_kind=config.model_kind,
                                rounds=config.rounds,
                                   target_bit=config.target_bit, feature_mode=config.model_features,
                                   prediction_mode=config.prediction_mode).to(device)
        initial_sha = hashlib.sha256(b"".join(
            name.encode() + value.detach().cpu().contiguous().numpy().tobytes()
            for name, value in model.state_dict().items())).hexdigest()
        write_json(output / "initialization.json", {
            "state_sha256": initial_sha,
            "trainable_sha256": hashlib.sha256(b"".join(
                name.encode() + value.detach().cpu().contiguous().numpy().tobytes()
                for name, value in model.named_parameters())).hexdigest(),
            "features": list(model.indices.cpu().tolist()),
            "parameters": sum(p.numel() for p in model.parameters())})
        if isinstance(model, StructuredSimonPredictor):
            write_json(output / "model_metadata.json", {
                "kind": "simon-structure", "rounds": config.rounds,
                "target_bit": config.target_bit,
                "unknowns": "independent learned round-constant effects; not true key input",
                "initialization": "random signs and noise from init_seed, no data/key access",
                "key_schedule_constraint": False,
                "selection": "soft absolute DEV BCE; hard constants report-only",
                "gate_counts": model.cone.gate_counts,
                "symbolic_coordinates": list(model.cone.round_key_coordinates)})
        if isinstance(model, (TwoRoundTablePredictor, TwoRoundLocalPredictor,
                              TwoRoundVectorPredictor)):
            is_table = isinstance(model, TwoRoundTablePredictor)
            is_vector = isinstance(model, TwoRoundVectorPredictor)
            metadata: dict[str, Any] = {
                "kind": config.model_kind,
                "role": "positive-control" if is_table else "neural-candidate",
                "rounds": 2, "target_bit": config.target_bit,
                "supervision": "absolute final target-bit BCE only",
                "key_free": True,
                "input": "public plaintext bits only; no key or intermediate states; "
                         "final absolute bit labels only",
                "selection": "absolute DEV BCE",
            }
            if isinstance(model, TwoRoundTablePredictor):
                with torch.no_grad():
                    cells, public_t = model.public_table_features(x_train.to(device))
                    residual = torch.logical_xor(
                        y_train.to(device).bool().squeeze(1), public_t.bool().squeeze(1)
                    )
                    cell_counts: list[int] = []
                    residual_ones: list[int] = []
                    conflicts: list[bool] = []
                    for cell in range(4):
                        cell_mask = cells == cell
                        count = int(cell_mask.sum())
                        ones = int((residual & cell_mask).sum())
                        cell_counts.append(count)
                        residual_ones.append(ones)
                        conflicts.append(ones not in (0, count))
                covered = sum(count > 0 for count in cell_counts)
                metadata.update({
                    "parameters": 4,
                    "formula": "U=Y XOR F(X); A=U[j+1], B=U[j+8], "
                               "T=X[j] XOR U[j+2]; logit=(1-2T)*table_logits[2A+B]",
                    "train_table": {
                        "counts": cell_counts,
                        "residual_ones": residual_ones,
                        "covered_all_cells": covered == 4,
                        "conflicting_cells": [cell for cell, conflict in enumerate(conflicts)
                                              if conflict],
                    },
                })
            elif is_vector:
                metadata.update({
                    "parameters": 557,
                    "modules": "three independent 4->16->4 local modules with Tanh hidden "
                               "activation only, then 12->8->1 with Tanh hidden activation",
                    "local_features": "centered [P_Y[q],P_X[q+1],P_X[q+8],P_X[q+2]], "
                                     "q=j+1,j+8,j+2",
                    "hidden_channels": "sigmoid per local logit then center; concatenate 12",
                    "output_sign": "(1-2*P_X[j])",
                    "parameter_sharing": "independent local module weights; no sharing",
                })
            else:
                metadata.update({
                    "parameters": 332,
                    "modules": "three independent 4->16->1 Tanh MLPs, then 3->8->1 Tanh",
                    "local_features": "centered [P_Y[q],P_X[q+1],P_X[q+8],P_X[q+2]], "
                                     "q=j+1,j+8,j+2",
                    "output_sign": "(1-2*P_X[j])",
                    "parameter_sharing": "independent local module weights; no sharing",
                })
            write_json(output / "model_metadata.json", metadata)
        if isinstance(model, UnrolledSimonPredictor):
            write_json(output / "model_metadata.json", {
                "kind": "simon-unrolled", "rounds": config.rounds,
                "target_bit": config.target_bit, "prediction_mode": config.prediction_mode,
                "supervision": "final-only absolute target-bit BCE; no real intermediate states",
                "public_reference": (
                    "plaintext target bit at target_bit in the high half (indices 0..31); "
                    "sign factor (1-2*x[target_bit])"
                    if config.prediction_mode in (
                        "plaintext-xor", "plaintext-xor-no-final-low")
                    else "none (direct mode)"),
                "final_low_input": (
                    "zero after centering for every position in the final round only"
                    if config.prediction_mode == "plaintext-xor-no-final-low"
                    else "centered local low-state value in every round"),
                "input": "plaintext only; no true key or round-key input",
                "public_prior": "Feistel copy and rotations 1,8,2; local neural update",
                "state": "continuous latent scalars, not verified cipher round states",
                "cell": "shared 8->hidden_width->1 Tanh MLP, sigmoid hidden updates",
                "conditions": "learned [round,position,4], randomly initialized",
                "key_schedule_constraint": False,
                "selection": "absolute DEV BCE; no discrete parameter search"})
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        shuffle = torch.Generator(device="cpu").manual_seed(config.data_seed + 1)
        best = math.inf
        selected_epoch = 0
        for epoch in range(1, config.epochs + 1):
            current_epoch = epoch
            model.train()
            order = torch.randperm(len(y_train), generator=shuffle)
            loss_sum = 0.0
            for offset in range(0, len(order), config.batch_size):
                deadline()
                indices = order[offset:offset + config.batch_size]
                x, y = x_train[indices].to(device), y_train[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite training loss")
                loss.backward()  # type: ignore[no-untyped-call]
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                presentations += len(indices)
                loss_sum += float(loss.detach()) * len(indices)
            deadline()
            dev_evaluated = (epoch % config.dev_every_epochs == 0 or epoch == config.epochs)
            dev = (evaluate_model(model, *panels["dev"], device=device,
                                  batch_size=config.batch_size, majority=majority,
                                  probability=probability, reference=references.get("dev"),
                                  residual_majority=residual_majority,
                                  residual_probability=residual_probability)
                   if dev_evaluated else None)
            if dev is not None and dev["bce"] < best:
                best, selected_epoch = dev["bce"], epoch
                _save_checkpoint(output / "best.pt", model, config, epoch, updates,
                                 majority, probability)
            record = {"epoch": epoch, "updates": updates, "presentations": presentations,
                      "train_bce": loss_sum / len(y_train), "dev": dev,
                      "dev_evaluated": dev_evaluated,
                      "elapsed_seconds": time.monotonic() - started}
            append_progress(output / "progress.jsonl", record)
            message = (f"epoch={epoch}/{config.epochs} updates={updates} "
                       + (f"dev_bce={dev['bce']:.6f}" if dev is not None
                          else "dev=not evaluated (no BCE)"))
            with (output / "training.log").open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
            print(message, flush=True)
        deadline()
        _save_checkpoint(output / "last.pt", model, config, current_epoch, updates,
                         majority, probability)
        best_path = output / "best.pt"
        selected, _, _, _ = load_checkpoint(best_path, file_sha256(best_path), device=device)
        write_json(output / "selection.json", {
            "epoch": selected_epoch, "checkpoint_sha256": file_sha256(best_path),
            "test_labels_generated": False})
        if isinstance(selected, (TwoRoundLocalPredictor, TwoRoundVectorPredictor)):
            write_json(output / "diagnostics.json", two_round_local_diagnostics(
                selected, {split: panels[split] for split in ("train", "dev")},
                batch_size=config.batch_size, device=device))
        # No TEST labels have been requested by this training path before this point.
        panels["test"] = (inputs["test"], labels_for(config, inputs["test"], "test"))
        for split, (_, y) in panels.items():
            panel_sha[split]["y_sha256"] = hashlib.sha256(
                y.contiguous().numpy().tobytes()).hexdigest()
        write_json(output / "data.json", panel_sha)
        metrics = {split: evaluate_model(selected, *panels[split], device=device,
                                        batch_size=config.batch_size, majority=majority,
                                        probability=probability, reference=references.get(split),
                                        residual_majority=residual_majority,
                                        residual_probability=residual_probability)
                   for split in ("train", "dev", "test")}
        deadline()
        result = {"schema": "cipher-experiments-result-v1", "status": "completed",
                  "name": config.name, "dataset": config.dataset,
                  "config_sha256": config.sha256(), "updates": updates,
                  "model_features": config.model_features, "model_kind": config.model_kind,
                  "prediction_mode": config.prediction_mode,
                  "label_mode": config.label_mode, "initial_state_sha256": initial_sha,
                  "presentations": presentations, "selected_epoch": selected_epoch,
                  "metrics": metrics, "best_checkpoint": "best.pt",
                  "best_checkpoint_sha256": file_sha256(best_path),
                  "elapsed_seconds": time.monotonic() - started}
        write_json(output / "results.json", result)
        write_json(output / "exit.json", {"status": "completed", "exit_code": 0,
                                         "updates": updates, "presentations": presentations,
                                         "elapsed_seconds": time.monotonic() - started})
        return result
    except BaseException as error:
        import traceback
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(output / "exit.json", {"status": "failed", "exit_code": 1,
                                         "error": f"{type(error).__name__}: {error}",
                                         "epoch": current_epoch, "updates": updates,
                                         "presentations": presentations,
                                         "elapsed_seconds": time.monotonic() - started})
        raise

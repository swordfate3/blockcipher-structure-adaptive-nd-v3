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
from .models.structured_simon import StructuredSimonPredictor


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
    if config.dataset == "simon-diff":
        from .data.diffbit import make_diff_inputs
        return make_diff_inputs(config.train_rows, config.dev_rows, config.test_rows,
                                seed=config.data_seed)
    from .data.prediction import make_inputs
    capacity = config.train_rows if config.train_pool is None else config.train_pool
    panels = make_inputs(capacity, config.dev_rows, config.test_rows, seed=config.data_seed)
    if capacity != config.train_rows:
        # Keep DEV/TEST fixed while shrinking only the TRAIN budget.
        panels["train"] = panels["train"][: config.train_rows].contiguous()
    return panels


def labels_for(config: ExperimentConfig, x: Tensor, split: str) -> Tensor:
    if config.dataset == "circuit-synthetic":
        from .data.circuit import circuit_labels
        return circuit_labels(x)
    if config.dataset == "simon-diff":
        from .data.diffbit import make_diff_labels, make_null_labels
        if config.label_mode == "null":
            return make_null_labels(x, null_seed=config.null_seed, split=split)
        assert config.input_difference is not None
        assert config.train_key_count is not None
        return make_diff_labels(x, delta=config.input_difference, rounds=config.rounds,
                                target_bit=config.target_bit, key_seed=config.key_seed,
                                key_count=config.key_count,
                                train_key_count=config.train_key_count, split=split)
    from .data.prediction import make_labels
    return make_labels(x, dataset=config.dataset, rounds=config.rounds,
                       target_bit=config.target_bit,
                       key_seed=config.key_seed, label_mode=config.label_mode,
                       null_seed=config.null_seed, split=split)


def reference_panels(config: ExperimentConfig, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    if config.dataset == "simon-diff":
        from .data.diffbit import round_one_reference
        if config.rounds != 1 or config.input_difference is None:
            return {}
        return {s: round_one_reference(x, delta=config.input_difference,
                                       target_bit=config.target_bit)
                for s, x in inputs.items()}
    if config.dataset == "simon" and config.rounds <= 4:
        return {s: public_reference(x, config.rounds, config.target_bit) for s, x in inputs.items()}
    return {}


def key_marginal_panels(config: ExperimentConfig,
                        inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    """Per-row TRAIN-key majority probability; report-only, never used to select."""
    if config.dataset != "simon-diff" or config.label_mode != "cipher":
        return {}
    from .data.diffbit import key_marginal_probability
    if config.input_difference is None or config.train_key_count is None:
        return {}
    return {split: key_marginal_probability(
        x, delta=config.input_difference, rounds=config.rounds,
        target_bit=config.target_bit, key_seed=config.key_seed,
        train_key_count=config.train_key_count)
        for split, x in inputs.items()}


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
    if config.model_kind in ("simon-hard-search", "simon-hard-sat"):
        raise ValueError("hard search uses hard_training.train, not Adam training")
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
        "experiment_kind": "synthetic-engineering" if config.dataset in (
            "synthetic", "circuit-synthetic")
                           else "simon-ste-gradient-hard-forward" if (
                               config.model_kind == "simon-ste")
                           else "simon-structure-unknown-constants" if (
                               config.model_kind == "simon-structure")
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
        marginals = key_marginal_panels(config, inputs)
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
        order_seed = (config.data_seed + 1 if config.shuffle_seed is None
                      else config.shuffle_seed)
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
                "kind": config.model_kind, "rounds": config.rounds,
                "target_bit": config.target_bit,
                "unknowns": "independent learned round-constant effects; not true key input",
                "initialization": "random signs and noise from init_seed, no data/key access",
                "forward": "hard Boolean; logits +/-2" if config.model_kind == "simon-ste"
                           else "sigmoid soft constants and polynomial AND/XOR relaxation",
                "backward": "biased sigmoid straight-through at key leaves" if (
                    config.model_kind == "simon-ste") else "ordinary autograd",
                "parameter_search": False,
                "order_seed": order_seed,
                "order_seed_source": "shuffle_seed or data_seed+1",
                "score_floor_bce": (math.log1p(math.exp(-2.0))
                                    if config.model_kind == "simon-ste" else 0.0),
                "initial_key_logit_scale": 0.05 if config.model_kind == "simon-ste" else 1.0,
                "key_schedule_constraint": False,
                "selection": "hard-forward absolute DEV BCE" if config.model_kind == "simon-ste"
                             else "soft absolute DEV BCE; hard constants report-only",
                "gate_counts": model.cone.gate_counts,
                "symbolic_coordinates": list(model.cone.round_key_coordinates)})
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        shuffle = torch.Generator(device="cpu").manual_seed(order_seed)
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
            dev = evaluate_model(model, *panels["dev"], device=device,
                                 batch_size=config.batch_size, majority=majority,
                                 probability=probability, reference=references.get("dev"),
                                 residual_majority=residual_majority,
                                 residual_probability=residual_probability,
                                 key_marginal=marginals.get("dev"))
            if dev["bce"] < best:
                best, selected_epoch = dev["bce"], epoch
                _save_checkpoint(output / "best.pt", model, config, epoch, updates,
                                 majority, probability)
            record = {"epoch": epoch, "updates": updates, "presentations": presentations,
                      "train_bce": loss_sum / len(y_train), "dev": dev,
                      "elapsed_seconds": time.monotonic() - started}
            append_progress(output / "progress.jsonl", record)
            message = (f"epoch={epoch}/{config.epochs} updates={updates} "
                       f"dev_bce={dev['bce']:.6f}")
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
                                        residual_probability=residual_probability,
                                        key_marginal=marginals.get(split))
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

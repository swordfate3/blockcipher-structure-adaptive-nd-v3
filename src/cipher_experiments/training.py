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
from .checkpoints import load_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .evaluation import evaluate_model
from .models.prediction import build_predictor, public_reference
from .models.simon_two_round import TwoRoundLocalPredictor, TwoRoundVectorPredictor
from .prediction_reports import experiment_kind, model_metadata
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
    space_options: dict[str, Any] = {}
    if config.plaintext_space != "full":
        space_options["plaintext_space"] = config.plaintext_space
    if config.plaintext_fixed_value != 0:
        space_options["plaintext_fixed_value"] = config.plaintext_fixed_value
    return make_inputs(config.train_rows, config.dev_rows, config.test_rows,
                       seed=config.data_seed, train_pool_rows=config.train_pool_rows,
                       **space_options)


def labels_for(config: ExperimentConfig, x: Tensor, split: str) -> Tensor:
    if config.dataset == "circuit-synthetic":
        from .data.circuit import circuit_labels
        return circuit_labels(x)
    from .data.prediction import make_labels
    expression_options: dict[str, Any] = {}
    if config.label_expression != "bit":
        expression_options = {"label_expression": config.label_expression,
                              "label_bits": config.label_bits}
    return make_labels(x, dataset=config.dataset, rounds=config.rounds,
                       target_bit=config.target_bit,
                       key_seed=config.key_seed, label_mode=config.label_mode,
                       null_seed=config.null_seed, split=split, **expression_options)


def reference_panels(config: ExperimentConfig, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    if (config.dataset == "simon" and config.rounds <= 4
            and config.label_expression == "bit"):
        return {s: public_reference(x, config.rounds, config.target_bit) for s, x in inputs.items()}
    return {}


def baseline_parameters(labels: Tensor, reference: Tensor | None) -> tuple[int, float]:
    residual = labels if reference is None else (labels != reference).float()
    ones = int(residual.sum())
    return int(2 * ones > len(labels)), (ones + .5) / (len(labels) + 1)


def _input_semantics(config: ExperimentConfig) -> dict[str, Any]:
    """Describe the public SIMON plaintext panel for manifest and results."""
    # The circuit fixture is not a uniform uint64 panel.
    if config.dataset != "simon":
        return {}
    fixed_bits = (list(range(32)) if config.plaintext_space == "high32-zero"
                  else list(range(32, 64)) if config.plaintext_space == "low32-zero"
                  else [])
    return {
        "plaintext_space": config.plaintext_space,
        "active_plaintext_bits": 64 - len(fixed_bits),
        "fixed_plaintext_bits": fixed_bits,
        "fixed_plaintext_value": config.plaintext_fixed_value if fixed_bits else None,
        "plaintext_bit_order": "64-bit plaintext MSB=0, LSB=63",
        "plaintext_sampling": ("unique uniform uint64 draws; local random.Random(data_seed)"
                               if not fixed_bits else
                               "unique uniform uint32 draws; local random.Random(data_seed); "
                               "placed in active half, other half fixed to "
                               f"0x{config.plaintext_fixed_value:08X}"),
    }


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
    role = experiment_kind(config.dataset, config.model_kind)
    if config.label_expression != "bit":
        role = (f"simon-output-expression-{config.label_expression}-mlp-"
                + ("null-control" if config.label_mode == "null" else "baseline"))
    # Keep declaration fields separate from the dataset's actual label operands.
    label_semantics = {
        "label_expression": config.label_expression,
        "label_bits": list(config.label_bits),
        "effective_label_bits": list(config.label_bits or (config.target_bit,)),
        "label_bit_order": "64-bit ciphertext MSB=0, LSB=63",
        "label_source": ("independent uniform random bit; expression not evaluated"
                         if config.label_mode == "null" else "ciphertext expression"),
    }
    if config.dataset == "synthetic":
        label_semantics.update({
            "effective_label_bits": [0],
            "label_bit_order": "64-bit plaintext input MSB=0, LSB=63",
            "label_source": "first plaintext input bit: x[0]; target_bit ignored",
        })
    elif config.dataset == "circuit-synthetic":
        label_semantics.update({
            "effective_label_bits": list(range(8)),
            "label_bit_order": "input column indices 0..63; first eight bits are relevant",
            "label_source": ("XOR of four disjoint adjacent input ANDs: "
                             "(x[0] AND x[1]) XOR (x[2] AND x[3]) XOR "
                             "(x[4] AND x[5]) XOR (x[6] AND x[7]); target_bit ignored"),
        })
    input_semantics = _input_semantics(config)
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
        "experiment_kind": role, **label_semantics, **input_semantics})

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
        metadata = model_metadata(
            model, model_kind=config.model_kind, prediction_mode=config.prediction_mode,
            rounds=config.rounds, target_bit=config.target_bit,
            x_train=x_train, y_train=y_train, device=device)
        if metadata is not None:
            write_json(output / "model_metadata.json", metadata)
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
                save_checkpoint(output / "best.pt", model, config, epoch, updates,
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
        save_checkpoint(output / "last.pt", model, config, current_epoch, updates,
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
                  "experiment_kind": role, **label_semantics, **input_semantics,
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

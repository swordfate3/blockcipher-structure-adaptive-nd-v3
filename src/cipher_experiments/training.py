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
from .models.mlp import BitMLP


def device_for(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available() or device.index is None:
            raise ValueError(f"requested {name}, but CUDA is unavailable; no CPU fallback")
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"requested {name}, but only {torch.cuda.device_count()} GPUs visible")
    return device


def datasets(config: ExperimentConfig) -> dict[str, tuple[Tensor, Tensor]]:
    if config.dataset == "synthetic":
        from .data.synthetic import make_synthetic
        return make_synthetic(config.train_rows, config.dev_rows, config.test_rows,
                              seed=config.data_seed)
    from .data.simon import make_simon
    return make_simon(config.train_rows, config.dev_rows, config.test_rows,
                      seed=config.data_seed, key_seed=config.key_seed,
                      rounds=config.rounds, target_bit=config.target_bit)


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
        "test_policy": "prebuilt split; evaluated only after DEV selection, not sealed confirm",
        "experiment_kind": "synthetic-engineering" if config.dataset == "synthetic"
                           else "simon-output-bit-baseline"})

    def deadline() -> None:
        if time.monotonic() - started > config.max_seconds:
            raise TimeoutError("configured cooperative training time budget exceeded")

    try:
        torch.manual_seed(config.init_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.init_seed)
        torch.set_num_threads(1)
        panels = datasets(config)
        deadline()
        panel_sha: dict[str, Any] = {}
        for split, (x, y) in panels.items():
            import hashlib
            panel_sha[split] = {"rows": len(y), "x_sha256": hashlib.sha256(
                x.contiguous().numpy().tobytes()).hexdigest(), "y_sha256": hashlib.sha256(
                y.contiguous().numpy().tobytes()).hexdigest()}
        write_json(output / "data.json", panel_sha)
        x_train, y_train = panels["train"]
        ones = int(y_train.sum())
        majority = int(2 * ones > len(y_train))
        probability = (ones + .5) / (len(y_train) + 1)
        model = BitMLP(config.hidden_width).to(device)
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
            dev = evaluate_model(model, *panels["dev"], device=device,
                                 batch_size=config.batch_size, majority=majority,
                                 probability=probability)
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
        metrics = {split: evaluate_model(selected, *panels[split], device=device,
                                        batch_size=config.batch_size, majority=majority,
                                        probability=probability)
                   for split in ("train", "dev", "test")}
        deadline()
        result = {"schema": "cipher-experiments-result-v1", "status": "completed",
                  "name": config.name, "dataset": config.dataset,
                  "config_sha256": config.sha256(), "updates": updates,
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

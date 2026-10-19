"""Four-task shared-M experiment; ordinary current-machine execution only."""
from __future__ import annotations

import hashlib
import math
import platform
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from .artifacts import append_progress, file_sha256, git_identity, source_identity, write_json
from .config import ExperimentConfig
from .training import device_for

SCHEMA = "cipher-experiments-shared-m-v1"
TASK_NAMES = ("simeck64", "present80", "simon64", "skinny64")


def _panel_sha(bits: Tensor, labels: Tensor) -> dict[str, Any]:
    return {"groups": len(labels), "pairs_per_group": 4,
            "bits_sha256": hashlib.sha256(bits.numpy().tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels.numpy().tobytes()).hexdigest()}


@torch.no_grad()
def _evaluate(model: Any, panels: dict[str, Any], cards: dict[str, Tensor],
              config: ExperimentConfig, device: torch.device) -> dict[str, Any]:
    model.eval()
    result: dict[str, Any] = {}
    for name in TASK_NAMES:
        batch = panels[name]
        correct, loss = 0, 0.
        for start in range(0, len(batch.label), config.batch_size):
            labels = batch.label[start:start + config.batch_size].to(device)
            logits = model(batch.bits[start:start + config.batch_size].to(device), cards[name])
            loss += float(torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="sum"))
            correct += int(((logits > 0) == labels.bool()).sum())
        result[name] = {"groups": len(batch.label), "pairs_per_group": 4,
                        "accuracy": correct / len(batch.label), "bce": loss / len(batch.label),
                        "majority_accuracy": .5, "majority_bce": math.log(2)}
    return result


def _checkpoint(path: Path, model: Any, config: ExperimentConfig, epoch: int, updates: int
                ) -> None:
    torch.save({"schema": SCHEMA, "config": config.to_dict(), "config_sha256": config.sha256(),
                "epoch": epoch, "updates": updates,
                "state_dict": {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}},
               path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)


def train(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    from .ciphers.tasks import TaskSpec, public_features
    from .data.differential import generate_batch
    from .models.shared_m import ModelConfig, SharedDistinguisher

    if config.experiment != "innovation1":
        raise ValueError("shared-M requires innovation1 config")
    device = device_for(config.device)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    updates = 0
    presentations = 0
    counts = dict.fromkeys(TASK_NAMES, 0)
    try:
        write_json(output / "config.json", config.to_dict())
        torch.set_num_threads(1)
        torch.manual_seed(config.init_seed)
        tasks = (TaskSpec("simeck64", 16, 0x40), TaskSpec("present80", 7, 0xD00000),
                 TaskSpec("simon64", 13, 0x40), TaskSpec("skinny64", 8, 0x2000))
        model = SharedDistinguisher(ModelConfig()).to(device)
        write_json(output / "manifest.json", {
            "schema": SCHEMA, "git": git_identity(Path(__file__).resolve().parents[2]),
            "source_sha256": source_identity(), "config_sha256": config.sha256(),
            "started_utc": datetime.now(UTC).isoformat(), "experiment": "innovation1",
            "python": platform.python_version(), "torch": str(torch.__version__),
            "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
            "tasks": [{"cipher": t.name, "rounds": t.rounds,
                       "input_difference": hex(t.input_difference)} for t in tasks],
            "data_contract": "4pairs/group;onekey/group;negative independent plaintext encrypted",
            "scope": "fresh shared-M functional test, not old-recipe replication",
            "timeout": "cooperative only, no hard process termination",
            "test_policy": "prebuilt, scored only after DEV model selection"})

        def deadline() -> None:
            if time.monotonic() - began > config.max_seconds:
                raise TimeoutError("innovation1 cooperative time budget exceeded")

        panels: dict[str, dict[str, Any]] = {}
        cards = {t.name: torch.tensor(public_features(t.name), dtype=torch.float32, device=device)
                 for t in tasks}
        panel_ids: dict[str, Any] = {}
        for split_index, (split, rows) in enumerate((
                ("train", config.train_rows), ("dev", config.dev_rows),
                ("test", config.test_rows))):
            panels[split] = {}
            panel_ids[split] = {}
            for task_index, task in enumerate(tasks):
                seed = (config.data_seed + split_index * 100 + task_index) % (2**32)
                batch = generate_batch(task, groups=rows, seed=seed)
                panels[split][task.name] = batch
                panel_ids[split][task.name] = {"seed": seed,
                                             **_panel_sha(batch.bits, batch.label)}
                deadline()
        write_json(output / "data.json", panel_ids)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        best = math.inf
        selected_epoch = 0
        shuffle = torch.Generator().manual_seed(config.data_seed)
        for epoch in range(1, config.epochs + 1):
            model.train()
            # Every macro step updates each of the four tasks once; one joint state.
            order = torch.randperm(config.train_rows, generator=shuffle)
            for offset in range(0, config.train_rows, config.batch_size):
                for task in tasks:
                    deadline()
                    index = order[offset:offset + config.batch_size]
                    batch = panels["train"][task.name]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(batch.bits[index].to(device), cards[task.name])
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, batch.label[index].to(device))
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("nonfinite innovation1 loss")
                    loss.backward()  # type: ignore[no-untyped-call]
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                    counts[task.name] += 1
                    presentations += len(index)
            deadline()
            dev = _evaluate(model, panels["dev"], cards, config, device)
            dev_bce = sum(dev[t]["bce"] for t in TASK_NAMES) / 4
            if dev_bce < best:
                best, selected_epoch = dev_bce, epoch
                _checkpoint(output / "best.pt", model, config, epoch, updates)
            append_progress(output / "progress.jsonl", {
                "epoch": epoch, "updates": updates, "updates_by_task": counts,
                "presentations": presentations, "dev": dev,
                "elapsed_seconds": time.monotonic() - began})
            message = f"epoch={epoch}/{config.epochs} updates={updates} dev_bce={dev_bce:.6f}"
            print(message, flush=True)
            with (output / "training.log").open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        deadline()
        _checkpoint(output / "last.pt", model, config, config.epochs, updates)
        saved = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        if (saved["schema"] != SCHEMA or saved["config_sha256"] != config.sha256()
                or saved["epoch"] != selected_epoch):
            raise ValueError("shared-M selected checkpoint identity mismatch")
        _strict_state(model, saved["state_dict"])
        metrics = {split: _evaluate(model, panels[split], cards, config, device)
                   for split in ("train", "dev", "test")}
        deadline()
        result = {"schema": SCHEMA, "status": "completed", "experiment": "innovation1",
                  "config_sha256": config.sha256(), "updates": updates,
                  "updates_by_task": counts, "presentations": presentations,
                  "selected_epoch": selected_epoch, "metrics": metrics,
                  "best_checkpoint": "best.pt",
                  "best_checkpoint_sha256": file_sha256(output / "best.pt"),
                  "elapsed_seconds": time.monotonic() - began}
        write_json(output / "results.json", result)
        write_json(output / "exit.json", {"status": "completed", "exit_code": 0,
                                         "updates": updates, "updates_by_task": counts,
                                         "elapsed_seconds": time.monotonic() - began})
        return result
    except BaseException as error:
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(output / "exit.json", {"status": "failed", "exit_code": 1,
                                         "updates": updates, "updates_by_task": counts,
                                         "error": repr(error),
                                         "elapsed_seconds": time.monotonic() - began})
        raise


def _strict_state(model: Any, state: object) -> None:
    template = model.state_dict()
    if type(state) is not dict or set(state) != set(template):
        raise ValueError("shared-M checkpoint state names mismatch")
    for name, target in template.items():
        value = state[name]
        if (type(value) is not Tensor or value.shape != target.shape or value.dtype != target.dtype
                or value.layout != torch.strided or not bool(torch.isfinite(value).all())):
            raise ValueError(f"shared-M checkpoint tensor mismatch: {name}")
    model.load_state_dict(state, strict=True)


def evaluate_checkpoint(path: Path, sha256: str, device_name: str, output: Path) -> dict[str, Any]:
    """Explicit user-requested evaluation; not used during result retrieval."""
    from .ciphers.tasks import TaskSpec, public_features
    from .data.differential import generate_batch
    from .models.shared_m import ModelConfig, SharedDistinguisher

    if file_sha256(path) != sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("checkpoint exceeds 256 MiB")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if type(payload) is not dict or payload.keys() != {
            "schema", "config", "config_sha256", "epoch", "updates", "state_dict"}:
        raise ValueError("innovation1 checkpoint fields mismatch")
    if payload.get("schema") != SCHEMA:
        raise ValueError("not an innovation1 checkpoint")
    config = ExperimentConfig(**payload["config"])
    if config.sha256() != payload["config_sha256"] or config.experiment != "innovation1":
        raise ValueError("checkpoint config mismatch")
    device = device_for(device_name)
    model = SharedDistinguisher(ModelConfig()).to(device)
    _strict_state(model, payload["state_dict"])
    tasks = (TaskSpec("simeck64", 16, 0x40), TaskSpec("present80", 7, 0xD00000),
             TaskSpec("simon64", 13, 0x40), TaskSpec("skinny64", 8, 0x2000))
    panels = {t.name: generate_batch(t, groups=config.test_rows,
              seed=(config.data_seed + 200 + i) % (2**32)) for i, t in enumerate(tasks)}
    cards = {t.name: torch.tensor(public_features(t.name), dtype=torch.float32, device=device)
             for t in tasks}
    metrics = _evaluate(model, panels, cards, config, device)
    output.mkdir(parents=True, exist_ok=False)
    result = {"checkpoint_sha256": sha256, "experiment": "innovation1", "metrics": metrics}
    write_json(output / "evaluation.json", result)
    return cast(dict[str, Any], result)

"""Public assembly and ONE artificial 2-update/8-presentation fit per invocation.

No scientific labels: both data seams are replaced and production encryption fails.
Gradient probes use autograd.grad only; they do not add optimizer steps.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from cipher_experiments import training
from cipher_experiments.checkpoints import load_checkpoint
from cipher_experiments.config import ExperimentConfig
from cipher_experiments.data import prediction as prediction_data
from cipher_experiments.models.prediction import build_predictor
from cipher_experiments.models.public_prediction import PublicPredictor


def config(**changes: Any) -> ExperimentConfig:
    options: dict[str, Any] = dict(name="fake-public", dataset="simon", rounds=6,
                                   hidden_width=128, model_kind="public-mlp", label_mode="null",
                                   train_rows=8, dev_rows=4, test_rows=4, batch_size=4, epochs=1)
    options.update(changes)
    return ExperimentConfig(**options)


def artificial_panels() -> dict[str, Tensor]:
    return {split: torch.tensor([[int(((row + shift) * 19 + col * 7) % 23 < 11)
                                 for col in range(64)] for row in range(n)]).float()
            for split, n, shift in (("train", 8, 0), ("dev", 4, 10), ("test", 4, 20))}


def forbid_encrypt(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("Task3 production encryption forbidden")


class AssemblyComplete(Exception):
    """Stop the assembly probe before optimizer construction or any fit."""


@pytest.mark.parametrize("changes", [
    {}, {"public_input": "simon-interactions"},
    {"public_readout": "zero-round-constant-residual"},
    {"public_input": "simon-interactions", "public_readout": "zero-round-constant-residual"},
    {"rounds": 3, "public_input": "background-zero", "plaintext_space": "high32-zero",
     "plaintext_fixed_value": 0xA5A5A5A5, "label_expression": "xor2", "label_bits": (0, 8)},
    {"rounds": 3, "public_input": "background-zero", "plaintext_space": "high32-zero",
     "plaintext_fixed_value": 0xA5A5A5A5, "label_expression": "xor2", "label_bits": (0, 8),
     "background_init": "match-direct"},
    {"model_kind": "public-linear", "public_input": "simon-interactions",
     "public_readout": "zero-round-constant-residual", "l1_penalty": 1e-4},
])
def test_training_assembles_public_contract_before_optimizer_without_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any],
) -> None:
    # Missing forwarding must fail at the real first consumer, not merely a mock call.
    cfg = config(**changes)
    monkeypatch.setattr(prediction_data, "_encrypt", forbid_encrypt)
    monkeypatch.setattr(training, "input_panels", lambda _: artificial_panels())
    monkeypatch.setattr(training, "labels_for", lambda c, x, s: x[:, :1].clone())

    def inspect_model(*args: Any, **kwargs: Any) -> Any:
        model = build_predictor(*args, **kwargs)
        assert isinstance(model, PublicPredictor)
        for field in ("public_input", "public_readout", "background_init", "l1_penalty",
                      "plaintext_space", "plaintext_fixed_value"):
            assert getattr(model, field) == getattr(cfg, field)
        raise AssemblyComplete

    monkeypatch.setattr(training, "build_predictor", inspect_model)
    out = tmp_path / "assembly-only"
    with pytest.raises(AssemblyComplete):
        training.train(cfg, out)
    exit_record = json.loads((out / "exit.json").read_text())
    assert (exit_record["updates"], exit_record["presentations"]) == (0, 0)


def test_public_linear_fake_fit_uses_l1_objective_absolute_targets_and_strict_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Removing L1 from backward/clip, overwriting BCE logs, altering y or moving
    # TEST before selected strictload must each fail this real Adam integration.
    cfg = config(model_kind="public-linear", public_input="simon-interactions",
                 public_readout="zero-round-constant-residual", l1_penalty=1e-4)
    panels = artificial_panels()
    labels = {split: torch.tensor([[float((row + len(split)) % 3 == 0)]
                                   for row in range(len(x))]) for split, x in panels.items()}
    original_labels = {split: y.clone() for split, y in labels.items()}
    events: list[str] = []
    models: list[PublicPredictor] = []
    batches: list[dict[str, Any]] = []
    counts = {"updates": 0, "presentations": 0}
    out = tmp_path / "fake-fit"
    monkeypatch.setattr(prediction_data, "_encrypt", forbid_encrypt)
    monkeypatch.setattr(training, "input_panels", lambda _: panels)

    def fake_labels(c: ExperimentConfig, x: Tensor, split: str) -> Tensor:
        assert c == cfg and x is panels[split]
        if split == "test":
            assert events[-1] == "strict-loaded"
            selection = json.loads((out / "selection.json").read_text())
            assert selection["test_labels_generated"] is False
        events.append(split)
        return labels[split]

    def tracked_factory(*args: Any, **kwargs: Any) -> Any:
        model = build_predictor(*args, **kwargs)
        assert isinstance(model, PublicPredictor)
        models.append(model)
        return model

    real_bce = nn.functional.binary_cross_entropy_with_logits

    def tracked_bce(logits: Tensor, target: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        loss = real_bce(logits, target, *args, **kwargs)
        if logits.requires_grad:
            model = models[0]
            params = list(model.parameters())
            # Literal shuffled original labels: never use q or relabel residual targets.
            generator = torch.Generator().manual_seed(cfg.data_seed + 1)
            order = torch.randperm(8, generator=generator)
            offset = len(batches) * 4
            assert torch.equal(target, original_labels["train"][order[offset:offset + 4]])
            bce_grads = torch.autograd.grad(loss, params, retain_graph=True)
            penalty = 1e-4 * params[0].detach().abs().sum()
            expected_grads = [bce_grads[0] + 1e-4 * params[0].detach().sign(), bce_grads[1]]
            batches.append(dict(bce=float(loss.detach()), penalty=float(penalty),
                                grads=expected_grads, rows=len(target)))
        return loss

    real_backward = Tensor.backward

    def tracked_backward(tensor: Tensor, *args: Any, **kwargs: Any) -> None:
        batch = batches[-1]
        assert float(tensor.detach()) == pytest.approx(batch["bce"] + batch["penalty"], abs=1e-7)
        batch["backward"] = True
        real_backward(tensor, *args, **kwargs)  # type: ignore[no-untyped-call]

    real_clip = nn.utils.clip_grad_norm_

    def tracked_clip(parameters: Any, *args: Any, **kwargs: Any) -> Tensor:
        params = list(parameters)
        batch = batches[-1]
        assert batch["backward"]
        for p, expected in zip(params, batch["grads"], strict=True):
            torch.testing.assert_close(p.grad, expected, atol=1e-7, rtol=1e-5)
        assert args == (1.0,) and kwargs == {"error_if_nonfinite": True}
        norm = real_clip(params, *args, **kwargs)
        batch["clipped_grads"] = [p.grad.clone() for p in params]
        return norm

    real_step = torch.optim.Adam.step

    def tracked_step(optimizer: torch.optim.Adam, *args: Any, **kwargs: Any) -> Any:
        params = list(models[0].parameters())
        batch = batches[-1]
        before = [p.detach().clone() for p in params]
        for p, expected in zip(params, batch["clipped_grads"], strict=True):
            assert p.grad is not None
            assert torch.equal(p.grad, expected)
        result = real_step(optimizer, *args, **kwargs)
        counts["updates"] += 1
        counts["presentations"] += batch["rows"]
        assert any(not torch.equal(p, old) for p, old in zip(params, before, strict=True))
        assert all(int(optimizer.state[p]["step"]) == counts["updates"] for p in params)
        print(f"TASK3_ACTUAL updates={counts['updates']} presentations={counts['presentations']}")
        return result

    real_load = load_checkpoint

    def tracked_load(*args: Any, **kwargs: Any) -> Any:
        assert events == ["train", "dev"]
        loaded, restored, majority, probability = real_load(*args, **kwargs)
        assert isinstance(loaded, PublicPredictor) and restored == cfg
        assert loaded.checkpoint_config_sha256 == cfg.sha256()
        assert loaded.public_input == "simon-interactions"
        assert loaded.public_readout == "zero-round-constant-residual"
        events.append("strict-loaded")
        return loaded, restored, majority, probability

    monkeypatch.setattr(training, "labels_for", fake_labels)
    monkeypatch.setattr(training, "build_predictor", tracked_factory)
    monkeypatch.setattr(nn.functional, "binary_cross_entropy_with_logits", tracked_bce)
    monkeypatch.setattr(Tensor, "backward", tracked_backward)
    monkeypatch.setattr(nn.utils, "clip_grad_norm_", tracked_clip)
    monkeypatch.setattr(torch.optim.Adam, "step", tracked_step)
    monkeypatch.setattr(training, "load_checkpoint", tracked_load)
    try:
        result = training.train(cfg, out)
        assert counts == {"updates": 2, "presentations": 8}
        assert (result["updates"], result["presentations"]) == (2, 8)
        assert events == ["train", "dev", "strict-loaded", "test"]
        assert len(batches) == 2
        progress = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]
        assert len(progress) == 1
        record = progress[0]
        assert record["train_bce"] == pytest.approx(sum(b["bce"] * b["rows"] for b in batches) / 8)
        assert record["train_objective"] == pytest.approx(
            sum((b["bce"] + b["penalty"]) * b["rows"] for b in batches) / 8)
        assert record["train_objective"] > record["train_bce"]
        for split, y in labels.items():
            assert torch.equal(y, original_labels[split])
        exit_record = json.loads((out / "exit.json").read_text())
        assert (exit_record["status"], exit_record["updates"], exit_record["presentations"]) == (
            "completed", 2, 8)
    finally:
        print(f"TASK3_FINAL updates={counts['updates']} presentations={counts['presentations']}")
        if (out / "exit.json").exists():
            print("TASK3_EXIT " + (out / "exit.json").read_text())

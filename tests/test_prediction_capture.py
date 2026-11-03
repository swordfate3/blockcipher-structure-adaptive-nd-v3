"""Task4 artificial evidence: no cipher, at most ONE 2/8 fake fit per invocation."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from cipher_experiments import training
from cipher_experiments.checkpoints import load_checkpoint
from cipher_experiments.config import ExperimentConfig
from cipher_experiments.data import prediction as prediction_data
from cipher_experiments.evaluation import evaluate_model
from cipher_experiments.models.prediction import build_predictor
from cipher_experiments.prediction_evidence import PredictionCapture
from cipher_experiments.representations.prediction_public import zero_round_constant_reference


class CountingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: Tensor) -> Tensor:
        self.calls += 1
        return x[:, :1]


def evaluate(model: nn.Module, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
    return evaluate_model(model, x, y, device=torch.device("cpu"), batch_size=2,
                          majority=0, probability=.4, **kwargs)


def test_capture_copies_detaches_noncontiguous_batches_in_original_order() -> None:
    capture = PredictionCapture(3)
    logits = torch.tensor([[1., 9.], [-2., 9.]], requires_grad=True)[:, :1]
    targets = torch.tensor([[1., 9.], [0., 9.]])[:, :1]
    capture.append(logits, targets)
    with torch.no_grad():
        logits.fill_(99)
        targets.fill_(1)
    capture.append(torch.tensor([[0.]]), torch.tensor([[1.]]))
    z, y = capture.finish()
    np.testing.assert_array_equal(z, [1., -2., 0.])
    np.testing.assert_array_equal(y, [1, 0, 1])
    assert z.dtype == np.float32 and y.dtype == np.uint8
    assert z.shape == y.shape == (3,)
    assert z.flags.c_contiguous and y.flags.c_contiguous
    assert not np.shares_memory(z, y)
    with pytest.raises(RuntimeError):
        capture.finish()
    with pytest.raises(RuntimeError):
        capture.append(torch.ones(1, 1), torch.ones(1, 1))


@pytest.mark.parametrize("rows", [True, 0, -1, 1.0, np.int64(2)])
def test_capture_rejects_invalid_expected_rows(rows: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        PredictionCapture(rows)


@pytest.mark.parametrize("logits,targets", [
    (torch.zeros(0, 1), torch.zeros(0, 1)),
    (torch.zeros(2), torch.zeros(2)),
    (torch.zeros(2, 2), torch.zeros(2, 2)),
    (torch.zeros(2, 1), torch.zeros(1, 1)),
    (torch.zeros(2, 1, dtype=torch.float64), torch.zeros(2, 1)),
    (torch.zeros(2, 1), torch.zeros(2, 1, dtype=torch.uint8)),
    (torch.zeros(2, 1), torch.zeros(2, 1, dtype=torch.bool)),
    (torch.tensor([[float("nan")]]), torch.zeros(1, 1)),
    (torch.tensor([[float("inf")]]), torch.zeros(1, 1)),
    (torch.zeros(1, 1), torch.tensor([[float("nan")]])),
    (torch.zeros(1, 1), torch.tensor([[.5]])),
    (torch.zeros(1, 1), torch.tensor([[2.]])),
    (torch.zeros(1, 1), torch.tensor([[-1.]])),
    (np.zeros((1, 1), dtype=np.float32), torch.zeros(1, 1)),
    (torch.zeros(1, 1), np.zeros((1, 1), dtype=np.float32)),
])
def test_capture_invalid_append_does_not_consume_rows(logits: Any, targets: Any) -> None:
    capture = PredictionCapture(2)
    with pytest.raises((TypeError, ValueError)):
        capture.append(logits, targets)
    capture.append(torch.zeros(2, 1), torch.ones(2, 1))
    np.testing.assert_array_equal(capture.finish()[1], [1, 1])


def test_capture_rejects_empty_underfilled_and_overflow() -> None:
    capture = PredictionCapture(2)
    with pytest.raises(ValueError):
        capture.finish()
    capture.append(torch.ones(1, 1), torch.ones(1, 1))
    with pytest.raises(ValueError):
        capture.finish()
    with pytest.raises(ValueError):
        capture.append(torch.zeros(2, 1), torch.zeros(2, 1))
    capture.append(torch.zeros(1, 1), torch.zeros(1, 1))
    np.testing.assert_array_equal(capture.finish()[0], [1., 0.])


def test_evaluation_capture_uses_same_forward_and_exact_old_metrics() -> None:
    x = torch.tensor([[0.], [2.], [-3.], [1.], [-.5]])
    y = torch.tensor([[0.], [1.], [1.], [0.], [0.]])
    ref = torch.tensor([[1.], [0.], [1.], [0.], [1.]])
    plain, captured = CountingModel(), CountingModel()
    old = evaluate(plain, x, y, reference=ref, residual_majority=1, residual_probability=.3)
    capture = PredictionCapture(5)
    new = evaluate(captured, x, y, reference=ref, residual_majority=1,
                   residual_probability=.3, capture=capture)
    assert new == old
    assert plain.calls == captured.calls == 3
    assert set(old) == {"rows", "correct", "accuracy", "bce", "train_constant",
                        "public_reference_constant"}
    assert old["correct"] == 3  # logit == 0 predicts zero
    assert old["bce"] == pytest.approx(sum(
        math.log1p(math.exp(float(z))) - float(t) * float(z)
        for z, t in zip(x[:, 0], y[:, 0], strict=True)) / 5, abs=1e-7)
    z, targets = capture.finish()
    np.testing.assert_array_equal(z, x[:, 0].numpy())
    np.testing.assert_array_equal(targets, y[:, 0].numpy())


@pytest.mark.parametrize("majority,probability", [(0, .5), (0, .2), (1, .7)])
def test_q_baseline_manual_accuracy_probability_and_legacy_independence(
    majority: int, probability: float,
) -> None:
    x = torch.tensor([[0.], [1.], [-1.], [2.]])
    y = torch.tensor([[1.], [0.], [1.], [1.]])
    q = torch.tensor([[1.], [0.], [0.], [1.]])
    model = CountingModel()
    old = evaluate(CountingModel(), x, y, reference=1 - q,
                   residual_majority=1, residual_probability=.3)
    result = evaluate(model, x, y, reference=1 - q, residual_majority=1,
                      residual_probability=.3, zero_constant_reference=q,
                      zero_constant_majority=majority, zero_constant_probability=probability)
    baseline = result.pop("zero_round_constant_reference_constant")
    assert result == old and model.calls == 2
    predictions = [int(v) ^ majority for v in q[:, 0]]
    probabilities = [1 - probability if int(v) else probability for v in q[:, 0]]
    assert baseline == {"residual_bit": majority, "residual_probability": probability,
                        "accuracy": sum(p == int(t) for p, t in zip(
                            predictions, y[:, 0], strict=True)) / 4,
                        "bce": pytest.approx(-sum(
                            float(t) * math.log(p) + (1 - float(t)) * math.log1p(-p)
                            for p, t in zip(probabilities, y[:, 0], strict=True)) / 4)}
    if probability == .5:
        assert baseline["accuracy"] == .75  # tie: predictions are q, NOT all zeros


@pytest.mark.parametrize("reference", [
    torch.zeros(4), torch.zeros(4, 2), torch.zeros(3, 1),
    torch.zeros(4, 1, dtype=torch.float64), torch.zeros(4, 1, dtype=torch.bool),
    torch.full((4, 1), .5), torch.full((4, 1), float("nan")),
    torch.full((4, 1), float("inf")), np.zeros((4, 1), dtype=np.float32),
])
def test_q_rejects_invalid_reference_before_model_forward(reference: Any) -> None:
    model = CountingModel()
    with pytest.raises((TypeError, ValueError)):
        evaluate(model, torch.zeros(4, 1), torch.zeros(4, 1), zero_constant_reference=reference)
    assert model.calls == 0


@pytest.mark.parametrize("kwargs", [
    {"zero_constant_majority": True}, {"zero_constant_majority": 2},
    {"zero_constant_majority": 0.0}, {"zero_constant_majority": np.int64(0)},
    {"zero_constant_probability": float("nan")}, {"zero_constant_probability": float("inf")},
    {"zero_constant_probability": 0.}, {"zero_constant_probability": 1.},
    {"zero_constant_probability": -.1}, {"zero_constant_probability": 1.1},
    {"zero_constant_probability": True}, {"zero_constant_probability": np.float64(.5)},
])
def test_q_rejects_invalid_parameters_before_forward(kwargs: dict[str, Any]) -> None:
    model = CountingModel()
    with pytest.raises((TypeError, ValueError)):
        evaluate(model, torch.zeros(4, 1), torch.zeros(4, 1),
                 zero_constant_reference=torch.zeros(4, 1), **kwargs)
    assert model.calls == 0


# Only the final test below fits. Everything else is pure evaluation/assembly.
def fake_config(**changes: Any) -> ExperimentConfig:
    values: dict[str, Any] = dict(name="task4-artificial", dataset="simon", rounds=6,
                                 model_kind="public-mlp", hidden_width=128,
                                 label_mode="null", record_predictions=True,
                                 train_rows=8, dev_rows=4, test_rows=4, batch_size=4, epochs=1)
    values.update(changes)
    return ExperimentConfig(**values)


def artificial_panels() -> dict[str, Tensor]:
    return {split: torch.tensor([[float(((row + shift) * 19 + col * 7) % 23 < 11)
                                 for col in range(64)] for row in range(n)])
            for split, n, shift in (("train", 8, 0), ("dev", 4, 10), ("test", 4, 20))}


@pytest.mark.parametrize("changes,enabled", [
    ({}, True), ({"rounds": 8}, True), ({"rounds": 10}, True),
    ({"model_kind": "mlp", "public_input": "identity"}, True),
    ({"record_predictions": False}, False),
    ({"model_kind": "mlp", "public_input": "identity", "rounds": 5}, False),
    ({"model_kind": "mlp", "public_input": "identity", "target_bit": 1}, False),
    ({"model_kind": "mlp", "public_input": "identity", "label_expression": "xor2",
      "label_bits": (0, 8)}, False),
    ({"model_kind": "public-mlp", "public_input": "background-zero", "rounds": 3,
      "plaintext_space": "high32-zero", "label_expression": "xor2",
      "label_bits": (0, 8)}, False),
    ({"model_kind": "mlp", "public_input": "identity", "plaintext_space": "low32-zero"},
     False),
    ({"model_kind": "mlp", "public_input": "identity", "dataset": "synthetic",
      "label_mode": "cipher"}, False),
])
def test_q_panel_protocol_gate_without_fit(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any], enabled: bool,
) -> None:
    panels = artificial_panels()
    calls: list[str] = []

    def public_q(x: Tensor, *, rounds: int, target_bit: int) -> Tensor:
        calls.append(next(s for s, panel in panels.items() if panel is x))
        return x[:, :1].clone()

    monkeypatch.setattr(training, "zero_round_constant_reference", public_q)
    references = training._zero_reference_panels(fake_config(**changes), panels)
    assert calls == (["train", "dev", "test"] if enabled else [])
    assert set(references) == (set(panels) if enabled else set())


def test_selected_helper_default_no_capture_no_q_preserves_outputs() -> None:
    panels = {s: (torch.tensor([[0.], [1.]]), torch.tensor([[0.], [1.]]))
              for s in ("train", "dev", "test")}
    model = CountingModel()
    metrics = training._selected_metrics(
        model, panels, device=torch.device("cpu"), batch_size=2, majority=0,
        probability=.5, references={}, residual_majority=0, residual_probability=.5,
        zero_references={}, zero_majority=0, zero_probability=.5, captures={})
    assert model.calls == 3
    for result in metrics.values():
        assert set(result) == {"rows", "correct", "accuracy", "bce", "train_constant"}
        assert result["accuracy"] == 1


def test_public_linear_fake_fit_captures_only_selected_without_extra_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    cfg = fake_config(model_kind="public-linear", public_input="simon-interactions",
                      public_readout="zero-round-constant-residual", l1_penalty=1e-4)
    inputs = artificial_panels()
    labels = {s: torch.tensor([[float(row % 3 == 0)] for row in range(len(x))])
              for s, x in inputs.items()}
    out = tmp_path / "fake-fit"
    events: list[str] = []
    counts = {"updates": 0, "presentations": 0, "forwards": 0}
    q_calls: list[str] = []
    evaluations: list[tuple[str, bool, dict[str, Any]]] = []
    captured: dict[int, list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]] = {}
    finished: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = []
    real_factory = build_predictor
    real_load = load_checkpoint
    real_eval = evaluate_model
    real_step = torch.optim.Adam.step
    real_q = zero_round_constant_reference
    real_append = PredictionCapture.append
    real_finish = PredictionCapture.finish

    def forbid_encrypt(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Task4 production encryption forbidden")

    def fake_labels(c: ExperimentConfig, x: Tensor, split: str) -> Tensor:
        assert c == cfg and x is inputs[split]
        if split == "test":
            assert events == ["train", "dev", "strict-loaded"]
            assert counts["forwards"] == 3  # 2 train + 1 online DEV, no TEST forward
        events.append(split)
        return labels[split]

    def track_forward(module: nn.Module, args: Any, output: Any) -> None:
        counts["forwards"] += 1

    def factory(*args: Any, **kwargs: Any) -> Any:
        model = real_factory(*args, **kwargs)
        model.register_forward_hook(track_forward)
        return model

    def load(*args: Any, **kwargs: Any) -> Any:
        assert events == ["train", "dev"]
        loaded = real_load(*args, **kwargs)
        loaded[0].register_forward_hook(track_forward)
        events.append("strict-loaded")
        return loaded

    def step(optimizer: torch.optim.Adam, *args: Any, **kwargs: Any) -> Any:
        result = real_step(optimizer, *args, **kwargs)
        counts["updates"] += 1
        counts["presentations"] += 4
        print(f"TASK4_ACTUAL updates={counts['updates']} presentations={counts['presentations']}")
        return result

    def public_q(x: Tensor, **kwargs: Any) -> Tensor:
        q_calls.append(next(s for s, panel in inputs.items() if panel is x))
        return real_q(x, **kwargs)

    # Compute only public reference, no labels/optimizer/model call.
    train_q = real_q(inputs["train"], rounds=6, target_bit=0)
    residual_bit, residual_p = training.baseline_parameters(labels["train"], train_q)

    def eval_spy(model: nn.Module, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        split = next(s for s, panel in inputs.items() if panel is x)
        selected = "test" in events
        assert (kwargs.get("capture") is not None) == selected
        assert kwargs["zero_constant_majority"] == residual_bit
        assert kwargs["zero_constant_probability"] == residual_p
        assert torch.equal(kwargs["zero_constant_reference"], real_q(x, rounds=6, target_bit=0))
        assert y is labels[split]
        result = real_eval(model, x, y, **kwargs)
        evaluations.append((split, selected, result))
        return result

    def append(capture: PredictionCapture, logits: Tensor, targets: Tensor) -> None:
        assert events == ["train", "dev", "strict-loaded", "test"]
        captured.setdefault(id(capture), []).append(
            (logits.detach().numpy().copy(), targets.detach().numpy().copy()))
        real_append(capture, logits, targets)

    def finish(capture: PredictionCapture) -> Any:
        z, y = real_finish(capture)
        batches = captured[id(capture)]
        np.testing.assert_array_equal(z, np.concatenate([b[0][:, 0] for b in batches]))
        np.testing.assert_array_equal(y, np.concatenate([b[1][:, 0] for b in batches]))
        finished.append((z, y))
        return z, y

    monkeypatch.setattr(prediction_data, "_encrypt", forbid_encrypt)
    monkeypatch.setattr(training, "input_panels", lambda _: inputs)
    monkeypatch.setattr(training, "labels_for", fake_labels)
    monkeypatch.setattr(training, "build_predictor", factory)
    monkeypatch.setattr(training, "load_checkpoint", load)
    monkeypatch.setattr(training, "evaluate_model", eval_spy)
    monkeypatch.setattr(training, "zero_round_constant_reference", public_q)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    monkeypatch.setattr(PredictionCapture, "append", append)
    monkeypatch.setattr(PredictionCapture, "finish", finish)
    try:
        result = training.train(cfg, out)
        assert counts == {"updates": 2, "presentations": 8, "forwards": 7}
        assert events == ["train", "dev", "strict-loaded", "test"]
        assert q_calls == ["train", "dev", "test"]  # cached once, not every evaluation
        assert [(s, selected) for s, selected, _ in evaluations] == [
            ("dev", False), ("train", True), ("dev", True), ("test", True)]
        # Short fixture uses the same online + terminal counting equation as the 32+1 plan.
        expected_online = len([epoch for epoch in range(1, cfg.epochs + 1)
                               if epoch % cfg.dev_every_epochs == 0 or epoch == cfg.epochs])
        assert sum(s == "dev" and not selected for s, selected, _ in evaluations) == expected_online
        assert sum(s == "dev" for s, _, _ in evaluations) == expected_online + 1 == 2
        assert len(captured) == len(finished) == 3
        for (z, y), (split, target) in zip(finished, labels.items(), strict=True):
            assert len(z) == len(inputs[split])
            np.testing.assert_array_equal(y, target[:, 0].numpy())
            assert result["metrics"][split] == next(
                m for s, selected, m in evaluations if s == split and selected)
        assert (result["updates"], result["presentations"]) == (2, 8)
        saved = json.loads((out / "results.json").read_text())
        assert saved == result
        from cipher_experiments.artifacts import file_sha256
        from cipher_experiments.prediction_evidence import load_prediction_evidence

        assert set(result["prediction_evidence"]) == set(labels)
        assert len(list((out / "predictions").iterdir())) == 9
        for (z, y), split in zip(finished, labels, strict=True):
            index = result["prediction_evidence"][split]
            assert set(index) == {"manifest", "sha256", "fp64_bce_delta"}
            manifest = out / index["manifest"]
            assert index["sha256"] == file_sha256(manifest)
            assert abs(index["fp64_bce_delta"]) <= 1e-5
            evidence = load_prediction_evidence(manifest)
            np.testing.assert_array_equal(evidence.logits, z)
            np.testing.assert_array_equal(evidence.labels, y)
            np.testing.assert_array_equal(evidence.reference_bits, real_q(
                inputs[split], rounds=6, target_bit=0).numpy()[:, 0])
            assert evidence.config_sha256 == cfg.sha256()
            assert evidence.checkpoint_sha256 == result["best_checkpoint_sha256"]
            assert evidence.reference_majority == residual_bit
            assert evidence.reference_probability == residual_p
            assert (evidence.train_majority, evidence.train_probability) == (
                0, (int(labels["train"].sum()) + .5) / (len(labels["train"]) + 1))
            identities = json.loads((out / "data.json").read_text())[split]
            assert evidence.x_sha256 == identities["x_sha256"]
            assert evidence.y_sha256 == identities["y_sha256"]
    finally:
        print(f"TASK4_FINAL {counts}")
        if (out / "exit.json").exists():
            print("TASK4_EXIT " + (out / "exit.json").read_text())


def test_q_bce_preserves_probability_precision_without_clipping() -> None:
    # FP32 torch.where would round 1-p to 1 and turn BCE into PyTorch's 100 clamp.
    probability = 1e-12
    result = evaluate(CountingModel(), torch.zeros(2, 1), torch.tensor([[0.], [1.]]),
                      zero_constant_reference=torch.tensor([[1.], [0.]]),
                      zero_constant_probability=probability)
    bce = result["zero_round_constant_reference_constant"]["bce"]
    expected = -(math.log1p(-(1 - probability)) + math.log(probability)) / 2
    assert bce == pytest.approx(expected, abs=1e-12)


def test_train_residual_tie_majority_zero_uses_only_train_probability() -> None:
    train_q = torch.tensor([[1.], [1.], [0.], [0.]])
    train_y = torch.tensor([[1.], [0.], [0.], [1.]])
    bit, probability = training.baseline_parameters(train_y, train_q)
    assert (bit, probability) == (0, .5)
    held_y = torch.tensor([[1.], [1.]])  # not re-fitted, despite held residual imbalance
    result = evaluate(CountingModel(), torch.zeros(2, 1), held_y,
                      zero_constant_reference=torch.ones(2, 1),
                      zero_constant_majority=bit, zero_constant_probability=probability)
    assert result["zero_round_constant_reference_constant"]["accuracy"] == 1.
    assert result["zero_round_constant_reference_constant"]["bce"] == pytest.approx(math.log(2))


def test_selected_captures_other_protocol_without_q_or_extra_model_forward() -> None:
    panels = {s: (torch.tensor([[0.], [1.]]), torch.tensor([[0.], [1.]]))
              for s in ("train", "dev", "test")}
    captures = {s: PredictionCapture(2) for s in panels}
    model = CountingModel()
    metrics = training._selected_metrics(
        model, panels, device=torch.device("cpu"), batch_size=2, majority=0,
        probability=.5, references={}, residual_majority=0, residual_probability=.5,
        zero_references={}, zero_majority=0, zero_probability=.5, captures=captures)
    assert model.calls == 3
    for split, capture in captures.items():
        z, y = capture.finish()
        np.testing.assert_array_equal(z, [0., 1.])
        np.testing.assert_array_equal(y, [0, 1])
        assert "zero_round_constant_reference_constant" not in metrics[split]

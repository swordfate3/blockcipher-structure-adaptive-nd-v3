"""Bounded vector graph engineering: no target cipher labels or scientific fits."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from cipher_experiments.artifacts import file_sha256
from cipher_experiments.checkpoints import load_checkpoint
from cipher_experiments.cli import main
from cipher_experiments.config import ExperimentConfig
from cipher_experiments.models.prediction import build_predictor
from cipher_experiments.models.simon_vector import (
    SimonVectorPredictor,
    SymmetricVectorGate,
    bound_vector,
)
from cipher_experiments.representations.simon_cone import simon_cone
from cipher_experiments.representations.simon_vector import simon_vector_plan


def sequential_oracle(model: SimonVectorPredictor, x: Tensor) -> Tensor:
    """Independent per-node forward, no level groups/operand buffers/model gates."""
    x = x.to(dtype=model.constants.dtype)
    values: list[Tensor] = []
    cone = simon_cone(6, 0)
    key_ids = {key: i for i, key in enumerate(model.plan.symbolic_coordinates)}
    for node in cone.nodes:
        if node.kind == "plaintext":
            value = F.pad(1 - 2 * x[:, node.coordinate[0]:node.coordinate[0] + 1], (0, 31))
        elif node.kind == "round_key":
            coord = (node.coordinate[0], node.coordinate[1])
            value = model.constants[key_ids[coord]].expand(len(x), -1)
            value = value / torch.maximum(torch.ones_like(value[:, :1]),
                                          torch.sqrt((value * value).sum(1, keepdim=True)) / 2)
        else:
            a, b = (values[i] for i in node.args)
            gate = model.and_gate if node.kind == "and" else model.xor_gate
            linear = (a + b) @ gate.linear.weight.T
            product = (a * b) @ gate.product.weight.T
            mixed = ((a @ gate.project.weight.T) * (b @ gate.project.weight.T))
            raw = linear + product + mixed @ gate.mixed.weight.T + gate.bias
            value = raw / torch.maximum(torch.ones_like(raw[:, :1]),
                                        torch.sqrt((raw * raw).sum(1, keepdim=True)) / 2)
        values.append(value)
    return values[cone.output] @ model.readout.weight.T + model.readout.bias


def test_plan_reachable_identity_and_depth() -> None:
    p = simon_vector_plan()
    assert (p.node_count, p.depth, len(p.plaintext_indices), len(p.symbolic_coordinates)) == (
        406, 24, 56, 70)
    count = 126
    groups = []
    for g in p.groups:
        assert all(0 <= a < count and 0 <= b < count for a, b in g.operands)
        count += len(g.operands)
        groups.append(g.depth)
    assert count == 406 and groups == sorted(groups) and p.output < count
    assert len(set(p.plaintext_indices)) == 56 and len(set(p.symbolic_coordinates)) == 70


def test_bound_preserves_sign_witness_and_controls_norm() -> None:
    x = torch.tensor([[1., 0.], [-1., 0.], [0., 0.], [3., 4.]])
    y = bound_vector(x)
    assert torch.equal(y[:3], x[:3])
    torch.testing.assert_close(y[3], torch.tensor([1.2, 1.6]))
    assert bool((y.norm(dim=-1) <= 2.000001).all())
    assert torch.autograd.gradcheck(bound_vector, (torch.tensor(
        [[.3, .2], [3., 4.]], dtype=torch.float64, requires_grad=True),))


def test_exact_gate_truth_tables_remain_representable() -> None:
    gate = SymmetricVectorGate(32)
    a, b = torch.zeros(4, 32), torch.zeros(4, 32)
    a[:, 0] = torch.tensor([1., 1., -1., -1.])
    b[:, 0] = torch.tensor([1., -1., 1., -1.])
    for kind in ("and", "xor"):
        with torch.no_grad():
            for parameter in gate.parameters():
                parameter.zero_()
            if kind == "and":
                gate.linear.weight[0, 0] = .5
                gate.product.weight[0, 0] = -.5
                gate.bias[0] = .5
            else:
                gate.product.weight[0, 0] = 1.
        y = gate(a, b)
        truth = torch.tensor([1., 1., 1., -1.] if kind == "and"
                             else [1., -1., -1., 1.])
        assert torch.equal(y[:, 0], truth)
        assert torch.equal(y, gate(b, a))


@pytest.mark.parametrize("scale", [1., 10.])
def test_parallel_graph_matches_independent_sequential_forward_and_gradients(scale: float) -> None:
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = SimonVectorPredictor()
    if scale > 1:
        # Deep active projection paths amplify FP32 operation-order roundoff;
        # use FP64 for the independent derivative oracle, not looser tolerances.
        model.double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(scale)
    generator = torch.Generator().manual_seed(17)
    x = torch.randint(0, 2, (5, 64), generator=generator).float()
    expected = sequential_oracle(model, x)
    actual = model(x)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    a = torch.autograd.grad(actual.sum(), tuple(model.parameters()))
    b = torch.autograd.grad(expected.sum(), tuple(model.parameters()))
    for first, second in zip(a, b, strict=True):
        torch.testing.assert_close(first, second, atol=2e-7, rtol=2e-5)
    assert sum(p.numel() for p in model.parameters()) == 10529
    for gradient in a:
        assert torch.isfinite(gradient).all() and float(gradient.norm()) > 0
    torch.testing.assert_close(actual, torch.cat((model(x[:2]), model(x[2:]))))
    ignored = sorted(set(range(64)) - set(model.plan.plaintext_indices))
    changed = x.clone()
    changed[:, ignored] = 1 - changed[:, ignored]
    assert torch.equal(model(x), model(changed))


@pytest.mark.parametrize("patch", [
    {"dataset": "synthetic"}, {"rounds": 3}, {"rounds": True}, {"target_bit": 1},
    {"target_bit": False}, {"hidden_width": 16}, {"hidden_width": True},
    {"model_features": "support"}, {"prediction_mode": "public-coordinate"},
    {"experiment": "innovation1", "dataset": "differential"},
])
def test_configuration_rejects_out_of_scope(patch: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(**{"name": "bad", "dataset": "simon", "model_kind": "simon-vector",
                            "rounds": 6, "target_bit": 0, "hidden_width": 32, **patch})


def test_factory_modes_rejected() -> None:
    for features, mode in (("support", "direct"), ("full", "public-residual")):
        with pytest.raises(ValueError):
            build_predictor(model_kind="simon-vector", rounds=6,
                            feature_mode=features, prediction_mode=mode)


def test_cli_null_roundtrip_and_all_wiring_buffers_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_experiments import training
    from cipher_experiments.data import prediction

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no cipher labels in engineering")

    monkeypatch.setattr(prediction, "_encrypt", forbidden)
    config = ExperimentConfig(name="vector-null", dataset="simon", model_kind="simon-vector",
                              rounds=6, hidden_width=32, label_mode="null", train_rows=8,
                              dev_rows=4, test_rows=4, batch_size=4, epochs=1)
    path = tmp_path / "config.json"
    import json
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    out = tmp_path / "run"
    assert main(["check", "--config", str(path)]) == 0
    events: list[str] = []
    original_labels = training.labels_for

    def labels(c: ExperimentConfig, inputs: Tensor, split: str) -> Tensor:
        if split == "test":
            assert events[-1] == "loaded"
        events.append(split)
        return original_labels(c, inputs, split)

    def strict_load(*args: Any, **kwargs: Any) -> Any:
        loaded = load_checkpoint(*args, **kwargs)
        events.append("loaded")
        return loaded

    monkeypatch.setattr(training, "labels_for", labels)
    monkeypatch.setattr(training, "load_checkpoint", strict_load)
    assert main(["train", "--config", str(path), "--output", str(out)]) == 0
    assert events == ["train", "dev", "loaded", "test"]
    checkpoint = out / "best.pt"
    model, saved, _, _ = load_checkpoint(checkpoint, file_sha256(checkpoint),
                                        device=torch.device("cpu"))
    assert saved == config
    assert model(torch.zeros(1, 64)).shape == (1, 1)
    monkeypatch.setattr(training, "labels_for", original_labels)
    assert main(["evaluate", "--checkpoint", str(checkpoint), "--sha256",
                 file_sha256(checkpoint), "--output", str(tmp_path / "eval")]) == 0
    r = json.loads((out / "results.json").read_text(encoding="utf-8"))
    ev = json.loads((tmp_path / "eval/evaluation.json").read_text(encoding="utf-8"))
    assert r["metrics"]["test"] == ev["metrics"] and r["updates"] == 2
    assert r["presentations"] == 8
    payload = torch.load(checkpoint, weights_only=True)
    for name, buffer in model.named_buffers():
        changed = {k: v.clone() for k, v in payload["state_dict"].items()}
        changed[name] = buffer.detach().clone() + 1
        bad = tmp_path / "bad.pt"
        torch.save({**payload, "state_dict": changed}, bad)
        with pytest.raises(ValueError, match="noncanonical"):
            load_checkpoint(bad, file_sha256(bad), device=torch.device("cpu"))

"""Bounded vector graph engineering: no target cipher labels or scientific fits."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from cipher_experiments.artifacts import file_sha256
from cipher_experiments.checkpoints import load_checkpoint, save_checkpoint
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

# Verified against the unmodified simon_cone compiler: node/leaf/gate counts,
# levelized depth, and the closed-form parameter count
# 8d^2 + (symbolic_leaves + 3)d + 1 with d = hidden_width = 32.
PLAN_FACTS = {5: {"nodes": 266, "depth": 20, "plaintext": 46, "symbolic": 44,
                  "and": 44, "xor": 132, "groups": 20, "leaves": 90,
                  "parameters": 9697},
              6: {"nodes": 406, "depth": 24, "plaintext": 56, "symbolic": 70,
                  "and": 70, "xor": 210, "groups": 24, "leaves": 126,
                  "parameters": 10529}}


def sequential_oracle(model: SimonVectorPredictor, x: Tensor, rounds: int) -> Tensor:
    """Independent per-node forward, no level groups/operand buffers/model gates."""
    x = x.to(dtype=model.constants.dtype)
    values: list[Tensor] = []
    cone = simon_cone(rounds, 0)
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


@pytest.mark.parametrize("rounds", [5, 6])
def test_plan_reachable_identity_and_depth(rounds: int) -> None:
    facts = PLAN_FACTS[rounds]
    p = simon_vector_plan(rounds)
    assert (p.node_count, p.depth, len(p.plaintext_indices),
            len(p.symbolic_coordinates)) == (
        facts["nodes"], facts["depth"], facts["plaintext"], facts["symbolic"])
    count = facts["leaves"]
    groups = []
    gate_counts = {"and": 0, "xor": 0}
    for g in p.groups:
        assert all(0 <= a < count and 0 <= b < count for a, b in g.operands)
        count += len(g.operands)
        groups.append(g.depth)
        gate_counts[g.kind] += len(g.operands)
    assert count == p.node_count and groups == sorted(groups) and p.output < count
    assert len(set(p.plaintext_indices)) == facts["plaintext"]
    assert len(set(p.symbolic_coordinates)) == facts["symbolic"]
    assert len(p.groups) == facts["groups"]
    assert gate_counts == {"and": facts["and"], "xor": facts["xor"]}
    cone = simon_cone(rounds, 0)
    assert len(cone.nodes) == p.node_count and cone.gate_counts == {
        "AND": facts["and"], "XOR": facts["xor"]}


def test_plan_rejects_rounds_outside_five_and_six() -> None:
    for bad_rounds in (4, 7, 1, 44, 0):
        with pytest.raises(ValueError):
            simon_vector_plan(bad_rounds)
    for malformed in (True, False, 5.0, "5"):
        with pytest.raises(TypeError):
            simon_vector_plan(malformed)  # type: ignore[arg-type]
    assert simon_vector_plan().node_count == PLAN_FACTS[6]["nodes"]
    assert simon_vector_plan(6) == simon_vector_plan()
    assert simon_vector_plan(5).node_count == PLAN_FACTS[5]["nodes"]


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
@pytest.mark.parametrize("rounds", [5, 6])
def test_parallel_graph_matches_independent_sequential_forward_and_gradients(
        rounds: int, scale: float) -> None:
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = SimonVectorPredictor(rounds=rounds)
    if scale > 1:
        # Deep active projection paths amplify FP32 operation-order roundoff;
        # use FP64 for the independent derivative oracle, not looser tolerances.
        model.double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(scale)
    generator = torch.Generator().manual_seed(17)
    x = torch.randint(0, 2, (5, 64), generator=generator).float()
    expected = sequential_oracle(model, x, rounds)
    actual = model(x)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    a = torch.autograd.grad(actual.sum(), tuple(model.parameters()))
    b = torch.autograd.grad(expected.sum(), tuple(model.parameters()))
    for first, second in zip(a, b, strict=True):
        torch.testing.assert_close(first, second, atol=2e-7, rtol=2e-5)
    assert sum(p.numel() for p in model.parameters()) == PLAN_FACTS[rounds]["parameters"]
    for gradient in a:
        assert torch.isfinite(gradient).all() and float(gradient.norm()) > 0
    torch.testing.assert_close(actual, torch.cat((model(x[:2]), model(x[2:]))))
    ignored = sorted(set(range(64)) - set(model.plan.plaintext_indices))
    changed = x.clone()
    changed[:, ignored] = 1 - changed[:, ignored]
    assert torch.equal(model(x), model(changed))


def test_model_rejects_non_five_or_six_rounds() -> None:
    for bad in (4, 7, 1, 0):
        with pytest.raises(ValueError):
            SimonVectorPredictor(rounds=bad)
    with pytest.raises(TypeError):
        SimonVectorPredictor(rounds=True)
    with pytest.raises(TypeError):
        build_predictor(32, model_kind="simon-vector", rounds=True)


@pytest.mark.parametrize("patch", [
    {"dataset": "synthetic"}, {"rounds": 3}, {"rounds": 4}, {"rounds": 7},
    {"rounds": True}, {"target_bit": 1},
    {"target_bit": False}, {"hidden_width": 16}, {"hidden_width": True},
    {"model_features": "support"}, {"prediction_mode": "public-coordinate"},
    {"experiment": "innovation1", "dataset": "differential"},
])
def test_configuration_rejects_out_of_scope(patch: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(**{"name": "bad", "dataset": "simon", "model_kind": "simon-vector",
                            "rounds": 6, "target_bit": 0, "hidden_width": 32, **patch})


def test_configuration_accepts_exactly_five_or_six_rounds() -> None:
    for rounds in (5, 6):
        config = ExperimentConfig(name=f"vector-null-r{rounds}", dataset="simon",
                                  model_kind="simon-vector", rounds=rounds,
                                  hidden_width=32, label_mode="null", train_rows=8,
                                  dev_rows=4, test_rows=4, batch_size=4, epochs=1)
        assert config.rounds == rounds


def test_factory_modes_rejected() -> None:
    for features, mode in (("support", "direct"), ("full", "public-residual")):
        with pytest.raises(ValueError):
            build_predictor(model_kind="simon-vector", rounds=6,
                            feature_mode=features, prediction_mode=mode)


def test_cross_round_wiring_is_not_interchangeable(tmp_path: Path) -> None:
    with torch.random.fork_rng():
        torch.manual_seed(17)
        five, six = SimonVectorPredictor(rounds=5), SimonVectorPredictor(rounds=6)
    five_buffers = dict(five.named_buffers())
    six_buffers = dict(six.named_buffers())
    assert set(five_buffers) != set(six_buffers)
    assert len(five_buffers) == PLAN_FACTS[5]["groups"] + 3
    assert len(six_buffers) == PLAN_FACTS[6]["groups"] + 3
    # Canonical per-round wiring: the other round's operands/leaves are rejected.
    with pytest.raises(ValueError, match="noncanonical"):
        six.indices = five_buffers["indices"]
        six.validate_wiring()
    with pytest.raises(ValueError, match="noncanonical"):
        five.operands_0 = torch.tensor(simon_vector_plan(6).groups[0].operands)
        five.validate_wiring()
    # Parameters and state names are not interchangeable across rounds.
    with pytest.raises(RuntimeError):
        six.load_state_dict(five.state_dict())
    with pytest.raises(RuntimeError):
        five.load_state_dict(six.state_dict())
    config = ExperimentConfig(name="vector-null-r5", dataset="simon",
                              model_kind="simon-vector", rounds=5, hidden_width=32,
                              label_mode="null", train_rows=8, dev_rows=4, test_rows=4,
                              batch_size=4, epochs=1)
    checkpoint = tmp_path / "five.pt"
    save_checkpoint(checkpoint, five, config, 1, 2, 0, .5)
    payload = torch.load(checkpoint, weights_only=True)
    forged = dict(payload["config"])
    forged["rounds"] = 6
    import hashlib
    payload["config"] = forged
    payload["config_sha256"] = hashlib.sha256(
        json.dumps(forged, sort_keys=True, allow_nan=False).encode()).hexdigest()
    swapped = tmp_path / "swapped.pt"
    torch.save(payload, swapped)
    with pytest.raises(ValueError, match="state names mismatch"):
        load_checkpoint(swapped, file_sha256(swapped), device=torch.device("cpu"))


@pytest.mark.parametrize("rounds", [5, 6])
def test_cli_null_roundtrip_and_all_wiring_buffers_strict(
    rounds: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_experiments import training
    from cipher_experiments.data import prediction

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no cipher labels in engineering")

    monkeypatch.setattr(prediction, "_encrypt", forbidden)
    config = ExperimentConfig(name=f"vector-null-r{rounds}", dataset="simon",
                              model_kind="simon-vector",
                              rounds=rounds, hidden_width=32, label_mode="null",
                              train_rows=8, dev_rows=4, test_rows=4, batch_size=4,
                              epochs=1)
    path = tmp_path / "config.json"
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
    assert saved == config and model.rounds == rounds
    assert sum(p.numel() for p in model.parameters()) == PLAN_FACTS[rounds]["parameters"]
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


# Pinned pre-five-round baseline: the six-round vector sources of
# ae2c7ac160bda758d6ee716816da303ebfaa9870 are identical to 509b34f for every
# file below. Pinning a commit (instead of HEAD) keeps the comparison
# meaningful after this change itself is committed; a mutable HEAD would turn
# the probe into a self-comparison. No static cross-environment golden digests
# are used: torch RNG/forward bytes may differ between torch builds, so old and
# new sides are always probed twice within the same environment.
SIX_ROUND_BASELINE_COMMIT = "ae2c7ac160bda758d6ee716816da303ebfaa9870"

BASELINE_PACKAGE_FILES = (
    "__init__.py",
    "representations/__init__.py",
    "representations/simon_cone.py",
    "representations/simon_vector.py",
    "models/__init__.py",
    "models/mlp.py",
    "models/prediction.py",
    "models/simon_vector.py",
)

BASELINE_PROBE = """
import hashlib
import json
import sys

import torch

torch.set_num_threads(1)
from cipher_experiments.models.simon_vector import SimonVectorPredictor  # noqa: E402

with torch.random.fork_rng():
    torch.manual_seed(17)
    model = SimonVectorPredictor()
generator = torch.Generator().manual_seed(17)
x = torch.randint(0, 2, (5, 64), generator=generator).float()
with torch.no_grad():
    y = model(x)
state = {name: value.detach().contiguous() for name, value in model.state_dict().items()}
print(json.dumps({
    "state_sha256": hashlib.sha256(b"".join(
        name.encode() + state[name].numpy().tobytes() for name in sorted(state))).hexdigest(),
    "forward_sha256": hashlib.sha256(y.numpy().tobytes()).hexdigest(),
    "forward_values": y.flatten().tolist(),
    "parameters": sum(p.numel() for p in model.parameters()),
    "state_shapes": {name: list(state[name].shape) for name in sorted(state)},
}))
"""


def _baseline_probe(root: Path, tmp_path: Path) -> tuple[dict[str, Any] | None, str]:
    """Probe a temporary package copied from the pinned baseline commit.

    Returns the probe result (or None) plus a human-readable skip reason: no
    usable git executable, a repository without the pinned commit, or a commit
    whose extracted compiler is not the pre-five-round source.
    """
    package = tmp_path / "baseline"
    unavailable = (f"pinned baseline {SIX_ROUND_BASELINE_COMMIT[:9]} unavailable "
                   "(no git executable or history); six-round identity not re-verified")
    for relative in BASELINE_PACKAGE_FILES:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "show",
                 f"{SIX_ROUND_BASELINE_COMMIT}:src/cipher_experiments/{relative}"],
                capture_output=True, check=False)
        except (FileNotFoundError, PermissionError):
            # No usable git executable, for example a bare source archive.
            return None, unavailable
        if result.returncode != 0:
            return None, unavailable
        destination = package / "cipher_experiments" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(result.stdout)
    # Provenance guard: an archive nested inside an unrelated parent git
    # repository can still resolve the commit; require the extracted compiler
    # to carry the pre-five-round no-argument signature so the baseline cannot
    # silently mix sources from a different tree.
    compiler = package / "cipher_experiments/representations/simon_vector.py"
    if b"def simon_vector_plan() -> SimonVectorPlan:" not in compiler.read_bytes():
        return None, (f"pinned commit {SIX_ROUND_BASELINE_COMMIT[:9]} resolved in an "
                      "unexpected repository (source archive inside a parent git "
                      "tree); baseline provenance unclear, six-round identity "
                      "not re-verified")
    return _run_probe(package, tmp_path), ""


def _run_probe(source_root: Path, tmp_path: Path) -> dict[str, Any]:
    script = tmp_path / "probe.py"
    script.write_text(BASELINE_PROBE, encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    completed = subprocess.run([sys.executable, str(script)], env=environment,
                               capture_output=True, text=True, check=True,
                               timeout=600)
    result: dict[str, Any] = json.loads(completed.stdout.strip().splitlines()[-1])
    return result


def test_six_round_shapes_initial_state_and_forward_match_pinned_baseline(
    tmp_path: Path,
) -> None:
    """The default six-round model must stay bit-identical to the pinned baseline.

    Sources of SIX_ROUND_BASELINE_COMMIT are materialized into a temporary
    package copy (never the backups tree) and probed in a subprocess, then
    compared with the same probe against the modified working tree. When git or
    the commit is unavailable (for example a source archive), the test skips
    transparently instead of silently comparing against itself.
    """
    root = Path(__file__).resolve().parents[1]
    baseline, skip_reason = _baseline_probe(root, tmp_path)
    if baseline is None:
        pytest.skip(skip_reason)
    current = _run_probe(root / "src", tmp_path)
    assert current == baseline
    assert current["parameters"] == PLAN_FACTS[6]["parameters"]
    assert len(current["state_shapes"]) == PLAN_FACTS[6]["groups"] + 16
    operands = [name for name in current["state_shapes"] if name.startswith("operands_")]
    assert len(operands) == PLAN_FACTS[6]["groups"]
    assert current["state_shapes"]["constants"] == [PLAN_FACTS[6]["symbolic"], 32]
    assert current["state_shapes"]["indices"] == [PLAN_FACTS[6]["plaintext"]]

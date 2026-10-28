"""Six-round engineering probes, not cipher fitting or a production diagnostic API."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from cipher_experiments.models.unrolled_simon import UnrolledSimonPredictor


def _active_conditions(rounds: int, target_bit: int) -> list[list[int]]:
    """Backward public graph sets, independent of model autograd and parameters."""
    high, low = ({target_bit}, set()) if target_bit < 32 else (set(), {target_bit - 32})
    active: list[list[int]] = [[] for _ in range(rounds)]
    for index in range(rounds - 1, -1, -1):
        active[index] = sorted(high)
        high, low = low | {(i + s) % 32 for i in high for s in (1, 8, 2)}, high
    return active


def _fixture() -> tuple[UnrolledSimonPredictor, Tensor, Tensor]:
    # Preserve global initialization RNG in callers/tests.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = UnrolledSimonPredictor(32, rounds=6, target_bit=0)
    generator = torch.Generator().manual_seed(194101)
    x = torch.randint(0, 2, (64, 64), generator=generator).float()
    # Independent synthetic labels, no cipher or key involved.
    y = torch.randint(0, 2, (64, 1), generator=generator).float()
    return model, x, y


def _double_oracle(parameters: dict[str, Tensor], x: Tensor, rounds: int = 6) -> Tensor:
    """FP64 reference uses explicit indices and weights, never production forward/cell.

    An engineering arithmetic cross-check only: the same specified architecture
    is implemented twice, not an independent proof of cipher learnability.
    """
    high, low = x[:, :32].double(), x[:, 32:].double()
    positions = torch.arange(32)
    logits = torch.empty_like(high)
    for r in range(rounds):
        state = torch.stack([low, high[:, (positions + 1) % 32],
                             high[:, (positions + 8) % 32],
                             high[:, (positions + 2) % 32]], dim=2) * 2 - 1
        condition = parameters["conditions"][r].expand(len(x), -1, -1)
        values = torch.cat([state, condition], dim=2)
        hidden = torch.tanh(values @ parameters["cell.0.weight"].T
                            + parameters["cell.0.bias"])
        logits = (hidden @ parameters["cell.2.weight"].T
                  + parameters["cell.2.bias"]).squeeze(2)
        low, high = high, 1 / (1 + torch.exp(-logits))
    return logits[:, :1]


def _gradient_probe() -> dict[str, Any]:
    """Run one fixed synthetic batch, one Adam step, and finite differences."""
    model, x, y = _fixture()
    rounds = model.rounds
    active = _active_conditions(rounds, model.target_bit)
    outputs: list[Tensor] = []
    def hook(_module: torch.nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        output.retain_grad()
        outputs.append(output)
    handle = model.cell.register_forward_hook(hook)
    try:
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()  # type: ignore[no-untyped-call]
    finally:
        handle.remove()
    assert len(outputs) == rounds
    conditions_grad = model.conditions.grad
    assert conditions_grad is not None
    layers = []
    for r in range(rounds):
        g = conditions_grad[r]
        mask = torch.zeros(32, dtype=torch.bool)
        mask[active[r]] = True
        assert torch.count_nonzero(g[~mask]) == 0
        assert bool(torch.isfinite(g).all()) and float(g[mask].abs().sum()) > 0
        assert outputs[r].grad is not None
        logit_g = outputs[r].grad
        assert logit_g is not None
        values = outputs[r].detach()
        probabilities = torch.sigmoid(values)
        layers.append({"round": r + 1, "active_condition_positions": len(active[r]),
                       "condition_grad_l2": float(g.norm()),
                       "condition_grad_max_abs": float(g.abs().max()),
                       "condition_grad_nonzero_elements": int(torch.count_nonzero(g)),
                       "logit_grad_l2": float(logit_g.norm()),
                       "hidden_min": float(probabilities.min()),
                       "hidden_max": float(probabilities.max())})
    parameters = {n: p.detach().double().clone() for n, p in model.named_parameters()}
    reference = _double_oracle(parameters, x)
    torch.testing.assert_close(logits.detach().double(), reference, rtol=2e-5, atol=2e-7)
    coordinates: list[tuple[str, tuple[int, ...]]] = [
        ("conditions", (r, active[r][0], 0)) for r in range(rounds)]
    coordinates += [("cell.0.weight", (0, 0)), ("cell.2.weight", (0, 0))]
    named = dict(model.named_parameters())
    differences = []
    for name, coordinate in coordinates:
        tensor = parameters[name]
        center = float(tensor[coordinate])
        estimates = []
        for eps in (1e-3, 5e-4):
            try:
                tensor[coordinate] = center + eps
                plus = float(F.binary_cross_entropy_with_logits(_double_oracle(parameters, x),
                                                                y.double()))
                tensor[coordinate] = center - eps
                minus = float(F.binary_cross_entropy_with_logits(_double_oracle(parameters, x),
                                                                 y.double()))
            finally:
                tensor[coordinate] = center
            estimates.append((plus - minus) / (2 * eps))
        gradient = named[name].grad
        assert gradient is not None
        automatic = float(gradient[coordinate])
        assert all(math.isclose(automatic, n, rel_tol=.02, abs_tol=2e-10) for n in estimates)
        differences.append({"parameter": name, "coordinate": list(coordinate),
                            "autograd_fp32": automatic, "central_difference_fp64": estimates})
    gradients_before = {n: p.grad.clone() for n, p in model.named_parameters()
                        if p.grad is not None}
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                                                     error_if_nonfinite=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    optimizer.step()
    updates = []
    for r in range(rounds):
        delta = (model.conditions[r].detach() - initial["conditions"][r]).abs()
        assert float(delta.sum()) > 0
        updates.append({"round": r + 1, "condition_update_l2": float(delta.norm()),
                        "condition_update_max_abs": float(delta.max()),
                        "changed_elements": int(torch.count_nonzero(delta))})
    assert all(not torch.equal(p, initial[n]) for n, p in model.named_parameters())
    return {"kind": "one synthetic CPU batch at fixed initialization, not cipher training",
            "rows": len(x), "updates": 1, "real_cipher_labels": 0,
            "loss_before": float(loss.detach()), "gradient_norm_before_clip": total_norm,
            "gradient_norm_by_parameter": {n: float(g.norm()) for n, g in gradients_before.items()},
            "layers": layers, "finite_difference_samples": differences,
            "updates_by_round": updates,
            "loss_after": float(F.binary_cross_entropy_with_logits(model(x), y).detach()),
            "limits": "single init/synthetic batch/one step; not convergence or cipher-state proof"}


def test_six_round_final_loss_has_gradients_updates_and_numeric_agreement() -> None:
    report = _gradient_probe()
    assert report["rows"] == 64 and report["updates"] == 1
    assert len(report["finite_difference_samples"]) == 8
    assert report["real_cipher_labels"] == 0


def test_low_half_excludes_last_round_condition_by_copy_path() -> None:
    model, x, y = _fixture()
    model.target_bit = 32
    F.binary_cross_entropy_with_logits(model(x), y).backward()  # type: ignore[no-untyped-call]
    gradient = model.conditions.grad
    assert gradient is not None
    active = _active_conditions(6, 32)
    assert active[-1] == [] and torch.count_nonzero(gradient[-1]) == 0
    assert float(gradient[:-1].abs().sum()) > 0

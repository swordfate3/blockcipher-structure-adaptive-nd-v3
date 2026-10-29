"""No scientific labels: algebra, artificial graph, gradients and isolation only."""
from __future__ import annotations

import math

import pytest
import torch
from structure_vector_fixture import (
    ScalarGraph,
    SymmetricVectorGate,
    VectorGraph,
    all_inputs,
    artificial_graph,
    artificial_labels,
    depths,
    exact_witness,
)
from torch.nn import functional as F


def test_symbol_gate_identities() -> None:
    for a in (0, 1):
        for b in (0, 1):
            x, y = 1 - 2 * a, 1 - 2 * b
            assert x * y == 1 - 2 * (a ^ b)
            assert (1 + x + y - x * y) / 2 == 1 - 2 * (a & b)


def test_public_artificial_graph_topology() -> None:
    nodes = artificial_graph()
    assert len(nodes) == 40 and max(depths(nodes)) == 26
    assert sum(n.kind == "and" for n in nodes) == 8
    assert sum(n.kind == "xor" for n in nodes) == 18
    for i, node in enumerate(nodes):
        if node.kind in ("and", "xor"):
            assert 0 <= node.left < i and 0 <= node.right < i
    reached = set()
    pending = [len(nodes) - 1]
    while pending:
        i = pending.pop()
        if i in reached:
            continue
        reached.add(i)
        n = nodes[i]
        if n.kind in ("and", "xor"):
            pending.extend((n.left, n.right))
    assert reached == set(range(len(nodes)))


def test_truth_table_balanced_all_inputs_effective() -> None:
    x = all_inputs()
    y = artificial_labels(x)
    assert len({tuple(row) for row in x.tolist()}) == 4096 and int(y.sum()) == 2048
    for bit in range(12):
        changed = x.clone()
        changed[:, bit] = 1 - changed[:, bit]
        assert bool((artificial_labels(changed) != y).any())
    order = torch.randperm(4096, generator=torch.Generator().manual_seed(194301))
    train = order[:2048]
    functions = [artificial_labels(x, (a, b))[train] for a in (0, 1) for b in (0, 1)]
    assert all(not torch.equal(a, b) for i, a in enumerate(functions)
               for b in functions[i + 1:])


@pytest.mark.parametrize("width", [1, 32])
def test_exact_artificial_witness(width: int) -> None:
    model = exact_witness(artificial_graph(), width)
    x = all_inputs().double()
    with torch.no_grad():
        assert torch.equal((model(x) > 0).float(), artificial_labels(x))
        assert torch.equal(model(x).abs(), torch.full((4096, 1), 4., dtype=torch.float64))
    # This hand-set instance is destroyed, never passed to the learning study.


def test_gate_exchange_symmetry_and_reconvergence_identity() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(17)
        gate = SymmetricVectorGate(32).double()
        a, b = torch.randn(7, 32).double(), torch.randn(7, 32).double()
    assert torch.equal(gate(a, b), gate(b, a))
    model = exact_witness(artificial_graph())
    sign = torch.zeros(2, 32, dtype=torch.float64)
    sign[:, 0] = torch.tensor([-1., 1.])
    result = model.xor_gate(sign, sign)
    assert torch.equal(result[:, 0], torch.ones(2, dtype=torch.float64))
    # Only for Boolean-encoded inputs: arbitrary latent vectors need not cancel.
    assert .5 + .5 - 2 * .5 * .5 == .5  # old soft relaxation, not exact joint probability


def test_shared_parameters_no_static_or_plaintext_bypass() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = VectorGraph(artificial_graph())
    assert sum(p.numel() for p in model.parameters()) == 8353
    assert set(dict(model.named_children())) == {"and_gate", "xor_gate", "readout"}
    assert model.constants.shape == (2, 32)
    x = all_inputs()[:8]
    assert torch.allclose(model(x), torch.cat([model(x[:3]), model(x[3:])]), atol=1e-7)
    model.capture = True
    logits = model(x)
    assert torch.equal(logits, F.linear(model.trace[-1], model.readout.weight,
                                      model.readout.bias))
    scalar = ScalarGraph(artificial_graph())
    assert sum(p.numel() for p in scalar.parameters()) == 2


def test_vector_gradients_and_directional_derivative() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = VectorGraph(artificial_graph()).double()
    x = all_inputs()[::127].double()
    y = artificial_labels(x).double()
    loss = F.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()  # type: ignore[no-untyped-call]
    parameters = dict(model.named_parameters())
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert float(parameter.grad.norm()) > 0., name
    for name in ("constants", "and_gate.linear.weight", "xor_gate.product.weight",
                 "readout.weight"):
        parameter = parameters[name]
        gradient = parameter.grad
        assert gradient is not None
        direction = gradient / gradient.norm()
        predicted = float((gradient * direction).sum())
        original = parameter.detach().clone()
        values = []
        for sign in (1, -1):
            with torch.no_grad():
                parameter.copy_(original + sign * 1e-5 * direction)
                values.append(float(F.binary_cross_entropy_with_logits(model(x), y)))
        with torch.no_grad():
            parameter.copy_(original)
        assert math.isclose((values[0] - values[1]) / 2e-5, predicted,
                            rel_tol=.001, abs_tol=2e-9), name

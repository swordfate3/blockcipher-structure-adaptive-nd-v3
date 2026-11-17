"""Artificial pairs and injected backend only: never scientific forward/Adam."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from cipher_lab_plugins.differential_relations import training


class Ledger:
    def __init__(self) -> None:
        self.attempted: dict[str, int] = {}
        self.completed: dict[str, int] = {}

    def check(self) -> Any:
        pass

    def reserve(self, kind: Any, n: Any) -> Any:
        self.attempted[kind] = self.attempted.get(kind, 0) + n

    def complete(self, kind: Any, n: Any) -> Any:
        self.completed[kind] = self.completed.get(kind, 0) + n


class Backend:
    def __init__(self) -> None:
        self.batches: list[tuple[int, ...]] = []
        self.seeds: list[tuple[int, float]] = []
        self.scores = iter([1.0, 1.0, 2.0])

    def initialize(self, seed: Any, lr: Any) -> Any:
        self.seeds.append((seed, lr))
        return {"weight": 0}

    def update(self, model: Any, orbits: Any, labels: Any) -> Any:
        self.batches.append(tuple(map(int, orbits)))
        model["weight"] += 1

    def predict(self, model: Any, orbits: Any) -> Any:
        return np.zeros((len(orbits), 4))


def test_first_minimum_is_saved_and_read_from_disk() -> None:
    ledger = Ledger()
    backend = Backend()
    writes = []
    reads = []
    x = np.arange(8, dtype=np.uint64)
    y = np.tile([0, 1, 0, 1], (8, 1))
    options = SimpleNamespace(updates=3, dev_every=1, batch_groups=1, shuffle_seed=7)

    def save(model: Any, step: Any) -> Any:
        writes.append((dict(model), step))
        return ("model/F1-best.pt", "a" * 64)

    def restore(model: Any, ref: Any, digest: Any) -> Any:
        reads.append((ref, digest))
        model["weight"] = 99

    result = training.fit(
        backend=backend,
        options=options,
        seed=31,
        lr=0.001,
        train=x,
        train_labels=y,
        dev=x[:4],
        dev_labels=y[:4],
        meter=ledger,
        budget=nullcontext(),
        save=save,
        restore=restore,
    )
    assert result.selected_step == 1 and len(writes) == len(reads) == 1
    assert result.model["weight"] == 99
    assert ledger.attempted == ledger.completed == {"optimizer_updates": 3, "forward_rows": 24}


def test_same_shuffle_independent_of_init_and_lr() -> None:
    def batches(seed: Any, lr: Any) -> Any:
        b = Backend()
        x = np.arange(8, dtype=np.uint64)
        y = np.zeros((8, 4))
        training.fit(
            backend=b,
            options=SimpleNamespace(updates=4, dev_every=2, batch_groups=1, shuffle_seed=11),
            seed=seed,
            lr=lr,
            train=x,
            train_labels=y,
            dev=x[:4],
            dev_labels=y[:4],
            meter=Ledger(),
            budget=nullcontext(),
            save=lambda m, s: ("r", "a" * 64),
            restore=lambda *a: None,
        )
        return b.batches

    assert batches(31, 0.001) == batches(37, 0.0003)


def test_failed_update_keeps_reservation_without_completion() -> None:
    class Broken(Backend):
        def update(self, *a: Any) -> Any:
            raise RuntimeError("private detail")

    ledger = Ledger()
    x = np.arange(4, dtype=np.uint64)
    with pytest.raises(RuntimeError):
        training.fit(
            backend=Broken(),
            options=SimpleNamespace(updates=1, dev_every=1, batch_groups=1, shuffle_seed=11),
            seed=31,
            lr=0.001,
            train=x,
            train_labels=np.zeros((4, 4)),
            dev=x,
            dev_labels=np.zeros((4, 4)),
            meter=ledger,
            budget=nullcontext(),
            save=lambda *a: ("r", "a" * 64),
            restore=lambda *a: None,
        )
    assert ledger.attempted == {"optimizer_updates": 1, "forward_rows": 4}
    assert not ledger.completed


def test_fixed_32_dev_opportunities_first_minimum() -> None:
    b = Backend()
    x = np.arange(4, dtype=np.uint64)
    result = training.fit(
        backend=b,
        options=SimpleNamespace(updates=32, dev_every=1, batch_groups=1, shuffle_seed=7),
        seed=31,
        lr=0.001,
        train=x,
        train_labels=np.zeros((4, 4)),
        dev=x,
        dev_labels=np.zeros((4, 4)),
        meter=Ledger(),
        budget=nullcontext(),
        save=lambda *a: ("r", "a" * 64),
        restore=lambda *a: None,
    )
    assert len(result.curve) == 32 and result.selected_step == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_dev_stops_without_selection(bad: float) -> None:
    class Nonfinite(Backend):
        def predict(self, model: Any, orbits: Any) -> Any:
            return np.full((len(orbits), 4), bad)

    x = np.arange(4, dtype=np.uint64)
    with pytest.raises(ValueError, match="NONFINITE_FORWARD"):
        training.fit(
            backend=Nonfinite(),
            options=SimpleNamespace(updates=2, dev_every=1, batch_groups=1, shuffle_seed=7),
            seed=31,
            lr=0.001,
            train=x,
            train_labels=np.zeros((4, 4)),
            dev=x,
            dev_labels=np.zeros((4, 4)),
            meter=Ledger(),
            budget=nullcontext(),
            save=lambda *a: pytest.fail("nonfinite selection"),
            restore=lambda *a: pytest.fail("nonfinite reload"),
        )


def test_torch_update_checks_parameters_after_fake_step() -> None:
    from types import SimpleNamespace

    calls = []

    class Value:
        def __init__(self, finite: bool = True) -> None:
            self.finite = finite

        def to(self, *a: Any, **k: Any) -> Any:
            return self

        def reshape(self, *a: Any) -> Any:
            return self

        def all(self) -> bool:
            return self.finite

        def backward(self) -> None:
            calls.append("backward")

    parameter = Value()

    class Model:
        def train(self) -> None:
            pass

        def parameters(self) -> Any:
            return iter([Value(), parameter])

        def __call__(self, *a: Any) -> Any:
            return Value()

    def step() -> None:
        calls.append("fake-step")
        parameter.finite = False

    backend: Any = object.__new__(training.TorchBackend)
    backend.device = "fake"
    backend.encoder = lambda x: x
    backend.public = Value()
    backend.optimizer = SimpleNamespace(zero_grad=lambda **k: None, step=step)
    backend.torch = SimpleNamespace(
        from_numpy=lambda x: Value(),
        tensor=lambda *a, **k: Value(),
        float32="fake",
        isfinite=lambda x: x,
        nn=SimpleNamespace(
            functional=SimpleNamespace(binary_cross_entropy_with_logits=lambda *a: Value()),
            utils=SimpleNamespace(clip_grad_norm_=lambda *a, **k: Value()),
        ),
    )
    with pytest.raises(ValueError, match="NONFINITE_PARAMETER"):
        backend.update(Model(), None, None)
    assert calls == ["backward", "fake-step"]

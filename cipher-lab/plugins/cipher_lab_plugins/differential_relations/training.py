"""Fixed FP32 fitting and streaming forwards; no labels, storage or reports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FitResult:
    model: Any
    updates: int
    selected_step: int
    best_bce: float
    curve: list[dict[str, object]]
    ref: str
    sha256: str


def forward(backend: Any, model: Any, orbits: Any, *, meter: Any, batch_pairs: int = 1024) -> Any:
    """Each pair, not each four-pair group, consumes one forward unit."""
    outputs = []
    for start in range(0, len(orbits), batch_pairs):
        batch = orbits[start : start + batch_pairs]
        meter.reserve("forward_rows", len(batch))
        logits = np.asarray(backend.predict(model, batch), dtype=np.float64)
        if logits.shape != (len(batch), 4) or not np.isfinite(logits).all():
            raise ValueError("NONFINITE_FORWARD")
        meter.complete("forward_rows", len(batch))
        outputs.append(logits)
    meter.check()
    return np.concatenate(outputs)


def fit(
    *,
    backend: Any,
    options: Any,
    seed: int,
    lr: float,
    train: Any,
    train_labels: Any,
    dev: Any,
    dev_labels: Any,
    meter: Any,
    budget: Any,
    save: Callable[[Any, int], tuple[str, str]],
    restore: Callable[[Any, str, str], None],
    progress: Callable[[dict[str, object]], None] | None = None,
    on_step: Callable[[int], None] | None = None,
) -> FitResult:
    """Strict improvement only; mandatory durable reload before returning."""
    from .metrics import logloss

    batch_pairs = options.batch_groups * 4
    if len(train) % batch_pairs or len(dev) % 4:
        raise ValueError("BATCH_ALIGNMENT")
    # Separate NumPy RNG: model initialization/lr/null targets cannot alter order.
    rng = np.random.default_rng(options.shuffle_seed)
    groups = len(train) // 4
    order = np.empty(0, dtype=np.int64)
    position = 0
    best = float("inf")
    selected = 0
    curve: list[dict[str, object]] = []
    ref = digest = ""
    with budget:
        meter.check()
        model = backend.initialize(seed, lr)
        for step in range(1, options.updates + 1):
            if position >= len(order):
                order = rng.permutation(groups)
                position = 0
            indices = (
                order[position : position + options.batch_groups, None] * 4 + np.arange(4)
            ).reshape(-1)
            position += options.batch_groups
            meter.reserve("optimizer_updates", 1)
            meter.reserve("forward_rows", len(indices))
            backend.update(model, train[indices], train_labels[indices])
            meter.complete("forward_rows", len(indices))
            meter.complete("optimizer_updates", 1)
            if on_step is not None:
                on_step(step)
            meter.check()
            if step % options.dev_every == 0:
                logits = forward(backend, model, dev, meter=meter, batch_pairs=batch_pairs)
                score = float(logloss(logits, dev_labels).mean(dtype=np.float64))
                if not np.isfinite(score):
                    raise ValueError("NONFINITE_DEV")
                curve.append({"step": step, "bce": score})
                if score < best:
                    ref, digest = save(model, step)
                    best, selected = score, step
                if progress is not None:
                    progress(
                        {
                            "updates": step,
                            "selected_step": selected,
                            "dev_curve": curve,
                            "best_bce": best,
                            "checkpoint_ref": ref,
                            "checkpoint_sha256": digest,
                        }
                    )
                meter.check()
        if not selected:
            raise ValueError("NO_SELECTED_CHECKPOINT")
        restore(model, ref, digest)
        meter.check()
    return FitResult(model, options.updates, selected, best, curve, ref, digest)


class TorchBackend:
    """Production implementation; import/construct only after owner claim."""

    def __init__(self, device: str, encoder: Callable[[Any], Any]) -> None:
        import torch

        from cipher_lab.representations.tasks import public_features

        self.torch = torch
        self.device = device
        self.encoder = encoder
        self.public = torch.tensor(public_features("SIMON"), dtype=torch.float32, device=device)
        self.optimizer: Any = None

    def initialize(self, seed: int, lr: float) -> Any:
        from cipher_lab.models.shared_m_relations import SharedMRelations

        torch = self.torch
        # All parameters initialize on CPU; fork restores global CPU RNG exactly.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = SharedMRelations().to(dtype=torch.float32)
        model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        return model

    def update(self, model: Any, orbits: Any, labels: Any) -> None:
        torch = self.torch
        model.train()
        self.optimizer.zero_grad(set_to_none=True)
        pairs = torch.from_numpy(self.encoder(orbits)).to(self.device)
        targets = torch.tensor(labels, dtype=torch.float32, device=self.device)
        logits = model(pairs, self.public).reshape(-1, 4)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        if not bool(torch.isfinite(loss)):
            raise ValueError("NONFINITE_LOSS")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if not bool(torch.isfinite(norm)):
            raise ValueError("NONFINITE_GRADIENT")
        self.optimizer.step()
        for parameter in model.parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise ValueError("NONFINITE_PARAMETER")

    def predict(self, model: Any, orbits: Any) -> Any:
        torch = self.torch
        model.eval()
        with torch.no_grad():
            pairs = torch.from_numpy(self.encoder(orbits)).to(self.device)
            return model(pairs, self.public).reshape(-1, 4).detach().cpu().numpy()

    def release(self, model: Any) -> None:
        # Application retains at most one model in GPU memory at a time.
        self.optimizer = None
        model.cpu()

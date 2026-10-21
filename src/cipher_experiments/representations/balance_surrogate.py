"""Learned ranking of integral-balance questions, always verified by the engine.

The balance verdict itself is cheap to compute exactly, so a model is only useful
as a *fast scorer* inside a search: rank thousands of candidate cubes, then let
:func:`~cipher_experiments.representations.integral.cube_balance` decide the few
that matter. Nothing here replaces the exact engine or claims a proof.

Labels come from the engine, so ``1`` means "the engine proved balance" and ``0``
means "not proved" - which is not the same as "unbalanced".
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .integral import cube_balance
from .simon_cone import simon_cone


@dataclass(frozen=True)
class Sample:
    """One (rounds, cube, target) question with its engine verdict."""

    rounds: int
    cube: tuple[int, ...]
    target: int
    balanced: bool


def cube_key(sample: Sample) -> tuple[int, tuple[int, ...], int]:
    """Identity of a cube question, used for split-by-cube evaluation."""
    return (sample.rounds, sample.cube, sample.target)


def make_features(sample: Sample) -> list[float]:
    """Features: normalised rounds, cube indicator, target one-hot, cube size."""
    if not 1 <= sample.rounds <= 16:
        raise ValueError("rounds must be in 1..16")
    if not 0 <= sample.target < 64:
        raise ValueError("target must be in 0..63")
    if any(not 0 <= bit < 64 for bit in sample.cube):
        raise ValueError("cube bits must be in 0..63")
    features = [sample.rounds / 16.0, len(sample.cube) / 32.0]
    features.extend(1.0 if bit in sample.cube else 0.0 for bit in range(64))
    features.extend(1.0 if bit == sample.target else 0.0 for bit in range(64))
    return features


def sample_questions(rng: random.Random, *, count: int, rounds_range: tuple[int, int],
                     size_range: tuple[int, int], bits: int = 32) -> list[Sample]:
    """Draw random cube questions without touching the cipher."""
    rounds_low, rounds_high = rounds_range
    size_low, size_high = size_range
    out: list[Sample] = []
    for _ in range(count):
        rounds = rng.randint(rounds_low, rounds_high)
        size = rng.randint(size_low, size_high)
        cube = tuple(sorted(rng.sample(range(bits), size)))
        target = rng.randrange(64)
        out.append(Sample(rounds, cube, target, False))
    return out


def label_with_engine(samples: list[Sample]) -> list[Sample]:
    """Attach the exact engine verdict to each question."""
    labelled: list[Sample] = []
    for sample in samples:
        verdict = cube_balance(simon_cone(sample.rounds, sample.target), set(sample.cube))
        labelled.append(Sample(sample.rounds, sample.cube, sample.target, verdict.balanced))
    return labelled


def to_tensors(samples: list[Sample]) -> tuple[Tensor, Tensor]:
    """Feature matrix and label vector for a batch of labelled samples."""
    features = torch.tensor([make_features(sample) for sample in samples], dtype=torch.float32)
    labels = torch.tensor([[1.0 if sample.balanced else 0.0] for sample in samples],
                          dtype=torch.float32)
    return features, labels


class BalanceScorer(nn.Module):
    """Small MLP that scores balance questions; never used as a proof."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        if hidden < 8:
            raise ValueError("hidden must be at least 8")
        self.layers = nn.Sequential(
            nn.Linear(130, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        logits: Tensor = self.layers(features)
        if logits.dtype != torch.float32 or logits.shape != (features.shape[0], 1):
            raise RuntimeError("scorer produced invalid logits")
        return logits


def fit_scorer(samples: list[Sample], *, seed: int = 17, epochs: int = 30,
               batch_size: int = 64, learning_rate: float = 1e-3,
               hidden: int = 128) -> tuple[BalanceScorer, list[float]]:
    """Train the scorer on engine-labelled questions; returns model and losses."""
    if not samples:
        raise ValueError("training samples are required")
    torch.manual_seed(seed)
    features, labels = to_tensors(samples)
    model = BalanceScorer(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(epochs):
        permutation = torch.randperm(features.shape[0])
        total = 0.0
        for start in range(0, features.shape[0], batch_size):
            index = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(features[index]), labels[index])
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total += float(loss.detach()) * len(index)
        losses.append(total / features.shape[0])
    return model, losses


@torch.no_grad()
def score(model: BalanceScorer, samples: list[Sample]) -> Tensor:
    """Positive-class probability for each question."""
    model.eval()
    features, _ = to_tensors(samples)
    return torch.sigmoid(model(features)).flatten()

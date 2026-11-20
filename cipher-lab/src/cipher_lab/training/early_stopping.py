"""Track joint validation-loss stagnation independently of model selection."""
from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

Termination = Literal['EARLY_STOPPING', 'EPOCH_CEILING']


def _nonnegative(value: object) -> float:
    if type(value) is int or type(value) is float:
        try:
            if math.isfinite(value) and value >= 0:
                return float(value)
        except OverflowError:
            pass
    raise ValueError('expected a finite nonnegative loss or delta')


@dataclass(frozen=True)
class Settings:
    warmup_epochs: int
    patience: int
    min_delta: float

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 1 for n in (self.warmup_epochs, self.patience)):
            raise ValueError('expected positive integer warmup and patience')
        _nonnegative(self.min_delta)


class EarlyStopping[Version: Hashable]:
    """Consume consecutive 1-based epochs with all version losses, once per epoch."""

    def __init__(self, versions: Iterable[Version], settings: Settings, *,
                 epoch_ceiling: int) -> None:
        keys = tuple(versions)
        self._wait = dict.fromkeys(keys, 0)
        if not keys or len(keys) != len(self._wait):
            raise ValueError('expected unique versions')
        if type(epoch_ceiling) is not int or epoch_ceiling < settings.warmup_epochs:
            raise ValueError('epoch ceiling must include warmup')
        self._settings = settings
        self._ceiling = epoch_ceiling
        self._epoch = 0
        self._anchors: dict[Version, float] = {}
        self._termination: Termination | None = None

    def observe(self, epoch: int, losses: Mapping[Version, float]) -> Termination | None:
        if self._termination is not None or type(epoch) is not int or epoch != self._epoch + 1:
            raise ValueError('expected the next epoch of an active tracker')
        if losses.keys() != self._wait.keys():
            raise ValueError('expected a loss for every version and no extras')
        # Validate the full observation before changing any state.
        scores = {key: _nonnegative(losses[key]) for key in self._wait}
        settings = self._settings
        for key, loss in scores.items():
            anchor = self._anchors.get(key, math.inf)
            if epoch <= settings.warmup_epochs:
                self._anchors[key] = min(anchor, loss)
            else:
                # The contract is the float target (anchor - min_delta): a loss exactly
                # on it is enough improvement, one ULP above it is not. Comparing
                # `anchor - loss >= min_delta` instead lets the rounding of that
                # subtraction reject an exact-target loss.
                target = anchor - settings.min_delta
                if loss < anchor and loss <= target:
                    self._anchors[key] = loss
                    self._wait[key] = 0
                else:
                    self._wait[key] += 1
        self._epoch = epoch
        if epoch == self._ceiling:
            self._termination = 'EPOCH_CEILING'
        elif all(wait >= settings.patience for wait in self._wait.values()):
            self._termination = 'EARLY_STOPPING'
        return self._termination

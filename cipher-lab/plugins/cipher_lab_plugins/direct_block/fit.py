"""Named epoch fit and durable reload; independent of any experiment's data/modes."""
from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Meter
from cipher_lab.training.early_stopping import EarlyStopping, Settings

Floats = NDArray[np.float32]
Labels = NDArray[np.uint8]
Indices = NDArray[np.intp]
Best = tuple[str, str]  # durable (ref, sha256)


class Engine(Protocol):
    def train_forward(self, inputs: Floats) -> Callable[[Labels], float | None]: ...
    def update(self) -> None: ...
    def predict(self, inputs: Floats) -> Floats: ...
    def dump(self) -> bytes: ...
    def restore(self, raw: bytes) -> None: ...


class BatchPanel(Protocol):
    def __len__(self) -> int: ...
    def batch(self, indices: Indices) -> tuple[Floats, Labels]: ...


@dataclass(frozen=True, slots=True)
class FitSettings:
    epochs: int
    batch_size: int
    checkpoint_bytes: int
    selected_columns: tuple[int, ...] | None = None
    dev_every_steps: int | None = None  # existing Trial only; otherwise once per epoch
    record_train_bce: bool = True       # existing adapters return None from backward
    early_stopping: Settings | None = None

    def __post_init__(self) -> None:
        values: tuple[int, ...] = (self.epochs, self.batch_size, self.checkpoint_bytes)
        if self.dev_every_steps is not None:
            values += (self.dev_every_steps,)
        if any(type(n) is not int or n <= 0 for n in values):
            raise ValueError('invalid fit settings')
        stop = self.early_stopping
        if stop is not None and (self.dev_every_steps is not None
                                 or self.epochs < stop.warmup_epochs):
            raise ValueError('early stopping requires complete epoch DEV and warmup')
        cols = self.selected_columns
        if cols is not None and (not cols or len(set(cols)) != len(cols)
                                 or any(type(i) is not int or i < 0 for i in cols)):
            raise ValueError('invalid DEV columns')


def assert_source(store: RunStore) -> None:
    if store.current_source_identity() != store.source_snapshot():
        raise ArtifactError('SOURCE_CHANGED')


def seeded_order(rows: int, seed: int) -> Callable[[int], Indices]:
    """Zero-based epochs, independent of either arm's execution/RNG consumption."""
    def order(epoch: int) -> Indices:
        return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(rows)
    return order


def batch_indices(rows: int, batch_size: int) -> Iterator[Indices]:
    for start in range(0, rows, batch_size):
        yield np.arange(start, min(start + batch_size, rows), dtype=np.intp)


def predict_batch(engine: Engine, inputs: Floats, labels: Labels, meter: Meter) -> Floats:
    meter.reserve('forward_rows', len(inputs))
    logits = engine.predict(inputs)
    if logits.dtype != np.float32 or logits.shape != labels.shape or not np.isfinite(logits).all():
        raise ValueError('invalid logits')
    meter.complete('forward_rows', len(inputs))
    return logits


def _bce_sum(logits: Floats, labels: Labels) -> float:
    signed = np.where(labels == 0, logits, -logits).astype(np.float64)
    return float(np.logaddexp(0., signed).sum())


def _dev(panel: BatchPanel, engine: Engine, settings: FitSettings,
         meter: Meter) -> tuple[float, float]:
    active = full = 0.
    active_bits = full_bits = 0
    for at in batch_indices(len(panel), settings.batch_size):
        x, y = panel.batch(at)
        z = predict_batch(engine, x, y, meter)
        full += _bce_sum(z, y)
        full_bits += y.size
        cols = settings.selected_columns
        if cols is not None:
            if y.ndim != 2 or max(cols) >= y.shape[1]:
                raise ValueError('invalid DEV columns')
            active += _bce_sum(z[:, cols], y[:, cols])
            active_bits += y[:, cols].size
    return (active / active_bits if active_bits else full / full_bits), full / full_bits


def _no_phase(value: str) -> None:
    pass


def fit(train: BatchPanel, dev: BatchPanel, engine: Engine, order: Callable[[int], Indices],
        slot: str, arm: dict[str, object], summary: dict[str, object], settings: FitSettings,
        meter: Meter, store: RunStore, *, phase: Callable[[str], None] = _no_phase) -> Best | None:
    """Compute full-label training updates and select the earliest minimum DEV BCE."""
    batch = settings.batch_size
    rows = len(train)
    if not rows or rows % batch or not len(dev):
        raise ValueError('invalid fit panel size')
    best: Best | None = None
    best_bce = math.inf
    curve: list[dict[str, object]] = []
    arm['dev'] = curve
    step = 0
    # The checkpoint slot is this fit's unique arm key, not a joint tracker.
    stopper = (EarlyStopping((slot,), settings.early_stopping, epoch_ceiling=settings.epochs)
               if settings.early_stopping is not None else None)
    for epoch in range(settings.epochs):
        at = order(epoch)
        if (at.shape != (rows,) or at.dtype != np.intp or at.min() != 0 or at.max() != rows - 1
                or not (np.bincount(at) == 1).all()):
            raise ValueError('invalid training permutation')
        train_total = 0.
        train_rows = 0
        for start in range(0, rows, batch):
            step += 1
            phase('TRAIN')
            x, y = train.batch(at[start:start + batch])
            meter.reserve('forward_rows', len(x))
            backward = engine.train_forward(x)
            meter.complete('forward_rows', len(x))
            meter.check()
            loss = backward(y)
            del backward
            if settings.record_train_bce:
                if loss is None or not math.isfinite(loss) or loss < 0:
                    raise ValueError('invalid training BCE')
                train_total += loss * len(x)
                train_rows += len(x)
            meter.reserve('optimizer_updates', 1)
            engine.update()
            meter.complete('optimizer_updates', 1)
            every = settings.dev_every_steps
            if (step % every != 0 if every is not None else start + batch != rows):
                continue
            phase('DEV')
            assert_source(store)
            bce, full = _dev(dev, engine, settings, meter)
            meter.check()
            point: dict[str, object] = {'step': step, 'bce': bce}
            if every is None:
                point.update(epoch=epoch + 1, full_bce=full)
                if settings.selected_columns is not None:
                    point['active_bce'] = bce
            if settings.record_train_bce:
                point['train_bce'] = train_total / train_rows
            curve.append(point)
            improved = bce < best_bce
            if improved:
                phase('BEST_SAVE')
                raw = engine.dump()
                meter.check()
                if type(raw) is not bytes or not 0 < len(raw) <= settings.checkpoint_bytes:
                    raise ValueError('invalid checkpoint size')
                digest = sha256(raw).hexdigest()
                ref = store.write_model(slot, raw, replace=best is not None)
                del raw
                best, best_bce = (ref, digest), bce
                arm['selected'] = {**point, 'ref': ref, 'sha256': digest}
            termination = None
            if stopper is not None:
                termination = stopper.observe(epoch + 1, {slot: bce})
                arm['epochs_completed'] = epoch + 1
                if termination is not None:
                    arm['stop_reason'] = termination
            counts = meter.snapshot()
            store.write_snapshot({'counts': counts, 'summary': summary})
            event = {**point, 'new_best': improved}
            if slot != 'best':
                event['slot'] = slot
            store.append_event({'kind': 'dev', 'counts': counts, 'summary': event})
            # One DEV at the end of the epoch when stopping is enabled.
            if termination is not None:
                return best
    return best


def reload_best(engine: Engine, best: Best | None, arm: dict[str, object],
                summary: dict[str, object], checkpoint_bytes: int, meter: Meter, store: RunStore,
                *, phase: Callable[[str], None] = _no_phase) -> None:
    phase('BEST_RELOAD')
    assert_source(store)
    meter.check()
    if best is None:
        raise ValueError('missing best checkpoint')
    ref, digest = best
    raw = store.read_model(ref, digest, checkpoint_bytes)
    meter.check()
    engine.restore(raw)
    del raw
    assert_source(store)
    meter.check()
    arm['best_reloaded'] = True
    store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})


def reload_then[T](arms: Sequence[tuple[Engine, Best | None, dict[str, object]]],
                   summary: dict[str, object], settings: FitSettings, meter: Meter, store: RunStore,
                   label_test: Callable[[], T], *, phase: Callable[[str], None] = _no_phase) -> T:
    """Two distinct arms, both strictly reloaded before crossing the TEST boundary."""
    if (len(arms) != 2 or arms[0][0] is arms[1][0] or arms[0][1] is None or arms[1][1] is None
            or arms[0][1][0] == arms[1][1][0]):
        raise ValueError('two distinct persisted best arms required')
    for engine, best, arm in arms:
        reload_best(engine, best, arm, summary, settings.checkpoint_bytes, meter, store,
                    phase=phase)
    phase('TEST')
    assert_source(store)
    meter.check()
    return label_test()

"""One cube provider, serial fits, and an all-or-nothing TEST score."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter
from cipher_lab.training.early_stopping import Settings

from ..fit import (
    Best,
    Engine,
    FitSettings,
    assert_source,
    batch_indices,
    fit,
    predict_batch,
    reload_best,
    reload_then,
    seeded_order,
)
from . import config
from .data import Panel
from .metrics import LSTMScore, PairedScore, train_majority

Arm = Literal['mlp', 'sharedm']


class Panels(Protocol):
    def build(self, split: Literal['train', 'dev']) -> Panel: ...
    def label_test(self) -> Panel: ...


@dataclass(frozen=True, slots=True)
class Plan:
    """Production uses these defaults; small plans are only for injected tests."""

    families: tuple[int, int, int] = config.FAMILIES
    epochs: int = config.EPOCHS
    batch_size: int = config.BATCH
    early_stopping: Settings | None = None
    topology: Literal['dual', 'lstm-single'] = 'dual'


def _completed_epochs(summary: dict[str, object], plan: Plan, name: str) -> int:
    arms = summary.get('arms')
    if not isinstance(arms, dict):
        raise ValueError('missing completed arms')
    arm = arms.get(name)
    if not isinstance(arm, dict):
        raise ValueError('missing completed arm')
    epochs, reason, dev = arm.get('epochs_completed'), arm.get('stop_reason'), arm.get('dev')
    stop = plan.early_stopping
    if (type(epochs) is not int or not 1 <= epochs <= plan.epochs or stop is None
            or not isinstance(dev, list) or len(dev) != epochs
            or (reason != 'EPOCH_CEILING' if epochs == plan.epochs else
                reason != 'EARLY_STOPPING' or epochs < stop.warmup_epochs + stop.patience)):
        raise ValueError('invalid completed epochs')
    return epochs


def completed_epochs(summary: dict[str, object], plan: Plan) -> tuple[int, int]:
    return _completed_epochs(summary, plan, 'mlp'), _completed_epochs(summary, plan, 'sharedm')


def completed_lstm_epochs(summary: dict[str, object], plan: Plan) -> int:
    arms = summary.get('arms')
    if plan.topology != 'lstm-single' or not isinstance(arms, dict) or set(arms) != {'lstm'}:
        raise ValueError('expected one completed LSTM arm')
    return _completed_epochs(summary, plan, 'lstm')


def _quiet(phase: str) -> None:
    pass


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        make_panels: Callable[[Meter], Panels], make_engine: Callable[[Arm], Engine] | None = None,
        make_lstm_engine: Callable[[], Engine] | None = None,
        plan: Plan = Plan(), phase: Callable[[str], None] = _quiet) -> dict[str, object]:
    """Return one complete score only; never attach a TEST score to live snapshots."""
    single = plan.topology == 'lstm-single'
    roster: tuple[tuple[str, str], ...]
    if single and make_lstm_engine is not None and make_engine is None:
        if plan.early_stopping is None:
            raise ValueError('LSTM requires epoch early stopping')
        roster = (('lstm', config.LSTM_SLOT),)
    elif plan.topology == 'dual' and make_engine is not None and make_lstm_engine is None:
        roster = tuple(zip(config.ARMS, config.SLOTS, strict=True))
    else:
        raise ValueError('invalid cube fit topology')
    train_rows, dev_rows, test_rows = (n * config.PAIRS_PER_FAMILY for n in plan.families)
    settings = FitSettings(plan.epochs, plan.batch_size, config.CHECKPOINT_BYTES,
                           config.ACTIVE_COLUMNS, early_stopping=plan.early_stopping)

    def progress(stage: str) -> None:
        phase(stage)
        # A read/write can seal the store. Persist attempted/completed counts
        # before such boundaries, never a proposed TEST result.
        if stage in ('BEST_SAVE', 'BEST_RELOAD'):
            store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})

    def archive(split: str, panel: Panel) -> None:
        assert_source(store)
        meter.check()
        raw = panel.archive()
        meter.check()
        store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})
        store.write_dataset(split, raw)
        meter.check()

    phase('PANELS')
    assert_source(store)
    meter.check()
    panels = make_panels(meter)
    train = panels.build('train')
    if len(train) != train_rows:
        raise ValueError('TRAIN panel size')
    archive('train', train)
    dev = panels.build('dev')
    if len(dev) != dev_rows:
        raise ValueError('DEV panel size')
    archive('dev', dev)
    majority = train_majority(train.label_batch(at) for at in batch_indices(train_rows,
                                                                            plan.batch_size))
    order = seeded_order(train_rows, config.ORDER_SEED)
    arms: dict[str, dict[str, object]] = {}
    summary['arms'] = arms
    reloads: list[tuple[Engine, Best | None, dict[str, object]]] = []
    for name, slot in roster:
        phase('ENGINE')
        assert_source(store)
        meter.check()
        if make_lstm_engine is not None:
            engine = make_lstm_engine()
            method = config.LSTM_METHOD
        else:
            assert make_engine is not None
            engine = make_engine(cast(Arm, name))
            method = config.METHOD  # preserve legacy identities in both dual recipes
        arm: dict[str, object] = {
            'identity': f'{method}/{name}', 'slot': slot, 'best_reloaded': False,
        }
        arms[name] = arm
        best = fit(train, dev, engine, order, slot, arm, summary, settings,
                   meter, store, phase=progress)
        reloads.append((engine, best, arm))

    if single:
        [(engine, best, arm)] = reloads
        reload_best(engine, best, arm, summary, settings.checkpoint_bytes, meter, store,
                    phase=progress)
        phase('TEST')
        assert_source(store)
        meter.check()
        test = panels.label_test()
    else:
        test = reload_then(reloads, summary, settings, meter, store, panels.label_test,
                           phase=progress)
    if len(test) != test_rows:
        raise ValueError('TEST panel size')
    phase('TEST_SCORE')
    score = LSTMScore(test_rows, majority) if single else PairedScore(test_rows, majority)
    for at in batch_indices(test_rows, plan.batch_size):
        x, y = test.batch(at)
        first = predict_batch(reloads[0][0], x, y, meter)
        if isinstance(score, LSTMScore):
            score.add(test.family_ids[at], y, first)
        else:
            sharedm = predict_batch(reloads[1][0], x, y, meter)
            score.add(test.family_ids[at], y, first, sharedm)
    meter.check()
    result = score.result()
    phase('TEST_SAVE')
    archive('test', test)
    return result

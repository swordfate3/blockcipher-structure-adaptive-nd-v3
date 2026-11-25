"""One cube provider, serial fits, and an all-or-nothing TEST score."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter
from cipher_lab.training.early_stopping import Settings

from ..fit import (
    BatchPanel,
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
from .data import Indices, Labels, Panel, Words
from .direct_metrics import DirectScore
from .direct_metrics import train_majority as train_direct_majority
from .low_metrics import LowScore
from .low_metrics import train_majority as train_full_majority
from .metrics import LSTMScore, PairedScore, train_majority

Arm = Literal['mlp', 'sharedm']


class Panels(Protocol):
    def build(self, split: Literal['train', 'dev']) -> Panel: ...
    def label_test(self) -> Panel: ...


class ArchivedPanel(BatchPanel, Protocol):
    @property
    def family_ids(self) -> Words: ...
    def archive(self) -> bytes: ...
    def label_batch(self, indices: Indices) -> Labels: ...


class LowPanels(Protocol):
    def build(self, split: Literal['train', 'dev']) -> tuple[ArchivedPanel, ...]: ...
    def label_test(self) -> tuple[ArchivedPanel, ...]: ...


@dataclass(frozen=True, slots=True)
class Plan:
    """Production uses these defaults; small plans are only for injected tests."""

    families: tuple[int, int, int] = config.FAMILIES
    epochs: int = config.EPOCHS
    batch_size: int = config.BATCH
    early_stopping: Settings | None = None
    topology: Literal['dual', 'lstm-single', 'lstm-r1r3', 'direct64-r1r3'] = 'dual'


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


def completed_low_epochs(summary: dict[str, object], plan: Plan) -> tuple[int, int, int]:
    arms = summary.get('arms')
    if (plan.topology not in ('lstm-r1r3', 'direct64-r1r3') or not isinstance(arms, dict)
            or set(arms) != set(config.LOW_NAMES)):
        raise ValueError('expected three completed low-round arms')
    return (_completed_epochs(summary, plan, 'r1'), _completed_epochs(summary, plan, 'r2'),
            _completed_epochs(summary, plan, 'r3'))


def _quiet(phase: str) -> None:
    pass


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        make_panels: Callable[[Meter], Panels] | None = None,
        make_engine: Callable[[Arm], Engine] | None = None,
        make_lstm_engine: Callable[[], Engine] | None = None,
        make_low_panels: Callable[[Meter], LowPanels] | None = None,
        make_low_engine: Callable[[config.LowRound], Engine] | None = None,
        make_direct_panels: Callable[[Meter], LowPanels] | None = None,
        make_direct_engine: Callable[[config.LowRound], Engine] | None = None,
        plan: Plan = Plan(), phase: Callable[[str], None] = _quiet) -> dict[str, object]:
    """Return one complete score only; never attach a TEST score to live snapshots."""
    single = plan.topology == 'lstm-single'
    direct = plan.topology == 'direct64-r1r3'
    matrix = plan.topology in ('lstm-r1r3', 'direct64-r1r3')
    if direct:
        if make_low_panels is not None or make_low_engine is not None:
            raise ValueError('invalid direct64 topology')
        matrix_provider, matrix_engine = make_direct_panels, make_direct_engine
    else:
        if make_direct_panels is not None or make_direct_engine is not None:
            raise ValueError('invalid direct64 topology')
        matrix_provider, matrix_engine = make_low_panels, make_low_engine
    roster: tuple[tuple[str, str], ...]
    if matrix:
        if (matrix_provider is None or matrix_engine is None or plan.early_stopping is None
                or any(fn is not None for fn in (make_panels, make_engine, make_lstm_engine))):
            raise ValueError('invalid low-round fit topology')
        roster = tuple(zip(config.LOW_NAMES, config.LOW_SLOTS, strict=True))
    elif make_panels is None or matrix_provider is not None or matrix_engine is not None:
        raise ValueError('invalid cube panel topology')
    elif single and make_lstm_engine is not None and make_engine is None:
        if plan.early_stopping is None:
            raise ValueError('LSTM requires epoch early stopping')
        roster = (('lstm', config.LSTM_SLOT),)
    elif plan.topology == 'dual' and make_engine is not None and make_lstm_engine is None:
        roster = tuple(zip(config.ARMS, config.SLOTS, strict=True))
    else:
        raise ValueError('invalid cube fit topology')
    rows_per_family = config.PAIRS_PER_FAMILY * (2 if direct else 1)
    train_rows, dev_rows, test_rows = (n * rows_per_family for n in plan.families)
    settings = FitSettings(plan.epochs, plan.batch_size, config.CHECKPOINT_BYTES,
                           None if direct else config.ACTIVE_COLUMNS,
                           early_stopping=plan.early_stopping)

    def progress(stage: str) -> None:
        phase(stage)
        # A read/write can seal the store. Persist attempted/completed counts
        # before such boundaries, never a proposed TEST result.
        if stage in ('BEST_SAVE', 'BEST_RELOAD'):
            store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})

    def archive(split: str, panel: ArchivedPanel) -> None:
        assert_source(store)
        meter.check()
        raw = panel.archive()
        meter.check()
        store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})
        store.write_dataset(split, raw)
        meter.check()

    def prepare(split: str, panel: ArchivedPanel, rows: int) -> ArchivedPanel:
        if len(panel) != rows:
            raise ValueError(f'{split.upper()} panel size')
        archive(split, panel)
        return panel

    def matrix_train_labels(panel: ArchivedPanel) -> Iterator[Labels]:
        for at in batch_indices(len(panel), plan.batch_size):
            meter.check()
            yield panel.label_batch(at)
        meter.check()  # also check the last reduction before another phase

    phase('PANELS')
    assert_source(store)
    meter.check()
    panels: Panels | None = None
    matrix_panels: LowPanels | None = None
    majorities: tuple[tuple[int, ...], ...] = ()
    majority: tuple[int, ...] = ()
    if matrix:
        assert matrix_provider is not None
        matrix_panels = matrix_provider(meter)
        trains = tuple(prepare(f'{name}-train', panel, train_rows) for (name, _), panel in
                       zip(roster, matrix_panels.build('train'), strict=True))
        devs = tuple(prepare(f'{name}-dev', panel, dev_rows) for (name, _), panel in
                     zip(roster, matrix_panels.build('dev'), strict=True))
        reducer = train_direct_majority if direct else train_full_majority
        majorities = tuple(reducer(matrix_train_labels(panel)) for panel in trains)
    else:
        assert make_panels is not None
        panels = make_panels(meter)
        train = prepare('train', panels.build('train'), train_rows)
        dev = prepare('dev', panels.build('dev'), dev_rows)
        trains, devs = (train,) * len(roster), (dev,) * len(roster)
        majority = train_majority(train.label_batch(at) for at in
                                  batch_indices(train_rows, plan.batch_size))
    order = seeded_order(train_rows, config.ORDER_SEED)
    arms: dict[str, dict[str, object]] = {}
    summary['arms'] = arms
    reloads: list[tuple[Engine, Best | None, dict[str, object]]] = []
    for i, ((name, slot), train, dev) in enumerate(zip(roster, trains, devs, strict=True)):
        phase('ENGINE')
        assert_source(store)
        meter.check()
        if matrix_engine is not None:
            engine = matrix_engine(config.LOW_ROUNDS[i])
            method = config.DIRECT_METHOD if direct else config.LOW_METHOD
            if any(engine is prior for prior, _, _ in reloads):
                raise ValueError('expected independent low-round engines')
        elif make_lstm_engine is not None:
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

    if matrix_panels is not None:
        for (engine, best, arm), slot in zip(reloads, config.LOW_SLOTS, strict=True):
            if best is None or best[0] != f'model/{slot}.pt':
                raise ValueError('missing or incorrect low-round best slot')
            reload_best(engine, best, arm, summary, settings.checkpoint_bytes, meter, store,
                        phase=progress)
        phase('TEST')
        assert_source(store)
        meter.check()
        tests = matrix_panels.label_test()
        phase('TEST_SCORE')
        scores: dict[str, object] = {}
        for name, rounds, (engine, _, _), panel, bits in zip(
            config.LOW_NAMES, config.LOW_ROUNDS, reloads, tests, majorities, strict=True,
        ):
            if len(panel) != test_rows:
                raise ValueError('TEST panel size')
            matrix_score = (DirectScore(test_rows, bits, rounds) if direct else
                            LowScore(test_rows, bits, rounds))
            for at in batch_indices(test_rows, plan.batch_size):
                x, y = panel.batch(at)
                logits = predict_batch(engine, x, y, meter)
                if isinstance(matrix_score, DirectScore):
                    matrix_score.add(x, y, logits)
                else:
                    matrix_score.add(panel.family_ids[at], x, y, logits)
            meter.check()
            scores[name] = matrix_score.result()
        phase('TEST_SAVE')
        for name, panel in zip(config.LOW_NAMES, tests, strict=True):
            archive(f'{name}-test', panel)
        return {'rounds': scores}

    assert panels is not None
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

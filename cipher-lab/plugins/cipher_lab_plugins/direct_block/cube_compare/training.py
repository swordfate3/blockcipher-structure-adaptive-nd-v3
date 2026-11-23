"""One cube provider, serial fits, and an all-or-nothing paired TEST score."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter

from ..fit import (
    Best,
    Engine,
    FitSettings,
    assert_source,
    batch_indices,
    fit,
    predict_batch,
    reload_then,
    seeded_order,
)
from . import config
from .data import CubePanels, Panel
from .metrics import PairedScore, train_majority

Arm = Literal['mlp', 'sharedm']


@dataclass(frozen=True, slots=True)
class Plan:
    """Production uses these defaults; small plans are only for injected tests."""

    families: tuple[int, int, int] = config.FAMILIES
    epochs: int = config.EPOCHS
    batch_size: int = config.BATCH


def _quiet(phase: str) -> None:
    pass


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        make_engine: Callable[[Arm], Engine], make_panels: Callable[[Meter], CubePanels],
        plan: Plan = Plan(), phase: Callable[[str], None] = _quiet) -> dict[str, object]:
    """Return one complete score only; never attach a TEST score to live snapshots."""
    train_rows, dev_rows, test_rows = (n * config.PAIRS_PER_FAMILY for n in plan.families)
    settings = FitSettings(plan.epochs, plan.batch_size, config.CHECKPOINT_BYTES,
                           config.ACTIVE_COLUMNS)

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
    for name, slot in zip(config.ARMS, config.SLOTS, strict=True):
        phase('ENGINE')
        assert_source(store)
        meter.check()
        engine = make_engine(cast(Arm, name))
        arm: dict[str, object] = {
            'identity': f'{config.METHOD}/{name}', 'slot': slot, 'best_reloaded': False,
        }
        arms[name] = arm
        best = fit(train, dev, engine, order, slot, arm, summary, settings,
                   meter, store, phase=progress)
        reloads.append((engine, best, arm))

    test = reload_then(reloads, summary, settings, meter, store, panels.label_test, phase=progress)
    if len(test) != test_rows:
        raise ValueError('TEST panel size')
    phase('TEST_SCORE')
    score = PairedScore(test_rows, majority)
    for at in batch_indices(test_rows, plan.batch_size):
        x, y = test.batch(at)
        mlp = predict_batch(reloads[0][0], x, y, meter)
        sharedm = predict_batch(reloads[1][0], x, y, meter)
        score.add(test.family_ids[at], y, mlp, sharedm)
    meter.check()
    result = score.result()
    phase('TEST_SAVE')
    archive('test', test)
    return result

"""Run up to four matched rounds using the shared fit and restore owner."""
from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

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
)
from . import config, data, metrics


def _check_counts(meter: Meter, completed: int) -> None:
    expected = config.counts(completed)
    current = meter.snapshot()
    for kind, count in expected.items():
        if any(current[f'{status}_{kind}'] != count for status in ('attempted', 'completed')):
            raise ValueError('invalid round counters')


def _round(depth: int, meter: Meter, store: RunStore, summary: dict[str, object], *,
           panels: data.CyclePanels,
           make: Callable[[int, str, dict[str, object], int], Engine],
           ) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    panel = panels.for_round(depth)
    settings = FitSettings(config.EPOCHS, config.BATCH, config.CHECKPOINT_BYTES,
                           early_stopping=Settings(config.EPOCHS, 1, 0.0))
    arms: dict[str, dict[str, object]] = {}
    hashes: dict[str, dict[str, str]] = {}
    engines: dict[str, tuple[Engine, Best, dict[str, object]]] = {}
    labels: dict[int, data.Bytes] = {}
    summary.update(arms=arms, panel_hashes=hashes)

    def phase(name: str) -> None:
        summary['phase'] = name

    def load(repeat: int, split: str, rows: int) -> tuple[data.Panel, data.Bytes]:
        assert_source(store)
        meter.check()
        points = panel.points(repeat, split)
        if len(points) != rows:
            raise ValueError('wrong panel length')
        target = panel.label(repeat, split)
        encoded = data.Panel(points, target)
        hashes[f'r{repeat + 1}'][split] = data.panel_digest(points, target)
        return encoded, target

    for repeat, seed in enumerate(config.INIT_SEEDS):
        phase('TRAIN_DEV_LABELS')
        panel_id = f'r{repeat + 1}'
        hashes[panel_id] = {}
        train, labels[repeat] = load(repeat, 'train', config.TRAIN)
        dev, _ = load(repeat, 'dev', config.DEV)

        def order(epoch: int, index: int = repeat) -> NDArray[np.intp]:
            sequence = np.random.SeedSequence([config.ORDER_SEED, index, epoch])
            return np.random.default_rng(sequence).permutation(config.TRAIN)

        for name in config.ARMS:
            slot = f'c{depth}-{panel_id}-{name}'
            summary['active_slot'] = slot
            arm: dict[str, object] = {'init_seed': seed}
            arms[slot] = arm
            phase('MODEL')
            meter.check()
            engine = make(depth, slot, arm, seed)
            init = arm.get('initial_sha256')
            if (arm.get('parameters') != config.PARAMETERS[depth] or not isinstance(init, str)
                    or len(init) != 64 or any(c not in '0123456789abcdef' for c in init)
                    or any(engine is old[0] for old in engines.values())
                    or (name == 'shuffled'
                        and init != arms[f'c{depth}-{panel_id}-correct']['initial_sha256'])):
                raise ValueError('invalid initial model identity')
            started = perf_counter()
            best = fit(train, dev, engine, order, slot, arm, summary, settings, meter, store,
                       phase=phase)
            arm['fit_elapsed_seconds'] = perf_counter() - started
            if (best is None or best[0] != f'model/{slot}.pt'
                    or arm.get('epochs_completed') != config.EPOCHS
                    or arm.get('stop_reason') != 'EPOCH_CEILING'):
                raise ValueError('incomplete fit')
            engines[slot] = engine, best, arm
    if tuple(engines) != tuple(f'c{depth}-{slot}' for slot in config.PANEL_SLOTS):
        raise ValueError('missing fit slots')
    for slot, (engine, best, metadata) in engines.items():
        summary['active_slot'] = slot
        reload_best(engine, best, metadata, summary, config.CHECKPOINT_BYTES, meter, store,
                    phase=phase)
    summary.pop('active_slot')

    phase('TEST')
    tests: dict[str, dict[str, object]] = {}
    scores: dict[str, dict[str, object]] = {}
    for repeat in range(config.REPEATS):
        panel_id = f'r{repeat + 1}'
        test, target = load(repeat, 'test', config.TEST)
        items: dict[str, object] = {'constant': metrics.constant_score(labels[repeat], target)}
        for name in config.ARMS:
            subslot = f'{panel_id}-{name}'
            engine = engines[f'c{depth}-{subslot}'][0]
            logits = np.concatenate([predict_batch(engine, *test.batch(at), meter)
                                     for at in batch_indices(len(test), config.BATCH)])
            items[name] = scores[subslot] = metrics.score(logits, target)
        tests[panel_id] = items
    return {'arms': arms, 'test': tests, 'panel_hashes': hashes}, scores


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        gate: Callable[[Meter], None], make_panels: Callable[[Meter], data.CyclePanels],
        make: Callable[[int, str, dict[str, object], int], Engine],
        read_snapshot: Callable[[], dict[str, object]],
        peaks: Callable[[], dict[str, int]] | None = None) -> None:
    completed: dict[str, dict[str, object]] = {}
    summary.update(method=config.METHOD, phase='KAT', rounds=completed)
    assert_source(store)
    gate(meter)
    _check_counts(meter, 0)
    summary['public_kat'] = 'PASSED'
    store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})
    summary['phase'] = 'PANELS'
    panels = make_panels(meter)
    _check_counts(meter, 0)
    for depth in config.ROUNDS:
        meter.check()
        summary.update(phase='ROUND_START', current_round=depth)
        store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})
        result, scores = _round(depth, meter, store, summary, panels=panels, make=make)
        _check_counts(meter, len(completed) + 1)
        rid = str(depth)
        completed[rid] = result
        summary.pop('arms')
        summary.pop('panel_hashes')
        summary.update(phase='ROUND_SAVED', last_completed_round=depth)
        store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})
        stored = cast('dict[str, Any]', read_snapshot())
        if stored['status'] != 'RUNNING' or stored['summary']['rounds'][rid] != result:
            raise ValueError('saved round mismatch')
        decision = metrics.classify(depth, scores)
        result['decision'] = decision
        store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})
        if decision == 'CONTINUE':
            continue
        summary.update(phase='COMPLETE', stop_reason=decision)
        if peaks is not None:
            summary['gpu_peaks'] = peaks()
        assert_source(store)
        meter.check()
        store.finalize({'status': 'SUCCEEDED', 'counts': meter.snapshot(), 'summary': summary})
        return
    raise ValueError('missing terminal round decision')

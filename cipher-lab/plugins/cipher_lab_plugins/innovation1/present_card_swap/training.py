"""Fit two arms on one panel, diagnose both restored bests, then score shared TEST."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from typing import TYPE_CHECKING

import numpy as np

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter
from cipher_lab.training.early_stopping import Settings
from cipher_lab_plugins.direct_block.fit import (
    Best,
    FitSettings,
    assert_source,
    batch_indices,
    fit,
    predict_batch,
    reload_best,
    seeded_order,
)

from ..mask_aux.training import _counts, _hashes, _pair_scores, _require_capacity
from ..prefix_bridge.metrics import Score, compare, score
from ..present_card_fresh import config as pure
from . import config, diagnostics

if TYPE_CHECKING:
    from ..mask_aux_single.data import Panels
    from .runtime import FitRuntime


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        gate: Callable[[Meter], None], panels_for: Callable[[Meter], Panels],
        make: Callable[[str, dict[str, object], bool, int], FitRuntime],
        peaks: Callable[[], dict[str, int]]) -> None:
    store._failure_counts_supplier = meter.snapshot
    arms: dict[str, dict[str, object]] = {}
    bests: dict[str, Best] = {}
    epochs: list[int] = []
    summary['arms'] = arms

    def snapshot() -> None:
        meter.check()
        store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})
        meter.check()

    def phase(stage: str) -> None:
        summary['phase'] = stage
        if stage == 'BEST_SAVE':
            store.write_snapshot({'counts': meter.snapshot()})
            assert_source(store)
            meter.check()

    phase('KAT')
    assert_source(store)
    _require_capacity(meter, config.CAPS)
    _counts(meter, real_blocks=0)
    gate(meter)
    _counts(meter, real_blocks=pure.KAT_BLOCKS)
    summary['kat'] = dict(status='PASSED', name=pure.KAT_NAME, real_blocks=pure.KAT_BLOCKS)
    snapshot()

    phase('TRAIN_DEV_DATA')
    assert_source(store)
    meter.check()
    panels = panels_for(meter)
    train, dev = panels.train(config.SLOT), panels.dev(config.SLOT)
    if (len(train), len(dev)) != (config.TRAIN_ROWS, config.DEV_ROWS):
        raise ValueError('INVALID_PANEL_SIZE')
    hashes = dict(_hashes(panels, {'train', 'dev'}))
    summary['panel_hashes'] = hashes
    _counts(meter, **config.counts((), test=False))
    snapshot()

    settings = FitSettings(pure.EPOCHS, pure.BATCH_SIZE, pure.CHECKPOINT_BYTES,
                           early_stopping=Settings(pure.WARMUP, pure.PATIENCE, pure.MIN_DELTA))
    order = seeded_order(len(train), pure.ORDER_SEED)
    initial: tuple[str, dict[str, str]] | None = None
    for slot in config.SLOTS:
        summary['active_slot'] = slot
        phase('FIT_START')
        assert_source(store)
        meter.check()
        arm: dict[str, object] = dict(unit='pair', round=pure.ROUNDS[0])
        arms[slot] = arm
        origin = config.counts(tuple(epochs), test=False)['optimizer_updates']
        runtime = make(slot, arm, True, origin)
        try:
            identity = (runtime.initial_sha, dict(runtime.geometry))
            if (runtime.parameters != pure.PARAMETERS
                    or (initial is not None and identity != initial)):
                raise ValueError('MODEL_INITIAL_IDENTITY')
            initial = identity
            arm.update(initial_sha256=identity[0], geometry=identity[1], parameters=pure.PARAMETERS)
            chosen = fit(train, dev, runtime, order, slot, arm, summary, settings, meter, store,
                         phase=phase, batch_update=runtime.batch_update)
            n = arm.get('epochs_completed')
            if chosen is None or chosen[0] != f'model/{slot}.pt' or type(n) is not int:
                raise ValueError('FIT_INCOMPLETE')
            if not pure.MIN_EPOCHS <= n <= pure.EPOCHS or arm['stop_reason'] != (
                    'EPOCH_CEILING' if n == pure.EPOCHS else 'EARLY_STOPPING'):
                raise ValueError('FIT_STOP_REASON')
            bests[slot] = chosen
            epochs.append(n)
            _counts(meter, **config.counts(tuple(epochs), test=False))
        finally:
            runtime.release()

    before = config.counts(tuple(epochs), test=False)
    diagnosed = config.counts(tuple(epochs), test=False, diagnostic=True)
    totals = config.counts(tuple(epochs))
    results: dict[str, Score] = {}
    with ExitStack() as cleanup:
        restored: dict[str, FitRuntime] = {}
        for slot in config.SLOTS:
            summary['active_slot'] = slot
            phase('BEST_RELOAD')
            assert_source(store)
            meter.check()
            arm = arms[slot]
            engine = make(slot, arm, False, 0)
            cleanup.callback(engine.release)
            reload_best(engine, bests[slot], arm, summary, settings.checkpoint_bytes,
                        meter, store, phase=phase)
            if arm['best_reloaded'] is not True:
                raise ValueError('BEST_RELOAD_REQUIRED')
            restored[slot] = engine
        _counts(meter, **before)

        dev_results: dict[str, object] = {}
        for slot, engine in restored.items():
            summary['active_slot'] = slot
            phase('DEV_DIAGNOSTICS')
            assert_source(store)
            meter.check()
            engine.activate()
            dev_results[slot] = diagnostics.evaluate(engine, dev, settings.batch_size, meter)
        _counts(meter, **diagnosed)
        summary['dev_diagnostics'] = dev_results
        summary.pop('active_slot')
        snapshot()

        phase('TEST_DATA')
        assert_source(store)
        meter.check()
        panels.test()
        test = panels.view('test', config.SLOT)
        if len(test) != config.TEST_ROWS:
            raise ValueError('INVALID_PANEL_SIZE')
        final_hashes = _hashes(panels, {'train', 'dev', 'test'})
        if any(final_hashes[k] != value for k, value in hashes.items()):
            raise ValueError('PANEL_IDENTITY')
        _counts(meter, real_blocks=totals['real_blocks'],
                optimizer_updates=diagnosed['optimizer_updates'],
                forward_rows=diagnosed['forward_rows'])
        snapshot()

        y = np.empty((len(test), 1), dtype=np.uint8)
        for slot, engine in restored.items():
            summary['active_slot'] = slot
            phase('TEST_SCORE')
            assert_source(store)
            meter.check()
            logits = np.empty(y.shape, dtype=np.float32)
            for at in batch_indices(len(test), settings.batch_size):
                x, labels = test.batch(at)
                logits[at] = predict_batch(engine, x, labels, meter)
                if slot == config.SLOT:
                    y[at] = labels
                elif not np.array_equal(labels, y[at]):
                    raise ValueError('UNALIGNED_TEST_PAIRS')
            if int(y.sum()) != len(y) // 2:
                raise ValueError('UNBALANCED_TEST')
            results[slot] = score(logits, y)

    summary.pop('active_slot')
    _counts(meter, **totals)
    summary['cuda_peaks'] = peaks()
    snapshot()
    assert_source(store)
    meter.check()
    terminal = {**summary, 'phase': 'COMPLETE', 'panel_hashes': final_hashes,
                'test': _pair_scores(results), 'paired': {
                    config.CONTRAST: compare(results[config.SLOTS[1]], results[config.SLOT])}}
    store.finalize(dict(status='SUCCEEDED', counts=meter.snapshot(), summary=terminal))

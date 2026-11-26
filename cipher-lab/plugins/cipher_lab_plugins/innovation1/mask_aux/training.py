"""Two serial fresh fits; a shared holdout after both strict disk restores."""
from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Protocol

import numpy as np

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter, Stopped
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

from ..prefix_bridge import metrics
from ..prefix_bridge.data import BatchPanel
from ..prefix_bridge.training import ArmRuntime
from . import config
from .data import Panels


class Engine(ArmRuntime, Protocol):
    def take_aux_epoch(self) -> tuple[float, int]: ...


def _counts(meter: Meter, *, real_blocks: int, null_words: int = 0,
            optimizer_updates: int = 0, forward_rows: int = 0) -> None:
    ledger = meter.snapshot()
    expected = dict(real_blocks=real_blocks, null_words=null_words,
                    optimizer_updates=optimizer_updates, forward_rows=forward_rows)
    if any(ledger[f'{status}_{name}'] != n for name, n in expected.items()
           for status in ('attempted', 'completed')):
        raise ValueError('COUNT_MISMATCH')


def _require_capacity(meter: Meter) -> None:
    for kind, amount in config.counts(config.EPOCHS, config.EPOCHS).items():
        if getattr(meter._policy, kind) < amount:
            raise Stopped(f'{kind.upper()}_LIMIT')


def _hashes(panels: Panels, expected: set[str]) -> dict[str, str]:
    hashes = panels.hashes
    if (hashes.keys() != expected
            or any(re.fullmatch('[0-9a-f]{64}', h) is None for h in hashes.values())):
        raise ValueError('PANEL_IDENTITY')
    return hashes


def _score(panel: BatchPanel, engines: dict[str, Engine], meter: Meter,
           store: RunStore) -> dict[str, metrics.Score]:
    # Both arms get the very same panel in the same canonical order. Keep
    # per-group values only in memory, for paired correctness.
    labels = np.empty((len(panel), 1), dtype=np.uint8)
    scores: dict[str, metrics.Score] = {}
    for slot, engine in engines.items():
        assert_source(store)
        meter.check()
        engine.activate()
        logits = np.empty((len(panel), 1), dtype=np.float32)
        for at in batch_indices(len(panel), config.BATCH_SIZE):
            x, y = panel.batch(at)
            logits[at] = predict_batch(engine, x, y, meter)
            if slot == config.BASELINE:
                labels[at] = y
            elif not np.array_equal(labels[at], y):
                raise ValueError('UNALIGNED_TEST_GROUPS')
        meter.check()
        if int(labels.sum()) != len(panel) // 2:
            raise ValueError('UNBALANCED_TEST')
        scores[slot] = metrics.score(logits, labels)
    return scores


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        gate: Callable[[Meter], None], make: Callable[[str, dict[str, object], bool], Engine],
        make_panels: Callable[[Meter], Panels], peaks: Callable[[], dict[str, int]]) -> None:
    store._failure_counts_supplier = meter.snapshot

    def snapshot() -> None:
        # TEST aggregates and digest are never in this live mapping.
        store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})
        meter.check()

    def phase(stage: str) -> None:
        summary['phase'] = stage
        if stage == 'BEST_SAVE':
            snapshot()  # A storage failure may seal the owner; save paid work first.
            assert_source(store)
            meter.check()

    phase('KAT')
    assert_source(store)
    meter.check()
    _counts(meter, real_blocks=0)
    _require_capacity(meter)
    gate(meter)
    _counts(meter, real_blocks=config.KAT_BLOCKS)
    summary['kat'] = {'status': 'PASSED', 'name': config.KAT_NAME, 'real_blocks': config.KAT_BLOCKS}
    snapshot()

    phase('TRAIN_DEV_DATA')
    assert_source(store)
    meter.check()
    panels = make_panels(meter)
    train, dev = panels.train(config.PANEL), panels.dev(config.PANEL)
    if len(train) != config.TRAIN_ROWS or len(dev) != config.DEV_ROWS:
        raise ValueError('INVALID_PANEL_SIZE')
    prior_hashes = _hashes(panels, {'train', 'dev'})
    real_blocks = config.KAT_BLOCKS + config.blocks(len(train) + len(dev))
    _counts(meter, real_blocks=real_blocks)
    summary['panel_hashes'] = prior_hashes
    snapshot()

    stopping = Settings(config.WARMUP, config.PATIENCE, config.MIN_DELTA)
    settings = FitSettings(config.EPOCHS, config.BATCH_SIZE, config.CHECKPOINT_BYTES,
                           early_stopping=stopping)
    order = seeded_order(len(train), config.ORDER_SEED)
    arms: dict[str, dict[str, object]] = {}
    bests: dict[str, Best] = {}
    epochs: list[int] = []
    forward_rows = updates = 0
    classifier: tuple[str, int] | None = None
    summary['fits'] = arms

    def train_one(slot: str, arm: dict[str, object], engine: Engine) -> Best:
        aux: list[dict[str, object]] = []
        if slot == config.AUXILIARY:
            arm['aux_train'] = aux

        def fit_phase(stage: str) -> None:
            phase(stage)
            if stage == 'DEV' and slot == config.AUXILIARY:
                bce, rows = engine.take_aux_epoch()
                if not math.isfinite(bce) or bce < 0 or rows != len(train):
                    raise ValueError('INVALID_AUX_EPOCH')
                aux.append(dict(epoch=len(aux) + 1, train_bce=bce, masked_forward_rows=rows,
                                target_bits=rows * config.TARGETS_PER_ROW))

        best = fit(train, dev, engine, order, slot, arm, summary, settings, meter, store,
                   phase=fit_phase, batch_update=engine.batch_update)
        n = arm.get('epochs_completed')
        if best is None or type(n) is not int or not config.MIN_EPOCHS <= n <= config.EPOCHS:
            raise ValueError('FIT_INCOMPLETE')
        if best[0] != f'model/{slot}.pt' or (slot == config.AUXILIARY and len(aux) != n):
            raise ValueError('FIT_IDENTITY')
        epochs.append(n)
        return best

    for slot in config.SLOTS:
        phase('FIT_START')
        assert_source(store)
        meter.check()
        r, mask = config.slot_input(slot)
        arm: dict[str, object] = {'round': r, 'mask': mask}
        arms[slot] = arm
        engine = make(slot, arm, True)
        try:
            extra = config.AUX_PARAMETERS if slot == config.AUXILIARY else 0
            actual = engine.initial_sha, engine.parameters - extra
            if (re.fullmatch('[0-9a-f]{64}', actual[0]) is None or actual[1] <= 0
                    or (classifier is not None and actual != classifier)):
                raise ValueError('MODEL_INITIAL_IDENTITY')
            classifier = actual
            arm.update(initial_sha256=actual[0], parameters=engine.parameters)
            bests[slot] = train_one(slot, arm, engine)
        finally:
            engine.release()
        n = epochs[-1]
        train_rows, dev_rows = n * len(train), n * len(dev)
        masked_rows = train_rows if slot == config.AUXILIARY else 0
        steps = train_rows // settings.batch_size
        forward_rows += train_rows + dev_rows + masked_rows
        updates += steps
        _counts(meter, real_blocks=real_blocks, optimizer_updates=updates,
                forward_rows=forward_rows)
        arm['work'] = dict(train_class_rows=train_rows, dev_class_rows=dev_rows,
                           masked_forward_rows=masked_rows, aux_target_bits=(
                               masked_rows * config.TARGETS_PER_ROW), optimizer_updates=steps)
        snapshot()

    restored: dict[str, Engine] = {}
    try:
        for slot in config.SLOTS:
            phase('BEST_RELOAD')
            assert_source(store)
            meter.check()
            arm = arms[slot]
            engine = make(slot, arm, False)
            restored[slot] = engine
            reload_best(engine, bests[slot], arm, summary, settings.checkpoint_bytes,
                        meter, store, phase=phase)
            if (engine.initial_sha != arm['initial_sha256']
                    or engine.parameters != arm['parameters']):
                raise ValueError('MODEL_INITIAL_IDENTITY')
        _counts(meter, real_blocks=real_blocks, optimizer_updates=updates,
                forward_rows=forward_rows)
        phase('TEST_DATA')
        assert_source(store)
        meter.check()
        panels.test()
        panel = panels.view('test', config.PANEL)
        hashes = _hashes(panels, {'train', 'dev', 'test'})
        if (len(panel) != config.TEST_ROWS
                or any(hashes[k] != digest for k, digest in prior_hashes.items())):
            raise ValueError('PANEL_IDENTITY')
        totals = config.counts(*epochs)
        _counts(meter, real_blocks=totals['real_blocks'], optimizer_updates=updates,
                forward_rows=forward_rows)
        snapshot()
        phase('TEST_SCORE')
        scores = _score(panel, restored, meter, store)
    finally:
        for engine in restored.values():
            engine.release()

    phase('FINAL_ACCOUNTING')
    _counts(meter, **totals)
    snapshot()  # Also protect paid TEST work if a late check/commit fails.
    summary['cuda_peaks'] = peaks()
    meter.check()
    paired = metrics.compare(scores[config.AUXILIARY], scores[config.BASELINE])
    test = {'unit': 'group', 'by_slot': {k: v.saved for k, v in scores.items()},
            'forward_rows_by_slot': dict.fromkeys(config.SLOTS, len(panel)), 'b_vs_a': paired}
    phase('FINALIZE')
    assert_source(store)
    meter.check()
    store.finalize({'status': 'SUCCEEDED', 'counts': meter.snapshot(), 'summary': {
        **summary, 'phase': 'COMPLETE', 'panel_hashes': hashes, 'test': test,
        'shared_c7_outputs': totals['real_blocks'] - config.KAT_BLOCKS,
    }})

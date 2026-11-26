"""Two serial fresh fits; a shared holdout after both strict disk restores."""
from __future__ import annotations

import math
import re
from collections.abc import Callable
from contextlib import suppress
from typing import Literal, Protocol, cast

import numpy as np

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter, Stopped
from cipher_lab.training.early_stopping import Settings
from cipher_lab_plugins.direct_block.fit import (
    BatchPanel,
    Best,
    FitSettings,
    Floats,
    Labels,
    assert_source,
    batch_indices,
    fit,
    predict_batch,
    reload_best,
    seeded_order,
)

from ..prefix_bridge import metrics
from ..prefix_bridge.config import Counts
from ..prefix_bridge.training import ArmRuntime
from . import config


class Engine(ArmRuntime, Protocol):
    def take_aux_epoch(self) -> tuple[float, int]: ...


class _Panels(Protocol):
    @property
    def hashes(self) -> dict[str, str]: ...
    def train(self, slot: str) -> BatchPanel: ...
    def dev(self, slot: str) -> BatchPanel: ...
    def test(self) -> None: ...
    def view(self, split: Literal['test'], slot: str) -> BatchPanel: ...


class _Recipe(Protocol):
    SLOTS: tuple[str, str]
    BASELINE: str
    AUXILIARY: str
    PANEL: str
    EPOCHS: int
    BATCH_SIZE: int
    CHECKPOINT_BYTES: int
    ORDER_SEED: int
    KAT_NAME: str
    KAT_BLOCKS: int
    TRAIN_ROWS: int
    DEV_ROWS: int
    TEST_ROWS: int
    AUX_PARAMETERS: int
    TARGETS_PER_ROW: int

    def counts(self, a: int, b: int) -> Counts: ...
    def blocks(self, rows: int) -> int: ...
    def slot_input(self, slot: str) -> tuple[int, int]: ...


def _counts(meter: Meter, *, real_blocks: int, null_words: int = 0,
            optimizer_updates: int = 0, forward_rows: int = 0) -> None:
    ledger = meter.snapshot()
    expected = dict(real_blocks=real_blocks, null_words=null_words,
                    optimizer_updates=optimizer_updates, forward_rows=forward_rows)
    if any(ledger[f'{status}_{name}'] != n for name, n in expected.items()
           for status in ('attempted', 'completed')):
        raise ValueError('COUNT_MISMATCH')


def _require_capacity(meter: Meter, totals: Counts) -> None:
    for kind, amount in totals.items():
        if getattr(meter._policy, kind) < amount:
            raise Stopped(f'{kind.upper()}_LIMIT')


def _hashes(panels: _Panels, expected: set[str]) -> dict[str, str]:
    hashes = panels.hashes
    if (hashes.keys() != expected
            or any(re.fullmatch('[0-9a-f]{64}', h) is None for h in hashes.values())):
        raise ValueError('PANEL_IDENTITY')
    return hashes


def _score(panel: BatchPanel, engines: dict[str, Engine], meter: Meter,
           store: RunStore, recipe: _Recipe) -> dict[str, metrics.Score]:
    # Both arms get the very same panel in the same canonical order. Keep
    # per-row values only in memory, for paired correctness.
    labels = np.empty((len(panel), 1), dtype=np.uint8)
    scores: dict[str, metrics.Score] = {}
    for slot, engine in engines.items():
        assert_source(store)
        meter.check()
        engine.activate()
        logits = np.empty((len(panel), 1), dtype=np.float32)
        for at in batch_indices(len(panel), recipe.BATCH_SIZE):
            x, y = panel.batch(at)
            logits[at] = predict_batch(engine, x, y, meter)
            if slot == recipe.BASELINE:
                labels[at] = y
            elif not np.array_equal(labels[at], y):
                raise ValueError('UNALIGNED_TEST_GROUPS')
        meter.check()
        if int(labels.sum()) != len(panel) // 2:
            raise ValueError('UNBALANCED_TEST')
        scores[slot] = metrics.score(logits, labels)
    return scores


def _pair_scores(scores: dict[str, metrics.Score]) -> dict[str, object]:
    result: dict[str, object] = {}
    for slot, score in scores.items():
        saved = dict(score.saved)
        saved['pairs'] = saved.pop('groups')
        saved['correct_pairs'] = saved.pop('correct_groups')
        saved['by_label'] = {
            k: {'pairs': v['groups'], 'correct': v['correct']}
            for k, v in cast(dict[str, dict[str, int]], saved['by_label']).items()
        }
        result[slot] = saved
    return result


class _PairFitWork:
    """Partition a single fit's deltas from the existing Meter, not a second ledger."""

    def __init__(self, start: dict[str, object]) -> None:
        self._start = start
        self._train = [0, 0]
        self._masked = [0, 0]

    def batch(self, update: Callable[[Floats, Labels, Meter], float],
              x: Floats, y: Labels, meter: Meter) -> float:
        before = meter.snapshot()
        try:
            return update(x, y, meter)
        finally:
            after = meter.snapshot()
            for i, status in enumerate(('attempted', 'completed')):
                key = f'{status}_forward_rows'
                rows = cast(int, after[key]) - cast(int, before[key])
                # The batch seam reserves/completes one full view first. Only
                # then can it reserve/complete the same number of masked rows.
                full = min(rows, len(x))
                self._train[i] += full
                self._masked[i] += rows - full

    def snapshot(self, after: dict[str, object]) -> tuple[dict[str, int], dict[str, float]]:
        work: dict[str, int] = {}
        for i, status in enumerate(('attempted', 'completed')):
            key = f'{status}_forward_rows'
            total = cast(int, after[key]) - cast(int, self._start[key])
            update_key = f'{status}_optimizer_updates'
            prefix = 'attempted_' if i == 0 else ''
            work[f'{prefix}train_class_rows'] = self._train[i]
            work[f'{prefix}masked_forward_rows'] = self._masked[i]
            # Within fit, every forward outside the training callback is DEV.
            work[f'{prefix}dev_class_rows'] = total - self._train[i] - self._masked[i]
            work[f'{prefix}optimizer_updates'] = (
                cast(int, after[update_key]) - cast(int, self._start[update_key]))
        times = {f'{kind}_seconds': cast(float, after[f'{kind}_elapsed_seconds'])
                 - cast(float, self._start[f'{kind}_elapsed_seconds']) for kind in ('wall', 'cpu')}
        return work, times


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        gate: Callable[[Meter], None], make: Callable[[str, dict[str, object], bool], Engine],
        make_panels: Callable[[Meter], _Panels], peaks: Callable[[], dict[str, int]],
        _pair_recipe: _Recipe | None = None) -> None:
    pair = _pair_recipe is not None
    recipe: _Recipe = config if _pair_recipe is None else _pair_recipe
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

    timing: dict[str, object] = {}
    if pair:
        summary['timing'] = timing

    def elapsed() -> tuple[float, float]:
        if not pair:
            return 0., 0.
        meter.check()
        ledger = meter.snapshot()
        return (cast(float, ledger['wall_elapsed_seconds']),
                cast(float, ledger['cpu_elapsed_seconds']))

    def duration(since: tuple[float, float]) -> dict[str, float]:
        wall, cpu = elapsed()
        return {'wall_seconds': wall - since[0], 'cpu_seconds': cpu - since[1]}

    phase('KAT')
    assert_source(store)
    meter.check()
    _counts(meter, real_blocks=0)
    ceiling = recipe.counts(recipe.EPOCHS, recipe.EPOCHS)
    _require_capacity(meter, ceiling)
    started = elapsed()
    gate(meter)
    if pair:
        timing['gate'] = duration(started)
    _counts(meter, real_blocks=recipe.KAT_BLOCKS)
    summary['kat'] = {'status': 'PASSED', 'name': recipe.KAT_NAME, 'real_blocks': recipe.KAT_BLOCKS}
    snapshot()

    phase('TRAIN_DEV_DATA')
    assert_source(store)
    meter.check()
    started = elapsed()
    panels = make_panels(meter)
    if pair:
        timing['train_dev_data'] = duration(started)
    train, dev = panels.train(recipe.PANEL), panels.dev(recipe.PANEL)
    if len(train) != recipe.TRAIN_ROWS or len(dev) != recipe.DEV_ROWS:
        raise ValueError('INVALID_PANEL_SIZE')
    prior_hashes = _hashes(panels, {'train', 'dev'})
    real_blocks = recipe.KAT_BLOCKS + recipe.blocks(len(train) + len(dev))
    _counts(meter, real_blocks=real_blocks)
    summary['panel_hashes'] = prior_hashes
    snapshot()

    stopping = None if pair else Settings(config.WARMUP, config.PATIENCE, config.MIN_DELTA)
    min_epochs = recipe.EPOCHS if pair else config.MIN_EPOCHS
    settings = FitSettings(recipe.EPOCHS, recipe.BATCH_SIZE, recipe.CHECKPOINT_BYTES,
                           early_stopping=stopping, allow_partial_batch=pair)
    order = seeded_order(len(train), recipe.ORDER_SEED)
    arms: dict[str, dict[str, object]] = {}
    bests: dict[str, Best] = {}
    epochs: list[int] = []
    forward_rows = updates = 0
    classifier: tuple[str, int] | None = None
    summary['fits'] = arms

    def train_one(slot: str, arm: dict[str, object], engine: Engine,
                  work: _PairFitWork | None) -> Best:
        aux: list[dict[str, object]] = []
        if slot == recipe.AUXILIARY:
            arm['aux_train'] = aux

        def fit_phase(stage: str) -> None:
            phase(stage)
            if stage == 'DEV' and slot == recipe.AUXILIARY:
                bce, rows = engine.take_aux_epoch()
                if not math.isfinite(bce) or bce < 0 or rows != len(train):
                    raise ValueError('INVALID_AUX_EPOCH')
                aux.append(dict(epoch=len(aux) + 1, train_bce=bce, masked_forward_rows=rows,
                                target_bits=rows * recipe.TARGETS_PER_ROW))

        def metered_batch(x: Floats, y: Labels, m: Meter) -> float:
            assert work is not None
            return work.batch(engine.batch_update, x, y, m)

        best = fit(train, dev, engine, order, slot, arm, summary, settings, meter, store,
                   phase=fit_phase, batch_update=(engine.batch_update if work is None else
                                                 metered_batch))
        n = arm.get('epochs_completed')
        if best is None or type(n) is not int or not min_epochs <= n <= recipe.EPOCHS:
            raise ValueError('FIT_INCOMPLETE')
        if best[0] != f'model/{slot}.pt' or (slot == recipe.AUXILIARY and len(aux) != n):
            raise ValueError('FIT_IDENTITY')
        epochs.append(n)
        return best

    fit_times: dict[str, dict[str, float]] = {}
    fit_evidence: dict[str, dict[str, object]] = {}
    active: tuple[str, _PairFitWork, dict[str, object]] | None = None

    def failure_pair_work() -> dict[str, dict[str, object]]:
        evidence = dict(fit_evidence)
        if active is not None:
            slot, tracker, arm = active
            counters, times = tracker.snapshot(meter.snapshot())
            completed = arm.get('epochs_completed', 0)
            evidence[slot] = {**counters, **times, 'epochs_completed': completed}
        return evidence

    if pair:
        timing['fits'] = fit_times
        store._failure_pair_fit_supplier = failure_pair_work
    for slot in recipe.SLOTS:
        phase('FIT_START')
        assert_source(store)
        meter.check()
        r, mask = recipe.slot_input(slot)
        arm: dict[str, object] = {'round': r, 'mask': mask}
        arms[slot] = arm
        if pair:
            meter.check()
        work = _PairFitWork(meter.snapshot()) if pair else None
        if work is not None:
            active = slot, work, arm
        try:
            engine = make(slot, arm, True)
            succeeded = False
            try:
                extra = recipe.AUX_PARAMETERS if slot == recipe.AUXILIARY else 0
                actual = engine.initial_sha, engine.parameters - extra
                if (re.fullmatch('[0-9a-f]{64}', actual[0]) is None or actual[1] <= 0
                        or (classifier is not None and actual != classifier)):
                    raise ValueError('MODEL_INITIAL_IDENTITY')
                classifier = actual
                arm.update(initial_sha256=actual[0], parameters=engine.parameters)
                bests[slot] = train_one(slot, arm, engine, work)
                succeeded = True
            finally:
                if pair and not succeeded:
                    # Preserve the first fit fault, including an interrupt.
                    with suppress(Exception, KeyboardInterrupt):
                        engine.release()
                else:
                    engine.release()
            if pair:
                meter.check()  # retain the success-path boundary check
        finally:
            # Do not sample clocks or check budgets while preserving a failure.
            if work is not None:
                counters, fit_times[slot] = work.snapshot(meter.snapshot())
                completed = arm.setdefault('epochs_completed', 0)
                fit_evidence[slot] = {**counters, **fit_times[slot], 'epochs_completed': completed}
                active = None
                # Sites associated with validated masked views, not confirmed loss/updates.
                arm['work'] = {**counters, 'aux_target_bits': (
                    counters['masked_forward_rows'] * recipe.TARGETS_PER_ROW)}
        n = epochs[-1]
        train_rows, dev_rows = n * len(train), n * len(dev)
        masked_rows = train_rows if slot == recipe.AUXILIARY else 0
        steps = n * ((len(train) + settings.batch_size - 1) // settings.batch_size)
        forward_rows += train_rows + dev_rows + masked_rows
        updates += steps
        _counts(meter, real_blocks=real_blocks, optimizer_updates=updates,
                forward_rows=forward_rows)
        expected_work = dict(train_class_rows=train_rows, dev_class_rows=dev_rows,
                             masked_forward_rows=masked_rows, aux_target_bits=(
                                 masked_rows * recipe.TARGETS_PER_ROW), optimizer_updates=steps)
        if pair:
            expected_work.update({f'attempted_{key}': count for key, count in expected_work.items()
                                  if key != 'aux_target_bits'})
            if arm['work'] != expected_work:
                raise ValueError('COUNT_MISMATCH')
        else:
            arm['work'] = expected_work
        snapshot()

    restored: dict[str, Engine] = {}
    started = elapsed()
    try:
        for slot in recipe.SLOTS:
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
        if pair:
            timing['best_reload'] = duration(started)
            if ceiling != dict(real_blocks=real_blocks + recipe.blocks(recipe.TEST_ROWS),
                               null_words=0, optimizer_updates=updates,
                               forward_rows=forward_rows + 2 * recipe.TEST_ROWS):
                raise ValueError('COUNT_MISMATCH')
        phase('TEST_DATA')
        assert_source(store)
        meter.check()
        started = elapsed()
        panels.test()
        if pair:
            timing['test_data'] = duration(started)
        panel = panels.view('test', recipe.PANEL)
        hashes = _hashes(panels, {'train', 'dev', 'test'})
        if (len(panel) != recipe.TEST_ROWS
                or any(hashes[k] != digest for k, digest in prior_hashes.items())):
            raise ValueError('PANEL_IDENTITY')
        totals = recipe.counts(*epochs)
        _counts(meter, real_blocks=totals['real_blocks'], optimizer_updates=updates,
                forward_rows=forward_rows)
        snapshot()
        phase('TEST_SCORE')
        started = elapsed()
        scores = _score(panel, restored, meter, store, recipe)
        if pair:
            timing['test_score'] = duration(started)
    finally:
        for engine in restored.values():
            engine.release()

    phase('FINAL_ACCOUNTING')
    _counts(meter, **totals)
    snapshot()  # Also protect paid TEST work if a late check/commit fails.
    summary['cuda_peaks'] = peaks()
    meter.check()
    paired = metrics.compare(scores[recipe.AUXILIARY], scores[recipe.BASELINE])
    test = {'unit': 'pair' if pair else 'group',
            'by_slot': _pair_scores(scores) if pair else {k: v.saved for k, v in scores.items()},
            'forward_rows_by_slot': dict.fromkeys(recipe.SLOTS, len(panel)), 'b_vs_a': paired}
    phase('FINALIZE')
    assert_source(store)
    meter.check()
    store.finalize({'status': 'SUCCEEDED', 'counts': meter.snapshot(), 'summary': {
        **summary, 'phase': 'COMPLETE', 'panel_hashes': hashes, 'test': test,
        'shared_c7_outputs': totals['real_blocks'] - recipe.KAT_BLOCKS,
    }})

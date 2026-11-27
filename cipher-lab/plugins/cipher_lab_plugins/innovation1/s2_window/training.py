"""Single chronological fit; strictly reload the selected disk bytes before TEST."""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Protocol, cast

import numpy as np

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Kind, Meter, Stopped
from cipher_lab.training.early_stopping import Settings
from cipher_lab_plugins.direct_block.fit import (
    BatchPanel,
    FitSettings,
    assert_source,
    batch_indices,
    fit,
    predict_batch,
    reload_best,
    seeded_order,
)

from ..prefix_bridge.config import Counts
from ..prefix_bridge.data import Split
from ..prefix_bridge.metrics import score
from ..prefix_bridge.training import ArmRuntime
from . import config


class _Panels(Protocol):
    @property
    def hashes(self) -> dict[str, str]: ...
    def view(self, split: Split) -> BatchPanel: ...
    def test(self) -> None: ...


class _CardRuntime(ArmRuntime, Protocol):
    @property
    def geometry(self) -> dict[str, str]: ...


class _CardRecipe(Protocol):
    SLOT: str
    ROUNDS: tuple[int, ...]
    TRAIN_ROWS: int
    DEV_ROWS: int
    TEST_ROWS: int
    KAT_BLOCKS: int
    KAT_NAME: str
    MIN_EPOCHS: int
    EPOCHS: int
    PARAMETERS: int
    WARMUP: int
    PATIENCE: int
    MIN_DELTA: float
    BATCH_SIZE: int
    CHECKPOINT_BYTES: int
    ORDER_SEED: int

    def counts(self, epochs: int, *, test: bool = True) -> Counts: ...


def _check_counts(meter: Meter, *, real_blocks: int, null_words: int = 0,
                  optimizer_updates: int = 0, forward_rows: int = 0) -> None:
    current = meter.snapshot()
    expected = dict(real_blocks=real_blocks, null_words=null_words,
                    optimizer_updates=optimizer_updates, forward_rows=forward_rows)
    if any(current[f'{status}_{name}'] != n for name, n in expected.items()
           for status in ('attempted', 'completed')):
        raise ValueError('COUNT_MISMATCH')


def _check_capacity(meter: Meter, *, real_blocks: int, optimizer_updates: int = 0,
                    forward_rows: int = 0) -> None:
    # Meter owns a frozen Policy and charges only immediately before actual work.
    # Test admission is non-charging: do not start a split that cannot finish.
    required: dict[Kind, int] = {'real_blocks': real_blocks,
                                'optimizer_updates': optimizer_updates,
                                'forward_rows': forward_rows}
    for kind, n in required.items():
        if n > getattr(meter._policy, kind):
            raise Stopped(f'{kind.upper()}_LIMIT')


def _hashes(hashes: Mapping[str, str], expected: set[str]) -> dict[str, str]:
    if hashes.keys() != expected or any(re.fullmatch('[a-f0-9]{64}', h) is None
                                        for h in hashes.values()):
        raise ValueError('PANEL_IDENTITY')
    return dict(hashes)


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        gate: Callable[[Meter], None], make: Callable[[str, dict[str, object], bool], ArmRuntime],
        make_panels: Callable[[Meter], _Panels], peaks: Callable[[], dict[str, int]],
        _card_recipe: _CardRecipe | None = None) -> None:
    recipe = config if _card_recipe is None else _card_recipe
    if _card_recipe is not None:
        store._failure_counts_supplier = meter.snapshot

    def phase(stage: str) -> None:
        summary['phase'] = stage
        if stage == 'BEST_SAVE':
            # The write may seal the owner. Journal paid work before any
            # serialization/storage, without publishing any proposed scores.
            meter.check()
            store.write_snapshot({'counts': meter.snapshot()})
            assert_source(store)
            meter.check()

    def snapshot() -> None:
        meter.check()
        store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})
        meter.check()

    phase('KAT')
    assert_source(store)
    meter.check()
    _check_counts(meter, real_blocks=0)
    _check_capacity(meter, real_blocks=recipe.KAT_BLOCKS)
    gate(meter)
    _check_counts(meter, real_blocks=recipe.KAT_BLOCKS)
    summary['kat'] = {'status': 'PASSED', 'name': recipe.KAT_NAME, 'real_blocks': recipe.KAT_BLOCKS}
    snapshot()

    phase('TRAIN_DEV_DATA')
    assert_source(store)
    earliest = recipe.counts(recipe.MIN_EPOCHS, test=False)
    _check_capacity(meter, real_blocks=earliest['real_blocks'],
                    optimizer_updates=earliest['optimizer_updates'],
                    forward_rows=earliest['forward_rows'])
    panels = make_panels(meter)
    train, dev = panels.view('train'), panels.view('dev')
    if len(train) != recipe.TRAIN_ROWS or len(dev) != recipe.DEV_ROWS:
        raise ValueError('INVALID_PANEL_SIZE')
    prior_hashes = _hashes(panels.hashes, {'train', 'dev'})
    initial_blocks = (recipe.KAT_BLOCKS + 3 * config.trajectories(len(train) + len(dev))
                      if _card_recipe is None else earliest['real_blocks'])
    _check_counts(meter, real_blocks=initial_blocks)
    summary['panel_hashes'] = prior_hashes
    snapshot()

    slot = recipe.SLOT
    arm: dict[str, object] = {'round': recipe.ROUNDS[-1]}
    if _card_recipe is None:
        arm['mask'] = config.MASK
    else:
        arm['unit'] = 'pair'
    summary['fit'] = arm
    stopping = Settings(recipe.WARMUP, recipe.PATIENCE, recipe.MIN_DELTA)
    settings = FitSettings(recipe.EPOCHS, recipe.BATCH_SIZE, recipe.CHECKPOINT_BYTES,
                           early_stopping=stopping)
    order = seeded_order(recipe.TRAIN_ROWS, recipe.ORDER_SEED)
    phase('FIT_START')
    assert_source(store)
    meter.check()
    runtime = make(slot, arm, True)
    try:
        if (runtime.parameters != recipe.PARAMETERS
                or re.fullmatch('[a-f0-9]{64}', runtime.initial_sha) is None):
            raise ValueError('MODEL_INITIAL_IDENTITY')
        arm.update(parameters=runtime.parameters, initial_sha256=runtime.initial_sha)
        if _card_recipe is not None:
            arm['geometry'] = cast(_CardRuntime, runtime).geometry
        best = fit(train, dev, runtime, order, slot, arm, summary, settings, meter, store,
                   phase=phase, batch_update=runtime.batch_update)
        epochs = arm.get('epochs_completed')
        if (best is None or type(epochs) is not int
                or not recipe.MIN_EPOCHS <= epochs <= recipe.EPOCHS):
            raise ValueError('FIT_INCOMPLETE')
        if (_card_recipe is not None and arm.get('stop_reason') != (
                'EPOCH_CEILING' if epochs == recipe.EPOCHS else 'EARLY_STOPPING')):
            raise ValueError('FIT_STOP_REASON')
        if best[0] != f'model/{slot}.pt':
            raise ValueError('BEST_SLOT_MISMATCH')
        prior = recipe.counts(epochs, test=False)
        _check_counts(meter, **prior)
    finally:
        runtime.release()
    summary['train_presentations'] = epochs * recipe.TRAIN_ROWS

    phase('BEST_RELOAD')
    assert_source(store)
    meter.check()
    restored = make(slot, arm, False)
    try:
        reload_best(restored, best, arm, summary, settings.checkpoint_bytes, meter, store,
                    phase=phase)
        if (restored.parameters != arm['parameters']
                or restored.initial_sha != arm['initial_sha256']
                or arm['best_reloaded'] is not True
                or (_card_recipe is not None
                    and cast(_CardRuntime, restored).geometry != arm['geometry'])):
            raise ValueError('MODEL_INITIAL_IDENTITY')
        _check_counts(meter, **prior)

        phase('TEST_DATA')
        assert_source(store)
        meter.check()
        totals = recipe.counts(epochs)
        _check_capacity(meter, real_blocks=totals['real_blocks'],
                        optimizer_updates=totals['optimizer_updates'],
                        forward_rows=totals['forward_rows'])
        panels.test()
        panel = panels.view('test')
        if len(panel) != recipe.TEST_ROWS:
            raise ValueError('INVALID_PANEL_SIZE')
        hashes = _hashes(panels.hashes, {'train', 'dev', 'test'})
        if any(hashes[name] != digest for name, digest in prior_hashes.items()):
            raise ValueError('PANEL_IDENTITY')
        _check_counts(meter, real_blocks=totals['real_blocks'],
                      forward_rows=prior['forward_rows'],
                      optimizer_updates=prior['optimizer_updates'])
        snapshot()  # Not the TEST digest or any TEST scores.

        phase('TEST_SCORE')
        assert_source(store)
        meter.check()
        restored.activate()
        logits = np.empty((len(panel), 1), dtype=np.float32)
        labels = np.empty((len(panel), 1), dtype=np.uint8)
        for at in batch_indices(len(panel), settings.batch_size):
            x, y = panel.batch(at)
            logits[at] = predict_batch(restored, x, y, meter)
            labels[at] = y
        meter.check()
        if int(labels.sum()) != recipe.TEST_ROWS // 2:
            raise ValueError('UNBALANCED_TEST')
        result = score(logits, labels).saved
    finally:
        restored.release()

    phase('FINAL_ACCOUNTING')
    meter.check()
    summary['cuda_peaks'] = peaks()
    meter.check()
    _check_counts(meter, **totals)
    # Keep paid counts durable if the terminal check or write fails. The TEST
    # aggregates and digest have no publication path except the success record.
    snapshot()
    phase('FINALIZE')
    assert_source(store)
    meter.check()
    terminal = {**summary, 'phase': 'COMPLETE', 'panel_hashes': hashes}
    if _card_recipe is None:
        n = config.trajectories(recipe.TRAIN_ROWS + recipe.DEV_ROWS + recipe.TEST_ROWS)
        terminal['science'] = {'inverse_operations': n, 'full7_forward_trajectories': n,
                               'charged_output_units': 3*n}
        terminal['test'] = {**result, 'unit': 'group', 'reference_accuracy': .5}
    else:
        # The scorer already computed all correctness/BCE; change only the unit.
        labels_by_class = cast('dict[str, dict[str, int]]', result.pop('by_label'))
        result['pairs'] = result.pop('groups')
        result['correct_pairs'] = result.pop('correct_groups')
        terminal['test'] = {**result, 'unit': 'pair', 'reference_accuracy': .5,
                            'by_label': {label: {'pairs': n['groups'], 'correct': n['correct']}
                                         for label, n in labels_by_class.items()}}
    store.finalize({'status': 'SUCCEEDED', 'counts': meter.snapshot(), 'summary': terminal})

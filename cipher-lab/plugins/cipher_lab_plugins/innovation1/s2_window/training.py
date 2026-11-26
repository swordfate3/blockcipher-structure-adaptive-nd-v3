"""Single chronological fit; strictly reload the selected disk bytes before TEST."""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping

import numpy as np

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Kind, Meter, Stopped
from cipher_lab.training.early_stopping import Settings
from cipher_lab_plugins.direct_block.fit import (
    FitSettings,
    assert_source,
    batch_indices,
    fit,
    predict_batch,
    reload_best,
    seeded_order,
)

from ..prefix_bridge.metrics import score
from ..prefix_bridge.training import ArmRuntime
from . import config
from .data import Panels


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
        make_panels: Callable[[Meter], Panels], peaks: Callable[[], dict[str, int]]) -> None:
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
    _check_capacity(meter, real_blocks=config.KAT_BLOCKS)
    gate(meter)
    _check_counts(meter, real_blocks=config.KAT_BLOCKS)
    summary['kat'] = {'status': 'PASSED', 'name': config.KAT_NAME, 'real_blocks': config.KAT_BLOCKS}
    snapshot()

    phase('TRAIN_DEV_DATA')
    assert_source(store)
    earliest = config.counts(config.MIN_EPOCHS, test=False)
    _check_capacity(meter, real_blocks=earliest['real_blocks'],
                    optimizer_updates=earliest['optimizer_updates'],
                    forward_rows=earliest['forward_rows'])
    panels = make_panels(meter)
    train, dev = panels.view('train'), panels.view('dev')
    if len(train) != config.TRAIN_ROWS or len(dev) != config.DEV_ROWS:
        raise ValueError('INVALID_PANEL_SIZE')
    prior_hashes = _hashes(panels.hashes, {'train', 'dev'})
    initial_blocks = config.KAT_BLOCKS + 3 * config.trajectories(len(train) + len(dev))
    _check_counts(meter, real_blocks=initial_blocks)
    summary['panel_hashes'] = prior_hashes
    snapshot()

    slot = config.SLOT
    arm: dict[str, object] = {'round': config.ROUNDS[-1], 'mask': config.MASK}
    summary['fit'] = arm
    stopping = Settings(config.WARMUP, config.PATIENCE, config.MIN_DELTA)
    settings = FitSettings(config.EPOCHS, config.BATCH_SIZE, config.CHECKPOINT_BYTES,
                           early_stopping=stopping)
    order = seeded_order(config.TRAIN_ROWS, config.ORDER_SEED)
    phase('FIT_START')
    assert_source(store)
    meter.check()
    runtime = make(slot, arm, True)
    try:
        if (runtime.parameters != config.PARAMETERS
                or re.fullmatch('[a-f0-9]{64}', runtime.initial_sha) is None):
            raise ValueError('MODEL_INITIAL_IDENTITY')
        arm.update(parameters=runtime.parameters, initial_sha256=runtime.initial_sha)
        best = fit(train, dev, runtime, order, slot, arm, summary, settings, meter, store,
                   phase=phase, batch_update=runtime.batch_update)
        epochs = arm.get('epochs_completed')
        if (best is None or type(epochs) is not int
                or not config.MIN_EPOCHS <= epochs <= config.EPOCHS):
            raise ValueError('FIT_INCOMPLETE')
        if best[0] != f'model/{slot}.pt':
            raise ValueError('BEST_SLOT_MISMATCH')
        prior = config.counts(epochs, test=False)
        _check_counts(meter, **prior)
    finally:
        runtime.release()
    summary['train_presentations'] = epochs * config.TRAIN_ROWS

    phase('BEST_RELOAD')
    assert_source(store)
    meter.check()
    restored = make(slot, arm, False)
    try:
        reload_best(restored, best, arm, summary, settings.checkpoint_bytes, meter, store,
                    phase=phase)
        if (restored.parameters != arm['parameters']
                or restored.initial_sha != arm['initial_sha256']
                or arm['best_reloaded'] is not True):
            raise ValueError('MODEL_INITIAL_IDENTITY')
        _check_counts(meter, **prior)

        phase('TEST_DATA')
        assert_source(store)
        meter.check()
        totals = config.counts(epochs)
        _check_capacity(meter, real_blocks=totals['real_blocks'],
                        optimizer_updates=totals['optimizer_updates'],
                        forward_rows=totals['forward_rows'])
        panels.test()
        panel = panels.view('test')
        if len(panel) != config.TEST_ROWS:
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
        if int(labels.sum()) != config.TEST_ROWS // 2:
            raise ValueError('UNBALANCED_TEST')
        result = score(logits, labels).saved
    finally:
        restored.release()

    phase('FINAL_ACCOUNTING')
    meter.check()
    summary['cuda_peaks'] = peaks()
    meter.check()
    _check_counts(meter, **totals)
    n = config.trajectories(config.TRAIN_ROWS + config.DEV_ROWS + config.TEST_ROWS)
    # Keep paid counts durable if the terminal check or write fails. The TEST
    # aggregates and digest have no publication path except the success record.
    snapshot()
    phase('FINALIZE')
    assert_source(store)
    meter.check()
    store.finalize({'status': 'SUCCEEDED', 'counts': meter.snapshot(), 'summary': {
        **summary, 'phase': 'COMPLETE', 'panel_hashes': hashes,
        'science': {'inverse_operations': n, 'full7_forward_trajectories': n,
                    'charged_output_units': 3*n},
        'test': {**result, 'unit': 'group', 'reference_accuracy': .5},
    }})

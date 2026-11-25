"""Single serial owner: public KAT, thirteen fits, thirteen strict restores, TEST."""
from __future__ import annotations

import io
from collections.abc import Callable
from typing import Protocol

import numpy as np

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter
from cipher_lab.training.early_stopping import Settings
from cipher_lab_plugins.direct_block.fit import (
    BatchPanel,
    Best,
    Engine,
    FitSettings,
    assert_source,
    batch_indices,
    fit,
    predict_batch,
    reload_best,
    seeded_order,
)

from . import config, metrics
from .data import Floats, Labels, Panels


class ArmRuntime(Engine, Protocol):
    @property
    def initial_sha(self) -> str: ...
    @property
    def parameters(self) -> int: ...
    def activate(self) -> None: ...
    def release(self) -> None: ...
    def batch_update(self, inputs: Floats, labels: Labels, meter: Meter) -> float: ...


def _encode_test(labels: Labels, logits: dict[str, Floats]) -> bytes:
    stream = io.BytesIO()
    np.savez(stream, labels=labels, **logits)  # type: ignore[arg-type]
    return stream.getvalue()


def _score(panel: BatchPanel, engine: Engine, meter: Meter) -> tuple[Floats, Labels]:
    logits = np.empty((len(panel), 1), dtype=np.float32)
    labels = np.empty((len(panel), 1), dtype=np.uint8)
    for at in batch_indices(len(panel), config.BATCH_SIZE):
        x, y = panel.batch(at)
        logits[at] = predict_batch(engine, x, y, meter)
        labels[at] = y
    meter.check()
    return logits, labels


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        gate: Callable[[Meter], None], make: Callable[[str, dict[str, object], bool], ArmRuntime],
        make_panels: Callable[[Meter], Panels], peaks: Callable[[], dict[str, int]],
        encode_test: Callable[[Labels, dict[str, Floats]], bytes] = _encode_test) -> None:
    def phase(stage: str) -> None:
        summary['phase'] = stage
        if stage == 'BEST_SAVE':
            # fit already completed DEV; make those paid counts durable BEFORE
            # encoding or writing the first (or any replacement) best.
            snapshot()
            assert_source(store)

    def snapshot() -> None:
        meter.check()
        store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})
        meter.check()

    phase('KAT')
    assert_source(store)
    meter.check()
    gate(meter)
    counts = meter.snapshot()
    if any(counts[f'{prefix}_real_blocks'] != config.KAT_BLOCKS
           for prefix in ('attempted', 'completed')):
        raise ValueError('KAT_COUNT_MISMATCH')
    summary['kat'] = {'status': 'PASSED', 'real_blocks': config.KAT_BLOCKS}
    snapshot()
    phase('TRAIN_DEV_DATA')
    assert_source(store)
    panels = make_panels(meter)
    summary['panel_hashes'] = panels.hashes
    snapshot()

    stopping = Settings(config.WARMUP, config.PATIENCE, config.MIN_DELTA)
    settings = FitSettings(config.EPOCHS, config.BATCH_SIZE, config.CHECKPOINT_BYTES,
                           early_stopping=stopping)
    order = seeded_order(config.TRAIN_ROWS, config.ORDER_SEED)
    arms: dict[str, dict[str, object]] = {}
    bests: dict[str, Best] = {}
    epochs: list[int] = []
    initial: tuple[int, str] | None = None
    summary['fits'] = arms
    for slot in config.SLOTS:
        phase('FIT_START')
        meter.check()
        assert_source(store)
        r, mask = config.slot_input(slot)
        arm: dict[str, object] = {'round': r, 'mask': mask}
        arms[slot] = arm
        runtime = make(slot, arm, True)
        try:
            actual = runtime.parameters, runtime.initial_sha
            if initial is not None and actual != initial:
                raise ValueError('MODEL_INITIAL_IDENTITY')
            initial = actual
            arm.update(parameters=actual[0], initial_sha256=actual[1])
            train, dev = panels.train(slot), panels.dev(slot)
            if len(train) != config.TRAIN_ROWS or len(dev) != config.DEV_ROWS:
                raise ValueError('INVALID_PANEL_SIZE')
            best = fit(train, dev, runtime, order, slot, arm, summary, settings, meter, store,
                       phase=phase, batch_update=runtime.batch_update)
            n = arm.get('epochs_completed')
            if best is None or type(n) is not int or not config.MIN_EPOCHS <= n <= config.EPOCHS:
                raise ValueError('FIT_INCOMPLETE')
            if best[0] != f'model/{slot}.pt':
                raise ValueError('BEST_SLOT_MISMATCH')
            bests[slot] = best
            epochs.append(n)
        finally:
            runtime.release()
    summary['epochs_total'] = sum(epochs)
    summary['train_presentations'] = sum(epochs) * config.TRAIN_ROWS

    # Retain the already strictly restored CPU modules; no later reread of
    # unchecked bytes and no optimizer for any of these reconstructed models.
    restored: dict[str, ArmRuntime] = {}
    logits: dict[str, Floats] = {}
    scores: dict[str, metrics.Score] = {}
    labels: Labels | None = None
    try:
        for slot in config.SLOTS:
            phase('BEST_RELOAD')
            assert_source(store)
            meter.check()
            cpu = make(slot, arms[slot], False)
            restored[slot] = cpu
            reload_best(cpu, bests[slot], arms[slot], summary, settings.checkpoint_bytes,
                        meter, store, phase=phase)
        phase('TEST_DATA')
        assert_source(store)
        meter.check()
        panels.test()
        summary['panel_hashes'] = panels.hashes
        summary['trajectories'] = config.TRAJECTORIES
        summary['roundstateoutputs'] = config.ROUNDSTATEOUTPUTS
        snapshot()
        phase('TEST_SCORE')
        for slot in config.SLOTS:
            assert_source(store)
            meter.check()
            engine = restored[slot]
            engine.activate()
            panel = panels.view('test', slot)
            if len(panel) != config.TEST_ROWS:
                raise ValueError('INVALID_PANEL_SIZE')
            z, y = _score(panel, engine, meter)
            if labels is not None and not np.array_equal(labels, y):
                raise ValueError('UNALIGNED_TEST_GROUPS')
            labels = y
            logits[slot], scores[slot] = z, metrics.score(z, y)
            del restored[slot]
            engine.release()
    finally:
        for engine in restored.values():
            engine.release()
    if labels is None:
        raise ValueError('MISSING_TEST')

    phase('FINAL_ACCOUNTING')
    meter.check()
    summary['cuda_peaks'] = peaks()
    meter.check()
    counts = meter.snapshot()
    if any(counts[f'{prefix}_{kind}'] != amount
           for kind, amount in config.counts(epochs).items()
           for prefix in ('attempted', 'completed')):
        raise ValueError('COUNT_MISMATCH')
    rounds: list[dict[str, object]] = []
    s2 = scores['s2-only']
    for r in config.ROUNDS:
        cipher, joint = scores[f'r{r}-cipher'], scores[f'r{r}-joint']
        rounds.append({'round': r, 'cipher_slot': f'r{r}-cipher', 'joint_slot': f'r{r}-joint',
                       's2_slot': 's2-only', 'joint_vs_cipher': metrics.compare(joint, cipher),
                       'joint_vs_s2': metrics.compare(joint, s2)})
    phase('TEST_SAVE')
    assert_source(store)
    snapshot()
    raw = encode_test(labels, logits)
    meter.check()
    ref = store.write_dataset('test-groups', raw)
    del raw
    phase('FINALIZE')
    assert_source(store)
    meter.check()
    test = {'ref': ref, 'unit': 'group', 'by_slot': {k: v.saved for k, v in scores.items()},
            'rounds': rounds}
    # Scores never enter any live snapshot or failure record.
    store.finalize({'status': 'SUCCEEDED', 'counts': meter.snapshot(),
                    'summary': {**summary, 'phase': 'COMPLETE', 'test': test}})

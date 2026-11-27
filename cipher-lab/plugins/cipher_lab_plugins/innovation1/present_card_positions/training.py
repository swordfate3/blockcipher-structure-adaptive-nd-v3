"""Train the three position arms serially and evaluate on one matched panel."""
from __future__ import annotations

import re
from collections.abc import Callable
from contextlib import ExitStack
from typing import Protocol, cast

import numpy as np

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter
from cipher_lab.training.early_stopping import Settings
from cipher_lab_plugins.direct_block.fit import (
    Best,
    Engine,
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

from ..mask_aux.training import _counts, _pair_scores, _require_capacity
from ..mask_aux_single.data import Panels
from ..prefix_bridge.config import Counts
from ..prefix_bridge.metrics import Score, compare, score
from ..present_card_fresh import config as pure
from ..present_card_mask import config as prior
from . import config


class _FitRuntime(Engine, Protocol):
    @property
    def initial_sha(self) -> str: ...
    @property
    def geometry(self) -> dict[str, str]: ...
    @property
    def parameters(self) -> int: ...
    def batch_update(self, inputs: Floats, labels: Labels, meter: Meter) -> float: ...
    def activate(self) -> None: ...
    def release(self) -> None: ...


class _FusionRuntime(_FitRuntime, Protocol):
    @property
    def backbone_initial_sha(self) -> str: ...
    @property
    def backbone_geometry(self) -> dict[str, str]: ...
    @property
    def fusion_initial_sha(self) -> str: ...


class _ThreeArmRecipe(Protocol):
    SLOT: str
    @property
    def SLOTS(self) -> tuple[str, ...]: ...
    TRAIN_ROWS: int
    DEV_ROWS: int
    TEST_ROWS: int
    CAPS: Counts

    def counts(self, epochs: tuple[int, ...], *, test: bool = True) -> Counts: ...


class _FusionRecipe(_ThreeArmRecipe, Protocol):
    METHOD: str
    BASE: str
    PARAMETERS: dict[str, int]
    @property
    def CONTRASTS(self) -> tuple[tuple[str, str], ...]: ...


def _hashes(hashes: dict[str, str], expected: set[str]) -> dict[str, str]:
    if hashes.keys() != expected or any(re.fullmatch('[a-f0-9]{64}', h) is None
                                        for h in hashes.values()):
        raise ValueError('PANEL_IDENTITY')
    return dict(hashes)


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        gate: Callable[[Meter], None], panels_for: Callable[[Meter], Panels],
        make: Callable[[str, dict[str, object], bool, int], _FitRuntime],
        peaks: Callable[[], dict[str, int]], _recipe: _FusionRecipe | None = None) -> None:
    recipe: _ThreeArmRecipe = config if _recipe is None else _recipe
    label_slot = config.RANDOM if _recipe is None else _recipe.BASE
    if _recipe is not None:
        summary['method'] = _recipe.METHOD
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
    _require_capacity(meter, recipe.CAPS)
    _counts(meter, real_blocks=0)
    gate(meter)
    _counts(meter, real_blocks=pure.KAT_BLOCKS)
    summary['kat'] = dict(status='PASSED', name=pure.KAT_NAME, real_blocks=pure.KAT_BLOCKS)
    snapshot()

    phase('TRAIN_DEV_DATA')
    panels = panels_for(meter)
    train = panels.view('train', recipe.SLOT)
    dev = panels.view('dev', recipe.SLOT)
    if (len(train), len(dev)) != (recipe.TRAIN_ROWS, recipe.DEV_ROWS):
        raise ValueError('INVALID_PANEL_SIZE')
    hashes = _hashes(panels.hashes, {'train', 'dev'})
    summary['panel_hashes'] = hashes
    snapshot()
    _counts(meter, **recipe.counts((), test=False))
    settings = FitSettings(pure.EPOCHS, pure.BATCH_SIZE, pure.CHECKPOINT_BYTES,
                           early_stopping=Settings(pure.WARMUP, pure.PATIENCE, pure.MIN_DELTA))
    order = seeded_order(len(train), pure.ORDER_SEED)
    initial: tuple[str, dict[str, str]] | None = None
    fusion_initial: str | None = None
    for slot in recipe.SLOTS:
        summary['active_slot'] = slot
        phase('FIT_START')
        assert_source(store)
        meter.check()
        arm: dict[str, object] = dict(unit='pair', round=pure.ROUNDS[0])
        arms[slot] = arm
        origin = sum(pure._steps(n) for n in epochs)
        runtime = make(slot, arm, True, origin)
        try:
            identity = (runtime.initial_sha, runtime.geometry)
            shared = identity
            parameters = prior.PARAMETERS
            if _recipe is not None:
                fusion = cast(_FusionRuntime, runtime)
                shared = (fusion.backbone_initial_sha, fusion.backbone_geometry)
                parameters = _recipe.PARAMETERS[slot]
                projection = fusion.fusion_initial_sha
                if slot == label_slot:
                    if projection != '':
                        raise ValueError('MODEL_INITIAL_IDENTITY')
                else:
                    if (not projection
                            or (fusion_initial is not None and projection != fusion_initial)):
                        raise ValueError('MODEL_INITIAL_IDENTITY')
                    fusion_initial = projection
                arm.update(backbone_initial_sha256=shared[0], backbone_geometry=shared[1],
                           fusion_initial_sha256=projection)
            if runtime.parameters != parameters or (initial is not None and shared != initial):
                raise ValueError('MODEL_INITIAL_IDENTITY')
            initial = shared
            arm.update(initial_sha256=identity[0], geometry=identity[1], parameters=parameters)
            started = (cast(float, meter.snapshot()['wall_elapsed_seconds'])
                       if _recipe is not None else 0.)
            chosen = fit(train, dev, runtime, order, slot, arm, summary, settings, meter, store,
                         phase=phase, batch_update=runtime.batch_update)
            if _recipe is not None:
                meter.check()
                finished = cast(float, meter.snapshot()['wall_elapsed_seconds'])
                arm['fit_elapsed_seconds'] = finished - started
            n = arm.get('epochs_completed')
            if chosen is None or chosen[0] != f'model/{slot}.pt' or type(n) is not int:
                raise ValueError('FIT_INCOMPLETE')
            if not pure.MIN_EPOCHS <= n <= pure.EPOCHS or arm['stop_reason'] != (
                    'EPOCH_CEILING' if n == pure.EPOCHS else 'EARLY_STOPPING'):
                raise ValueError('FIT_STOP_REASON')
            bests[slot] = chosen
            epochs.append(n)
            _counts(meter, **recipe.counts(tuple(epochs), test=False))
        finally:
            runtime.release()
    before = recipe.counts(tuple(epochs), test=False)
    totals = recipe.counts(tuple(epochs))
    results: dict[str, Score] = {}
    with ExitStack() as cleanup:
        restored: dict[str, _FitRuntime] = {}
        for slot in recipe.SLOTS:
            summary['active_slot'] = slot
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
        phase('TEST_DATA')
        assert_source(store)
        meter.check()
        panels.test()
        test = panels.view('test', recipe.SLOT)
        if len(test) != recipe.TEST_ROWS:
            raise ValueError('INVALID_PANEL_SIZE')
        final_hashes = _hashes(panels.hashes, {'train', 'dev', 'test'})
        if any(final_hashes[k] != value for k, value in hashes.items()):
            raise ValueError('PANEL_IDENTITY')
        _counts(meter, real_blocks=totals['real_blocks'],
                optimizer_updates=before['optimizer_updates'], forward_rows=before['forward_rows'])
        snapshot()
        y = np.empty((len(test), 1), dtype=np.uint8)
        for slot, engine in restored.items():
            summary['active_slot'] = slot
            phase('TEST_SCORE')
            assert_source(store)
            meter.check()
            engine.activate()
            logits = np.empty((len(test), 1), dtype=np.float32)
            for at in batch_indices(len(test), pure.BATCH_SIZE):
                x, labels = test.batch(at)
                logits[at] = predict_batch(engine, x, labels, meter)
                if slot == label_slot:
                    y[at] = labels
                elif not np.array_equal(labels, y[at]):
                    raise ValueError('UNALIGNED_TEST_PAIRS')
            if int(y.sum()) != len(y) // 2:
                raise ValueError('UNBALANCED_TEST')
            results[slot] = score(logits, y)
            engine.release()
    summary.pop('active_slot')
    _counts(meter, **totals)
    summary['cuda_peaks'] = peaks()
    snapshot()
    assert_source(store)
    meter.check()
    contrasts = (((config.FOURTH, config.FIRST), (config.FIRST, config.RANDOM),
                  (config.FOURTH, config.RANDOM)) if _recipe is None else _recipe.CONTRASTS)
    terminal = {**summary, 'phase': 'COMPLETE', 'panel_hashes': final_hashes,
                'test': _pair_scores(results), 'paired': {
                    f'{a}_vs_{b}': compare(results[a], results[b]) for a, b in contrasts}}
    store.finalize(dict(status='SUCCEEDED', counts=meter.snapshot(), summary=terminal))

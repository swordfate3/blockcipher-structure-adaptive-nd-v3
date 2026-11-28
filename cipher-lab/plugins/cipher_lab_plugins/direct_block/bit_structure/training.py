"""Six fits and one all-or-nothing final TEST record."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Protocol, cast

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


class PanelSource(Protocol):
    def points(self, repeat: int, split: str, /) -> NDArray[np.uint32]: ...
    def label(self, repeat: int, split: str, /) -> NDArray[np.uint8]: ...


def _check_counts(meter: Meter, *, epochs: Sequence[int], testing: bool) -> None:
    expected = config.counts(epochs, test=testing)
    counts = meter.snapshot()
    if any(counts[f'{status}_{key}'] != value for key, value in expected.items()
           for status in ('attempted', 'completed')):
        raise ValueError('experiment accounting mismatch')


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        gate: Callable[[Meter], None], make_panels: Callable[[Meter], PanelSource],
        make: Callable[[str, dict[str, object], int], Engine],
        peaks: Callable[[], dict[str, int]] | None = None) -> None:
    def phase(name: str) -> None:
        summary['phase'] = name

    def load(repeat: int, split: str, rows: int) -> tuple[data.Panel, NDArray[np.uint8], str]:
        points = panels.points(repeat, split)
        if points.shape != (rows,):
            raise ValueError('unexpected panel size')
        assert_source(store)
        meter.check()
        labels = panels.label(repeat, split)
        return data.Panel(points, labels), labels, data.panel_digest(points, labels)

    phase('KAT')
    assert_source(store)
    gate(meter)
    gate_counts = meter.snapshot()
    if any(gate_counts[f'{status}_real_blocks'] != config.KAT_BLOCKS
           for status in ('attempted', 'completed')):
        raise ValueError('invalid public check block count')
    summary['public_kat'] = 'PASSED'
    store.write_snapshot({'counts': gate_counts, 'summary': summary})
    phase('PANELS')
    panels = make_panels(meter)
    stopping = Settings(config.WARMUP, config.PATIENCE, config.MIN_DELTA)
    settings = FitSettings(config.EPOCHS, config.BATCH, config.CHECKPOINT_BYTES,
                           early_stopping=stopping)
    arms: dict[str, dict[str, object]] = {}
    hashes: dict[str, dict[str, str]] = {}
    pending: dict[str, tuple[Engine, Best]] = {}
    constants: list[float] = []
    epochs: list[int] = []
    summary.update(method=config.METHOD, arms=arms, panel_hashes=hashes)
    for repeat, seed in enumerate(config.INIT_SEEDS):
        phase('TRAIN_DEV_LABELS')
        panel_id = f'r{repeat + 1}'
        train, labels, train_hash = load(repeat, 'train', config.TRAIN)
        dev, _, dev_hash = load(repeat, 'dev', config.DEV)
        hashes[panel_id] = {'train': train_hash, 'dev': dev_hash}
        constants.append(metrics.train_constant(labels))

        def order(epoch: int, index: int = repeat) -> NDArray[np.intp]:
            seed_sequence = np.random.SeedSequence([config.ORDER_SEED, index, epoch])
            return np.random.default_rng(seed_sequence).permutation(config.TRAIN)

        for name in config.ARMS:
            phase('MODEL')
            meter.check()
            slot = f'{panel_id}-{name}'
            summary['active_slot'] = slot
            arm: dict[str, object] = {'init_seed': seed}
            arms[slot] = arm
            engine = make(slot, arm, seed)
            initial = arm.get('initial_sha256')
            correct_initial = arms.get(f'{panel_id}-correct', {}).get('initial_sha256')
            if (arm.get('parameters') != config.PARAMS[name] or not isinstance(initial, str)
                    or len(initial) != 64 or any(c not in '0123456789abcdef' for c in initial)
                    or (name == 'shuffled' and initial != correct_initial)
                    or any(engine is other for other, _ in pending.values())):
                raise ValueError('invalid model identity')
            start = perf_counter()
            best = fit(train, dev, engine, order, slot, arm, summary, settings, meter, store,
                       phase=phase)
            arm['fit_elapsed_seconds'] = perf_counter() - start
            completed = arm.get('epochs_completed')
            if (type(completed) is not int or not config.MIN_EPOCHS <= completed <= config.EPOCHS
                    or best is None or best[0] != f'model/{slot}.pt'):
                raise ValueError('incomplete fit')
            pending[slot] = engine, best
            epochs.append(completed)
    if tuple(pending) != config.SLOTS:
        raise ValueError('unexpected checkpoint slots')
    _check_counts(meter, epochs=epochs, testing=False)
    for slot, (engine, best) in pending.items():
        summary['active_slot'] = slot
        reload_best(engine, best, arms[slot], summary, settings.checkpoint_bytes, meter, store,
                    phase=phase)
    summary.pop('active_slot')

    phase('TEST')
    tests: dict[str, dict[str, object]] = {}
    test_hashes: dict[str, str] = {}
    advances = []
    for repeat, probability in enumerate(constants):
        panel_id = f'r{repeat + 1}'
        test, labels, test_hashes[panel_id] = load(repeat, 'test', config.TEST)
        scores = {}
        vectors = {}
        for name in config.ARMS:
            engine, _ = pending[f'{panel_id}-{name}']
            logits = np.concatenate([predict_batch(engine, *test.batch(at), meter)
                                     for at in batch_indices(len(test), config.BATCH)])
            scores[name], vectors[name] = metrics.score(logits, labels)
        baseline = metrics.constant_score(probability, labels)
        correct_bce = cast(float, scores['correct']['bce'])
        comparisons = {
            f'correct_minus_{control}': {
                'bce': correct_bce - cast(float, scores[control]['bce']),
                'paired': metrics.paired(vectors['correct'], vectors[control]),
            } for control in ('shuffled', 'mlp')
        }
        advances.append(all(cast(float, result['bce']) < 0 for result in comparisons.values())
                        and correct_bce < cast(float, baseline['bce']))
        tests[panel_id] = {'arms': scores, 'constant': baseline, 'comparisons': comparisons}
    if peaks is not None:
        summary['gpu_peaks'] = peaks()
    assert_source(store)
    meter.check()
    _check_counts(meter, epochs=epochs, testing=True)
    for panel_id, digest in test_hashes.items():
        hashes[panel_id]['test'] = digest
    finished = {**summary, 'phase': 'COMPLETE', 'test': tests,
                'confirmation_candidate': all(advances), 'epochs_completed': sum(epochs),
                'dev_evaluations': sum(epochs), 'train_presentations': sum(epochs) * config.TRAIN}
    store.finalize({'status': 'SUCCEEDED', 'counts': meter.snapshot(), 'summary': finished})
    summary.update(finished)

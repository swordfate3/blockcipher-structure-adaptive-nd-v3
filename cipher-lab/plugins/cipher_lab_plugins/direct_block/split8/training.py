"""Fixed 3×4 composition of the common epoch fit, durable best and TEST paths."""
from __future__ import annotations

from collections.abc import Callable

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
    seeded_order,
)
from . import config, data, metrics


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        make_panels: Callable[[Meter], data.Panels], make_engine: Callable[[int, int], Engine],
        phase: Callable[[str], None]) -> dict[str, metrics.Result]:
    sizes = config.SIZES
    settings = FitSettings(config.EPOCHS, sizes.batch, config.CHECKPOINT_BYTES,
                           selected_columns=None, early_stopping=Settings(
                               config.WARMUP, config.PATIENCE, config.MIN_DELTA))
    phase('DATA')
    assert_source(store)
    meter.check()
    panels = make_panels(meter)
    train_points, dev_points = panels.points('train'), panels.points('dev')
    order = seeded_order(sizes.train, config.ORDER_SEED)
    fits: dict[str, dict[str, object]] = {}
    summary['fits'] = fits
    engines: dict[int, list[Engine]] = {}
    selected: list[tuple[Engine, Best, dict[str, object]]] = []
    labels: dict[config.Split, dict[int, data.Bytes]] = {s: {} for s in config.SPLITS}
    baseline: dict[int, tuple[int, ...]] = {}

    def fit_phase(stage: str) -> None:
        phase(stage)
        if stage == 'BEST_SAVE':
            # A failed model write can seal RunStore using its last committed
            # snapshot. Persist the just-completed TRAIN/DEV ledger first.
            # TEST scores are never part of this live summary.
            store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})

    for rounds in config.ROUNDS:
        phase('DATA')
        assert_source(store)
        train = labels['train'][rounds] = panels.label('train', rounds)
        dev = labels['dev'][rounds] = panels.label('dev', rounds)
        phase('BASELINE')
        baseline[rounds] = metrics.train_majority(train, sizes.batch, meter)
        engines[rounds] = []
        for chunk in config.CHUNKS:
            phase('MODEL')
            meter.check()
            engine = make_engine(rounds, chunk)
            if any(engine is old for old, _, _ in selected):
                raise ValueError('reused engine')
            engines[rounds].append(engine)
            meter.check()
            slot = f'r{rounds}-b{chunk}'
            arm: dict[str, object] = {'rounds': rounds, 'chunk': chunk,
                                      'columns_msb0': list(config.COLS[chunk]),
                                      'init_seed': config.INIT_SEED + chunk}
            fits[slot] = arm
            best = fit(data.Panel(train_points, train, chunk), data.Panel(dev_points, dev, chunk),
                       engine, order, slot, arm, summary, settings, meter, store, phase=fit_phase)
            if best is None or best[0] != f'model/{slot}.pt':
                raise ValueError('missing or wrong best slot')
            selected.append((engine, best, arm))

    expected = config.counts(completed_epochs(fits))
    if len(selected) != config.FITS:
        raise ValueError('incomplete fits')
    for engine, best, arm in selected:
        reload_best(engine, best, arm, summary, settings.checkpoint_bytes, meter, store,
                    phase=phase)

    phase('TEST')
    assert_source(store)
    meter.check()
    test_points = panels.points('test')
    scores: dict[str, metrics.Result] = {}
    for rounds in config.ROUNDS:
        packed = labels['test'][rounds] = panels.label('test', rounds)
        score = metrics.Score(sizes.test, baseline[rounds])
        for at in batch_indices(sizes.test, sizes.batch):
            meter.check()
            x, y = data.input_bits(test_points[at]), data.high_bits(packed[at])
            blocks = [predict_batch(engine, x, y[:, cols], meter)
                      for engine, cols in zip(engines[rounds], config.COLS, strict=True)]
            meter.check()
            score.add(y, blocks)
        scores[f'r{rounds}'] = score.result()

    phase('DATASETS')
    # Keep actual accounting durable without exposing scores in a RUNNING record.
    assert_source(store)
    meter.check()
    measured = meter.snapshot()
    for kind, amount in expected.items():
        if any(measured[f'{prefix}_{kind}'] != amount for prefix in ('attempted', 'completed')):
            raise ValueError('count mismatch')
    store.write_snapshot({'counts': measured, 'summary': summary})
    for split, points in zip(config.SPLITS, (train_points, dev_points, test_points), strict=True):
        meter.check()
        raw = data.dataset(points, labels[split])
        meter.check()
        store.write_dataset(split, raw)
        del raw
    meter.check()
    return scores


def completed_epochs(fits: dict[str, dict[str, object]]) -> tuple[int, ...]:
    if fits.keys() != set(config.SLOTS):
        raise ValueError('incomplete fit records')
    epochs: list[int] = []
    for slot in config.SLOTS:
        arm = fits[slot]
        epoch = arm.get('epochs_completed')
        if (type(epoch) is not int or not config.MIN_EPOCHS <= epoch <= config.EPOCHS
                or arm.get('stop_reason') != (
                    'EPOCH_CEILING' if epoch == config.EPOCHS else 'EARLY_STOPPING'
                )):
            raise ValueError('incomplete epoch evidence')
        epochs.append(epoch)
    return tuple(epochs)

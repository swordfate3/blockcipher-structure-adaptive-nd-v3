"""Serial rounds with one shared meter and a decision from each saved round."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import Any, cast

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter

from ..bit_structure import config as bit
from ..bit_structure.training import run_round
from ..fit import Engine, assert_source
from . import config
from .data import CyclePanels


def near_chance(tests: Mapping[str, object]) -> bool:
    if set(tests) != {f'r{i + 1}' for i in range(bit.REPEATS)}:
        raise ValueError('invalid TEST panels')
    decisions = []
    for panel in tests.values():
        scores = cast('dict[str, Any]', panel)
        score = scores['arms']['correct']
        accuracy, loss, baseline = score['accuracy'], score['bce'], scores['constant']['bce']
        if (not all(type(v) in (float, int) and math.isfinite(v)
                    for v in (accuracy, loss, baseline))
                or not 0 <= accuracy <= 1 or min(loss, baseline) < 0):
            raise ValueError('invalid TEST score')
        # Compare the decimal values preserved by the JSON round trip.
        gain = Decimal(str(baseline)) - Decimal(str(loss))
        decisions.append(config.ACC_MIN <= accuracy <= config.ACC_MAX
                         and gain <= Decimal(str(config.BCE_GAIN_MAX)))
    return all(decisions)


def _check_counts(meter: Meter, completed: int, epochs: int) -> None:
    expected = config.counts(completed, epochs)
    snapshot = meter.snapshot()
    if any(snapshot[f'{state}_{field}'] != amount for field, amount in expected.items()
           for state in ('attempted', 'completed')):
        raise ValueError('invalid cycle counters')


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        gate: Callable[[Meter], None], make_panels: Callable[[Meter], CyclePanels],
        make: Callable[[int, str, dict[str, object], int], Engine],
        read_snapshot: Callable[[], dict[str, object]],
        peaks: Callable[[], dict[str, int]] | None = None) -> None:
    completed: dict[str, dict[str, object]] = {}
    summary.update(phase='KAT', method=config.METHOD, rounds=completed)
    assert_source(store)
    gate(meter)
    _check_counts(meter, 0, 0)
    summary['public_kat'] = 'PASSED'
    store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})
    summary['phase'] = 'PANELS'
    panels = make_panels(meter)
    _check_counts(meter, 0, 0)
    total_epochs = 0
    stop_reason = 'ROUND_CEILING'
    for depth in config.ROUNDS:
        meter.check()
        summary.update(phase='ROUND_START', current_round=depth)
        store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})

        def engine(slot: str, arm: dict[str, object], seed: int, rounds: int = depth) -> Engine:
            return make(rounds, slot, arm, seed)

        result = run_round(meter, store, summary, panels=panels.for_round(depth), make=engine,
                           prefix=f'c{depth}-', parameters=config.parameters(depth))
        epoch_count = result['epochs_completed']
        if type(epoch_count) is not int or not len(bit.SLOTS) * bit.MIN_EPOCHS <= epoch_count <= (
                len(bit.SLOTS) * bit.EPOCHS):
            raise ValueError('invalid round epochs')
        total_epochs += epoch_count
        _check_counts(meter, len(completed) + 1, total_epochs)
        rid = str(depth)
        completed[rid] = result
        summary.pop('arms', None)
        summary.pop('panel_hashes', None)
        summary.update(phase='ROUND_SAVED', method=config.METHOD,
                       last_completed_round=depth, epochs_completed=total_epochs)
        store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})
        saved = cast('dict[str, Any]', read_snapshot())
        saved_result = saved['summary']['rounds'][rid]
        if saved['status'] != 'RUNNING' or saved_result != result:
            raise ValueError('invalid saved round')
        result['near_chance'] = near_chance(saved_result['test'])
        store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})
        if result['near_chance']:
            stop_reason = 'NEAR_CHANCE'
            break

    summary.update(phase='FINALIZE', stop_reason=stop_reason)
    if peaks is not None:
        summary['gpu_peaks'] = peaks()
    assert_source(store)
    meter.check()
    summary['phase'] = 'COMPLETE'
    store.finalize({'status': 'SUCCEEDED', 'counts': meter.snapshot(), 'summary': summary})

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.fit import FitSettings, Floats, Indices, Labels, fit

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter


class _Panel:
    def __init__(self, rows: int) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return self.rows

    def batch(self, indices: Indices) -> tuple[Floats, Labels]:
        return (np.zeros((len(indices), 1, 6, 64), np.float32),
                np.zeros((len(indices), 1), np.uint8))


class _Engine:
    predictions = 0

    def train_forward(self, inputs: Floats) -> Callable[[Labels], float]:
        raise AssertionError('only the existing batch-update seam is expected')

    def update(self) -> None:
        raise AssertionError('only the existing batch-update seam is expected')

    def predict(self, inputs: Floats) -> Floats:
        self.predictions += 1
        return np.zeros((len(inputs), 1), np.float32)

    def dump(self) -> bytes:
        return b'artificial'

    def restore(self, raw: bytes) -> None:
        raise AssertionError('fit must not restore')


@pytest.mark.parametrize('mode', ['tail', 'default', 'fault', 'bad-bce', 'interval'])
def test_tail_dev_and_paid_work(
    mode: str, tmp_path: Path, bridge_meter: Meter,
) -> None:
    store = RunStore.claim(tmp_path, 'fit-tail', output_bytes=2**24, needs_models=True,
                           model_slots=('pair',),
                           source_supplier=lambda: {'artificial/source.py': 'a' * 64})
    arm: dict[str, Any] = {}
    engine = _Engine()
    attempts: list[int] = []
    phases: list[str] = []

    def update(x: Floats, y: Labels, meter: Meter) -> float:
        attempts.append(len(x))
        assert len(y) == len(x)
        meter.reserve('forward_rows', len(x))
        meter.complete('forward_rows', len(x))
        if len(x) == 2 and mode == 'fault':
            raise ValueError('tail failed')
        return 1. if len(x) == 4 else (float('nan') if mode == 'bad-bce' else 4.)

    def execute(settings: FitSettings) -> None:
        fit(_Panel(6), _Panel(3), engine, lambda e: np.arange(6, dtype=np.intp),
            'pair', arm, {'arm': arm}, settings, bridge_meter, store, phase=phases.append,
            batch_update=update)

    if mode in ('default', 'interval'):
        with pytest.raises(ValueError):
            execute(FitSettings(2, 4, 4096) if mode == 'default' else
                    FitSettings(2, 4, 4096, dev_every_steps=1, allow_partial_batch=True))
        assert not attempts and not phases and not arm
        return
    settings = FitSettings(2, 4, 4096, allow_partial_batch=True)
    if mode == 'tail':
        execute(settings)
        assert attempts == [4, 2, 4, 2]
        assert phases == ['TRAIN', 'TRAIN', 'DEV', 'BEST_SAVE', 'TRAIN', 'TRAIN', 'DEV']
        assert arm['epochs_completed'] == 2 and 'stop_reason' not in arm
        assert [(v['step'], v['train_bce']) for v in arm['dev']] == [(2, 2.), (4, 2.)]
        assert arm['selected']['epoch'] == 1
        assert engine.predictions == 2
    else:
        with pytest.raises(ValueError, match='tail failed|invalid training BCE'):
            execute(settings)
        assert attempts == [4, 2] and phases == ['TRAIN', 'TRAIN']
        assert not arm['dev'] and 'epochs_completed' not in arm
    counts = bridge_meter.snapshot()
    assert counts['attempted_optimizer_updates'] == (4 if mode == 'tail' else 2)
    assert counts['completed_optimizer_updates'] == {'tail': 4, 'fault': 1, 'bad-bce': 2}[mode]
    assert counts['completed_forward_rows'] == (18 if mode == 'tail' else 6)


def test_tail_opt_in_is_boolean() -> None:
    with pytest.raises(ValueError):
        FitSettings(2, 4, 4096, allow_partial_batch=1)  # type: ignore[arg-type]

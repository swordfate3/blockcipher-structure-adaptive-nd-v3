from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.fit import FitSettings, Floats, Indices, Labels, fit

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Meter
from cipher_lab.training.early_stopping import Settings


class ManualPanel:
    def __init__(self, rows: int) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return self.rows

    def batch(self, indices: Indices) -> tuple[Floats, Labels]:
        return (np.zeros((len(indices), 4, 6, 64), dtype=np.float32),
                np.zeros((len(indices), 1), dtype=np.uint8))


class ManualEngine:
    def __init__(self, score: Callable[[int], float], *, segmented: bool = False) -> None:
        self.steps = self.predictions = 0
        self.calls: list[str] = []
        self.score = score
        self.segmented = segmented
        self.update_offset = 0

    def train_forward(self, inputs: Floats) -> Callable[[Labels], float]:
        assert self.segmented
        self.calls.append('forward')

        def backward(labels: Labels) -> float:
            assert labels.shape == (len(inputs), 1)
            self.calls.append('backward')
            return .25

        return backward

    def update(self) -> None:
        assert self.segmented
        self.calls.append('update')
        self.steps += 1

    def batch_update(self, inputs: Floats, labels: Labels, meter: Meter) -> float:
        assert not self.segmented
        assert inputs.shape[1:] == (4, 6, 64) and labels.shape == (len(inputs), 1)
        before = meter.snapshot()
        assert before['attempted_optimizer_updates'] == self.update_offset + self.steps + 1
        assert before['completed_optimizer_updates'] == self.update_offset + self.steps
        self.calls.append('batch_update')
        meter.reserve('forward_rows', len(inputs))
        meter.complete('forward_rows', len(inputs))
        self.steps += 1
        return .25

    def predict(self, inputs: Floats) -> Floats:
        self.predictions += 1
        return np.full((len(inputs), 1), self.score(self.steps // 2), dtype=np.float32)

    def dump(self) -> bytes:
        return f'artificial-step-{self.steps}'.encode()

    def restore(self, raw: bytes) -> None:
        pytest.fail('fit must not restore')


def _order(epoch: int) -> Indices:
    return np.array([0, 1, 2, 3] if epoch % 2 else [3, 1, 2, 0], dtype=np.intp)


def _store(tmp_path: Path, *slots: str) -> RunStore:
    return RunStore.claim(tmp_path, 'bridge', output_bytes=2**24, needs_models=True,
                          model_slots=slots, source_supplier=lambda: {'fake/source.py': '1' * 64})


@pytest.mark.parametrize('segmented', [True, False])
def test_exact_accounting_and_dev_earliest_tie(
    bridge_meter: Meter, tmp_path: Path, segmented: bool,
) -> None:
    engine = ManualEngine(lambda epoch: 0., segmented=segmented)
    arm: dict[str, object] = {}
    summary: dict[str, object] = {'arm': arm}
    settings = FitSettings(2, 2, 4096)
    phases: list[str] = []
    store = _store(tmp_path, 's2-only')
    args = (ManualPanel(4), ManualPanel(1), engine, _order, 's2-only', arm, summary,
            settings, bridge_meter, store)
    if segmented:
        best = fit(*args, phase=phases.append)
    else:
        best = fit(*args, phase=phases.append, batch_update=engine.batch_update)
    assert engine.steps == 4 and engine.predictions == 2
    assert engine.calls == (['forward', 'backward', 'update'] * 4 if segmented
                            else ['batch_update'] * 4)
    assert phases == ['TRAIN', 'TRAIN', 'DEV', 'BEST_SAVE', 'TRAIN', 'TRAIN', 'DEV']
    curve = arm['dev']
    assert isinstance(curve, list) and [p['train_bce'] for p in curve] == [.25, .25]
    assert arm['selected'] == {**curve[0], 'ref': 'model/s2-only.pt',
                               'sha256': sha256(b'artificial-step-2').hexdigest()}
    assert best == ('model/s2-only.pt', sha256(b'artificial-step-2').hexdigest())
    counts = bridge_meter.snapshot()
    assert counts['attempted_forward_rows'] == counts['completed_forward_rows'] == 10
    assert counts['attempted_optimizer_updates'] == counts['completed_optimizer_updates'] == 4


@pytest.mark.parametrize('loss', [float('nan'), float('inf'), -0.1, None, 'exception'])
def test_batch_update_errors_do_not_enter_dev(
    bridge_meter: Meter, tmp_path: Path, loss: float | str | None,
) -> None:
    engine = ManualEngine(lambda epoch: 0.)
    arm: dict[str, object] = {}

    def callback(inputs: Floats, labels: Labels, meter: Meter) -> float:
        assert meter.snapshot()['attempted_optimizer_updates'] == 1
        if loss == 'exception':
            meter.reserve('forward_rows', len(inputs))
            raise ValueError('artificial runtime failure')
        engine.batch_update(inputs, labels, meter)
        return loss  # type: ignore[return-value]  # explicitly exercise out-of-contract response

    message = 'artificial runtime failure' if loss == 'exception' else 'invalid training BCE'
    with pytest.raises(ValueError, match=message):
        fit(ManualPanel(4), ManualPanel(1), engine, _order, 's2-only', arm, {'arm': arm},
            FitSettings(2, 2, 4096), bridge_meter, _store(tmp_path, 's2-only'),
            batch_update=callback)
    counts = bridge_meter.snapshot()
    assert counts['attempted_optimizer_updates'] == 1
    # Returned normally (including a bad report) means the complete update finished.
    assert counts['completed_optimizer_updates'] == (0 if loss == 'exception' else 1)
    assert counts['attempted_forward_rows'] == 2
    assert counts['completed_forward_rows'] == (0 if loss == 'exception' else 2)
    assert engine.predictions == 0 and arm == {'dev': []}


def test_independent_early_stopping_and_strict_best(
    bridge_meter: Meter, tmp_path: Path,
) -> None:
    options = FitSettings(30, 2, 4096, early_stopping=Settings(10, 8, 1e-5))
    store = _store(tmp_path, 's2-only', 'r2-joint')
    arms: dict[str, dict[str, object]] = {'s2-only': {}, 'r2-joint': {}}
    for slot, scorer, epochs, reason in (
        ('s2-only', lambda e: -e * 1e-6, 18, 'EARLY_STOPPING'),
        ('r2-joint', lambda e: -e * .01, 30, 'EPOCH_CEILING'),
    ):
        engine = ManualEngine(scorer)
        prior = bridge_meter.snapshot()['completed_optimizer_updates']
        assert isinstance(prior, int)
        engine.update_offset = prior
        arm = arms[slot]
        result = fit(ManualPanel(4), ManualPanel(1), engine, _order, slot, arm, {'arms': arms},
                     options, bridge_meter, store, batch_update=engine.batch_update)
        assert arm['epochs_completed'] == epochs and arm['stop_reason'] == reason
        selected = arm['selected']
        assert isinstance(selected, dict) and selected['epoch'] == epochs
        assert result is not None and result[0] == f'model/{slot}.pt'


def test_owner_phase_persists_paid_counts_before_first_failed_model_write(
    bridge_meter: Meter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ManualEngine(lambda epoch: 0.)
    arm: dict[str, object] = {}
    summary: dict[str, object] = {'arm': arm}
    store = _store(tmp_path, 's2-only')
    native_atomic = store._atomic
    model_attempts: list[str] = []
    run = tmp_path / 'data/bridge/run.json'

    def phase(name: str) -> None:
        if name == 'BEST_SAVE':
            store.write_snapshot({'counts': bridge_meter.snapshot(), 'summary': summary})

    def fail_model_write(path: Path, content: bytes) -> None:
        if path.suffix == '.pt':
            model_attempts.append(path.name)
            assert json.loads(run.read_bytes())['counts'] == bridge_meter.snapshot()
            raise OSError('artificial model media failure')
        native_atomic(path, content)

    monkeypatch.setattr(store, '_atomic', fail_model_write)
    with pytest.raises(ArtifactError, match='WRITE_FAILED'):
        fit(ManualPanel(4), ManualPanel(1), engine, _order, 's2-only', arm, summary,
            FitSettings(2, 2, 4096), bridge_meter, store,
            phase=phase, batch_update=engine.batch_update)
    saved = json.loads(run.read_bytes())
    assert saved['status'] == 'FAILED' and saved['artifacts'] == {}
    assert saved['counts']['completed_optimizer_updates'] == 2
    assert saved['counts']['completed_forward_rows'] == 5
    assert saved['counts'] == bridge_meter.snapshot()
    assert 'selected' not in arm and model_attempts == ['s2-only.pt']

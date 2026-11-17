"""Paired scheduling with fake updates/manual tensor outputs; zero actual fitting."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_training_steps import make_meter
from torch import Tensor

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Meter, Stopped
from cipher_lab_plugins.prediction_common.bundles import DatasetBundle, decode_bundle
from cipher_lab_plugins.prediction_common.checkpoints import load_checkpoint
from cipher_lab_plugins.residual.tests.test_residual_config import poison, residual_recipe
from cipher_lab_plugins.residual.tests.test_residual_formats import bundle

__all__ = ['poison']


class Panels:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.released = False

    def inputs(self, split: str) -> Tensor:
        if split == 'test':
            assert self.released
        x = bundle().panels['SIMON'][0].clone()
        x[1, 0] = 1.
        return x

    def labels(self, split: str) -> Tensor:
        if split == 'test':
            assert self.released
        self.events.append('labels/' + split)
        return bundle().panels['SIMON'][1]

    def reference(self, split: str) -> Tensor:
        if split == 'test':
            assert self.released
        return bundle().references['SIMON']

    def release_test(self) -> None:
        assert self.events.count('strictload') == 2
        self.events.append('release')
        self.released = True

    def test_bundle(self) -> DatasetBundle:
        assert self.released
        return replace(bundle(), panels={'SIMON': (self.inputs('test'), self.labels('test'))})


def fake_step(model: Any, x: Tensor, y: Tensor, context: object,
              optimizer: object, meter: Meter) -> float:
    # Ledger completions are fake reservations only, no real optimizer or backward.
    meter.reserve('optimizer_updates', 1)
    meter.reserve('forward_rows', len(x))
    meter.complete('forward_rows', len(x))
    meter.complete('optimizer_updates', 1)
    return .7


def prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any, Any]:
    from cipher_lab_plugins.residual import training as paired
    events: list[str] = []
    monkeypatch.setattr(paired, 'make_optimizer', lambda *args: object())
    monkeypatch.setattr(paired, 'train_step', fake_step)
    store = RunStore.claim(tmp_path, 'pair', output_bytes=10000000, needs_models=True)
    def strict(content: bytes, expected: str) -> Any:
        assert any(path.read_bytes() == content for path in (tmp_path/'models/pair').glob('*.pt'))
        loaded = load_checkpoint(content, expected)
        events.append('strictload')
        return loaded
    monkeypatch.setattr(paired, 'load_checkpoint', strict)
    return paired, Panels(events), store, make_meter()


def test_two_durable_strictloads_before_test_counts_no_repeat_forward(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    paired, data, store, meter = prepare(tmp_path, monkeypatch)
    seen_labels: list[Tensor] = []
    initial: list[str] = []
    orders: list[Tensor] = []
    original = paired.initialized_model
    def initialize(*args: Any) -> Any:
        model = original(*args)
        content = b''.join(t.numpy().tobytes() for t in model.state_dict().values())
        initial.append(hashlib.sha256(content).hexdigest())
        return model
    def step(model: Any, x: Tensor, y: Tensor, context: object,
             optimizer: object, ledger: Meter) -> float:
        seen_labels.append(y.clone())
        orders.append(x.clone())
        return fake_step(model, x, y, context, optimizer, ledger)
    monkeypatch.setattr(paired, 'initialized_model', initialize)
    monkeypatch.setattr(paired, 'train_step', step)
    result = paired.run_paired_prediction(residual_recipe(), data, meter, store,
                                         backend_identity='demo')
    assert result['status'] == 'SUCCEEDED'
    summary = result['summary']
    assert set(summary['arms']) == {'direct', 'residual'}
    assert set(summary['paired']) == {'train', 'dev', 'test'}
    assert initial[0] == initial[1]
    assert all(torch.equal(a, b) for a, b in zip(orders[:2], orders[2:], strict=True))
    assert summary['pairing']['initial_state_sha256']['direct'] == initial[0]
    assert summary['pairing']['initial_state_sha256']['residual'] == initial[1]
    assert set(summary['pairing']['panels']) == {'train', 'dev', 'test'}
    for identity in summary['pairing']['panels'].values():
        assert set(identity) == {'x_sha256', 'y_sha256', 'h_sha256', 'rows'}
    assert all(torch.equal(y.sort(dim=0).values, torch.tensor([[0.], [1.]]))
               for y in seen_labels[:2])
    assert all(torch.equal(y, torch.ones(2, 1)) for y in seen_labels[2:])
    assert data.events.index('release') > data.events.index('strictload')
    assert data.events.count('labels/train') == data.events.count('labels/dev') == 1
    assert result['counts']['train_presentations'] == 8
    assert result['counts']['dev_evaluations'] == 4
    assert meter.snapshot()['completed_optimizer_updates'] == 4  # fake only
    assert meter.snapshot()['completed_forward_rows'] == 28
    assert meter.snapshot()['completed_real_blocks'] == 0
    assert sorted(p.name for p in (tmp_path/'models/pair').iterdir()) == [
        'direct-e1.pt', 'residual-e1.pt']  # ties keep earliest
    saved = decode_bundle((tmp_path/'data/pair/dataset/test.npz').read_bytes())
    assert torch.equal(saved.references['SIMON'], bundle().references['SIMON'])
    assert summary['diagnostics']['reference_constant']['train_residual_constant'] == 1


@pytest.mark.parametrize('failure', ['second-load', 'sha', 'recipe', 'source', 'best-write',
                                     'load-baseline', 'load-selection', 'load-spec', 'load-recipe'])
def test_failure_never_releases_test_and_preserves_consumption(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    paired, data, store, meter = prepare(tmp_path, monkeypatch)
    original = paired.load_checkpoint
    count = 0
    def broken_load(content: bytes, expected: str) -> Any:
        nonlocal count
        count += 1
        if failure == 'second-load' and count == 2:
            raise ValueError('private canary')
        if failure == 'sha':
            return load_checkpoint(content, 'b'*64)
        value = original(content, expected)
        if failure == 'load-baseline':
            from cipher_lab_plugins.prediction_common.metrics import TrainBaseline
            value = replace(value, baseline={'SIMON': TrainBaseline(2, 0)})
        if failure == 'load-selection':
            value = replace(value, selection=replace(value.selection, epoch=2))
        if failure == 'load-recipe':
            value = replace(value, recipe_sha='b'*64)
        if failure == 'load-spec':
            value = replace(value, public_model_spec=replace(value.public_model_spec,
                                                            output_mode='residual'))
        return value
    monkeypatch.setattr(paired, 'load_checkpoint', broken_load)
    recipe = residual_recipe()
    if failure in ('recipe', 'source'):
        def step(*args: Any) -> float:
            loss = fake_step(*args)
            if failure == 'recipe':
                object.__setattr__(recipe, 'rounds', 5)
            else:
                from cipher_lab_plugins.prediction_common import steps as steps
                monkeypatch.setattr(
                    RunStore, 'current_source_identity', lambda self: {'tamper': 'yes'})
            return loss
        monkeypatch.setattr(paired, 'train_step', step)
    if failure == 'best-write':
        def fail(*args: Any, **kwargs: Any) -> str:
            raise ArtifactError('WRITE_FAILED')
        monkeypatch.setattr(store, 'write_model', fail)
        with pytest.raises(ArtifactError):
            paired.run_paired_prediction(recipe, data, meter, store, backend_identity='demo')
    else:
        result = paired.run_paired_prediction(recipe, data, meter, store, backend_identity='demo')
        assert result['status'] == ('STOPPED' if failure in ('source', 'recipe') else 'FAILED')
        assert 'private canary' not in repr(result)
    assert not data.released
    assert cast(int, meter.snapshot()['completed_optimizer_updates']) >= 1


@pytest.mark.parametrize('arm', ['direct', 'residual'])
@pytest.mark.parametrize('failure', ['after-forward-stop', 'after-forward-error', 'reserve'])
def test_failed_step_counts_completed_train_rows_not_attempted_batch(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, arm: str, failure: str) -> None:
    paired, data, store, meter = prepare(tmp_path, monkeypatch)
    recipe = residual_recipe()
    calls = 0
    fail_call = 1 if arm == 'direct' else recipe.training.epochs + 1
    previous_rows = 0 if arm == 'direct' else recipe.training.train * recipe.training.epochs
    previous_updates = fail_call - 1
    if failure == 'reserve':
        # Direct's completed TRAIN and epoch DEV rows precede the residual arm.
        limit = 0 if arm == 'direct' else recipe.training.epochs * (
            recipe.training.train + recipe.training.dev)
        meter = make_meter(rows=limit)

    def step(model: Any, x: Tensor, y: Tensor, context: object,
             optimizer: object, ledger: Meter) -> float:
        nonlocal calls
        calls += 1
        if calls < fail_call:
            return fake_step(model, x, y, context, optimizer, ledger)
        ledger.reserve('optimizer_updates', 1)
        ledger.reserve('forward_rows', len(x))
        # No actual model call/backward/optimizer: completion is a fake ledger
        # fixture for the precise train_step failure window after forward.
        ledger.complete('forward_rows', len(x))
        if failure == 'after-forward-error':
            raise ValueError('TRAIN_STEP_FAILED')
        raise Stopped('CPU_LIMIT')

    monkeypatch.setattr(paired, 'train_step', step)
    result = paired.run_paired_prediction(recipe, data, meter, store, backend_identity='demo')
    expected_rows = previous_rows + (0 if failure == 'reserve' else recipe.training.batch_size)
    assert result['status'] == ('FAILED' if failure == 'after-forward-error' else 'STOPPED')
    assert result['summary']['reason'] == {
        'after-forward-stop': 'CPU_LIMIT', 'after-forward-error': 'TRAINING_FAILED',
        'reserve': 'FORWARD_ROWS_LIMIT'}[failure]
    assert result['counts']['train_presentations'] == expected_rows
    assert result['counts']['updates'] == previous_updates
    assert meter.snapshot()['completed_optimizer_updates'] == previous_updates
    assert not data.released and 'labels/test' not in data.events
    # The terminal persisted record must retain this consumption, not only RAM.
    import json
    saved = json.loads((tmp_path/'data/pair/run.json').read_bytes())
    assert saved['counts']['train_presentations'] == expected_rows
    assert saved['counts']['updates'] == previous_updates


def test_each_improved_best_is_immutable_and_commits_before_selection(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    paired, data, store, meter = prepare(tmp_path, monkeypatch)
    calls = 0
    def controlled(model: Any, x: Tensor, y: Tensor, context: object,
                   optimizer: object, ledger: Meter) -> float:
        nonlocal calls
        bias = (2., 1., 1., 2.)[calls]
        calls += 1
        with torch.no_grad():
            for value in model.parameters():
                value.zero_()
            model.layers[4].bias.fill_(bias)
        return fake_step(model, x, y, context, optimizer, ledger)
    monkeypatch.setattr(paired, 'train_step', controlled)
    original = store.write_model
    def write(name: str, content: bytes, *, replace: bool = False) -> str:
        assert not replace
        # Selection in the durable running record cannot anticipate this write.
        import json
        saved = json.loads((tmp_path/'data/pair/run.json').read_bytes())
        mode = name.split('-')[0]
        selected = saved['summary']['arms'][mode]['selected']
        assert selected.get('ref') != 'model/' + name + '.pt'
        return cast(str, original(name, content))
    monkeypatch.setattr(store, 'write_model', write)
    result = paired.run_paired_prediction(residual_recipe(), data, meter, store,
                                         backend_identity='demo')
    assert result['status'] == 'SUCCEEDED'
    assert result['summary']['arms']['direct']['selected']['epoch'] == 2
    assert result['summary']['arms']['residual']['selected']['epoch'] == 2
    assert sorted(p.name for p in (tmp_path/'models/pair').iterdir()) == [
        'direct-e1.pt', 'direct-e2.pt', 'residual-e1.pt', 'residual-e2.pt']

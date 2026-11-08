"""Synthetic, explicitly counted engineering runs; state tests replace narrow step."""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from test_training_steps import make_meter
from torch import Tensor

from cipher_lab.artifacts import ArtifactError, RunStore, read_saved
from cipher_lab.data.bundles import DatasetBundle, PredictionTask, decode_bundle
from cipher_lab.execution import Meter, Stopped
from cipher_lab.training import prediction
from cipher_lab.training.prediction import run_prediction
from cipher_lab.training_config import MLPSpec, PredictionRecipe, TrainSpec


class ArtificialPrediction:
    def __init__(self) -> None:
        self.released = False
        self.calls: list[str] = []

    def inputs(self, split: str) -> Tensor:
        self.calls.append('inputs/' + split)
        if split == 'test':
            assert self.released
        return torch.zeros(2, 64)

    def labels(self, split: str) -> Tensor:
        self.calls.append('labels/' + split)
        if split == 'test':
            assert self.released
        return torch.tensor([[0.], [1.]])

    def release_test(self) -> None:
        self.calls.append('release')
        self.released = True

    def test_bundle(self) -> DatasetBundle:
        assert self.released
        self.calls.append('bundle')
        return DatasetBundle('bit-prediction-v1', (PredictionTask('SIMON', 3, 0, 'null'),),
                             {'SIMON': (self.inputs('test'), self.labels('test'))})


def recipe(epochs: int = 1) -> PredictionRecipe:
    return PredictionRecipe(1, 'bit-prediction-v1', MLPSpec(2),
                            TrainSpec(2, 2, 2, 2, epochs, .01, 5, 7, 11), 3, 0, 'null')


def fake_step(model: object, x: Tensor, y: Tensor, context: object,
              optimizer: object, meter: Meter) -> float:
    """No gradients or optimizer updates, ledger only for method sequencing."""
    meter.reserve('optimizer_updates', 1)
    meter.reserve('forward_rows', len(x))
    meter.complete('forward_rows', len(x))
    meter.complete('optimizer_updates', 1)
    return .7


def summary(result: Mapping[str, object]) -> dict[str, object]:
    return cast(dict[str, object], result['summary'])


def test_prediction_real_two_updates_closed_loop(tmp_path: Path) -> None:
    """Exactly two actual Adam steps per invocation, never scientific data."""
    data, meter = ArtificialPrediction(), make_meter()
    store = RunStore.claim(tmp_path, 'prediction', output_bytes=10000000, needs_models=True)
    rng = torch.get_rng_state().clone()
    start = time.monotonic()
    result = run_prediction(recipe(2), data, meter, store, backend_identity='demo')
    elapsed = time.monotonic() - start
    print(f'ENGINEERING prediction actual_optimizer_steps=2 elapsed_seconds={elapsed:.3f}')
    assert elapsed < 60 and result['status'] == 'SUCCEEDED'
    assert torch.equal(rng, torch.get_rng_state())
    stats = meter.snapshot()
    assert stats['completed_optimizer_updates'] == 2
    assert stats['completed_forward_rows'] == 14  # train4 + epochDEV4 + final6
    saved = read_saved(tmp_path, 'prediction')
    assert summary(saved)['scientific_validation'] is False
    assert summary(saved)['synthetic'] is True
    expected = hashlib.sha256(json.dumps(recipe(2).to_dict(), ensure_ascii=False,
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert summary(saved)['recipe_sha256'] == expected
    assert data.calls.index('release') < data.calls.index('inputs/test')
    assert data.calls.count('bundle') == 1
    bundle = decode_bundle((tmp_path / 'data/prediction/dataset/test.npz').read_bytes())
    assert bundle.task_declarations == (PredictionTask('SIMON', 3, 0, 'null'),)
    assert sorted(p.name for p in (tmp_path / 'models/prediction').iterdir()) == ['best.pt']


def test_tie_keeps_first_saved_and_load_precedes_release(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prediction, 'train_step', fake_step)
    data, meter = ArtificialPrediction(), make_meter()
    store = RunStore.claim(tmp_path, 'tie', output_bytes=10000000, needs_models=True)
    # No real optimizer; unchanged model gives identical epoch DEV BCE.
    result = run_prediction(recipe(2), data, meter, store, backend_identity='demo')
    assert result['status'] == 'SUCCEEDED'
    selected = cast(dict[str, object], summary(result)['selected'])
    assert selected['epoch'] == 1 and selected['updates'] == 1
    assert data.calls.count('labels/train') == 1
    assert data.calls.count('labels/dev') == 1


@pytest.mark.parametrize('failure', ['load', 'encode', 'model', 'dataset', 'source',
                                    'step', 'stop', 'rows', 'final-clock'])
def test_public_failure_no_early_test_and_no_storage_retry(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    from cipher_lab.training import steps
    data, meter = ArtificialPrediction(), make_meter(rows=0 if failure == 'rows' else 10000)
    store = RunStore.claim(tmp_path, 'failure', output_bytes=10000000, needs_models=True)
    monkeypatch.setattr(prediction, 'train_step', fake_step)
    def broken(*args: object, **kwargs: object) -> object:
        raise RuntimeError('PRIVATE-CANARY')
    if failure in ('load', 'encode'):
        monkeypatch.setattr(steps, failure + '_checkpoint', broken)
    if failure in ('model', 'dataset'):
        def storage_fail(*args: object, **kwargs: object) -> str:
            raise ArtifactError('WRITE_FAILED')
        monkeypatch.setattr(store, 'write_' + failure, storage_fail)
    if failure == 'source':
        monkeypatch.setattr(steps, 'source_identity', lambda: {'wrong': 'identity'})
    if failure in ('step', 'stop'):
        def step_fail(*args: object, **kwargs: object) -> float:
            if failure == 'stop':
                raise Stopped('CPU_LIMIT')
            raise RuntimeError('PRIVATE-CANARY')
        monkeypatch.setattr(prediction, 'train_step', step_fail)
    if failure == 'final-clock':
        original = store.write_payload
        def payload(name: str, value: object) -> str:
            ref = original(name, value)
            if name == 'metrics':
                def expired() -> None:
                    raise Stopped('WALL_LIMIT')
                monkeypatch.setattr(meter, 'check', expired)
            return ref
        monkeypatch.setattr(store, 'write_payload', payload)
    if failure in ('model', 'dataset'):
        with pytest.raises(ArtifactError, match='WRITE_FAILED'):
            run_prediction(recipe(), data, meter, store, backend_identity='demo')
    else:
        result = run_prediction(recipe(), data, meter, store, backend_identity='demo')
        assert result['status'] == ('STOPPED' if failure in (
            'source', 'stop', 'rows', 'final-clock') else 'FAILED')
        assert 'PRIVATE' not in repr(result)
    if failure not in ('dataset', 'final-clock'):
        assert not data.released and not any('test' in c for c in data.calls)
    if failure == 'source':
        assert data.calls == []
    if failure == 'load':
        assert (tmp_path / 'models/failure/best.pt').exists()
        assert meter.snapshot()['completed_optimizer_updates'] == 1


def test_backend_is_explicit_and_never_certifies_cipher(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prediction, 'train_step', fake_step)
    store = RunStore.claim(tmp_path, 'identity', output_bytes=10000000, needs_models=True)
    with pytest.raises(ValueError, match='INVALID_BACKEND_IDENTITY'):
        run_prediction(recipe(), ArtificialPrediction(), make_meter(), store,
                       backend_identity='ciphers')
    result = run_prediction(replace(recipe(), label_mode='cipher'), ArtificialPrediction(),
                            make_meter(), store, backend_identity='simon')
    # Artificial bundle declaration differs: method refuses it, not fake science.
    assert result['status'] == 'FAILED'
    assert summary(result)['scientific_validation'] is False
    assert summary(result)['synthetic'] is False


@pytest.mark.parametrize('biases,selected_epoch', [((1., 2.), 1), ((2., 1.), 2)])
def test_final_metrics_use_selected_not_last_and_best_replace(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, biases: tuple[float, float], selected_epoch: int) -> None:
    import math

    from cipher_lab.models import BitMLP
    calls = 0
    def controlled_step(model: BitMLP, x: Tensor, y: Tensor, task: object,
                        optimizer: object, meter: Meter) -> float:
        nonlocal calls
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            cast(Tensor, model.layers[4].bias).fill_(biases[calls])
        calls += 1
        return fake_step(model, x, y, task, optimizer, meter)
    monkeypatch.setattr(prediction, 'train_step', controlled_step)
    store = RunStore.claim(tmp_path, 'selected', output_bytes=10000000, needs_models=True)
    result = run_prediction(recipe(2), ArtificialPrediction(), make_meter(), store,
                            backend_identity='demo')
    assert result['status'] == 'SUCCEEDED'
    selected = cast(dict[str, object], summary(result)['selected'])
    assert selected['epoch'] == selected_epoch
    metrics = cast(dict[str, dict[str, object]], summary(result)['metrics'])
    for split in ('train', 'dev', 'test'):
        task = cast(dict[str, dict[str, float]], metrics[split]['tasks'])['SIMON']
        assert task['bce'] == pytest.approx(math.log1p(math.exp(1.)) - .5)
    saved = read_saved(tmp_path, 'selected')
    artifact = cast(dict[str, dict[str, object]], saved['artifacts'])['model/best.pt']
    assert artifact['sha256'] == selected['sha256']


def test_corrupt_checkpoint_rejected_before_release(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab.training import steps
    monkeypatch.setattr(prediction, 'train_step', fake_step)
    monkeypatch.setattr(steps, 'encode_checkpoint', lambda *a: b'not a checkpoint')
    store = RunStore.claim(tmp_path, 'corrupt', output_bytes=10000000, needs_models=True)
    data = ArtificialPrediction()
    result = run_prediction(recipe(), data, make_meter(), store, backend_identity='demo')
    assert result['status'] == 'FAILED' and not data.released


def test_source_change_after_update_keeps_completed_stats(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab.training import steps
    def change(model: object, x: Tensor, y: Tensor, task: object,
               optimizer: object, meter: Meter) -> float:
        value = fake_step(model, x, y, task, optimizer, meter)
        monkeypatch.setattr(steps, 'source_identity', lambda: {'changed': 'yes'})
        return value
    monkeypatch.setattr(prediction, 'train_step', change)
    store = RunStore.claim(tmp_path, 'changed', output_bytes=10000000, needs_models=True)
    meter = make_meter()
    result = run_prediction(recipe(), ArtificialPrediction(), meter, store, backend_identity='demo')
    assert result['status'] == 'STOPPED'
    assert summary(result)['reason'] == 'IDENTITY_CHANGED'
    assert meter.snapshot()['completed_optimizer_updates'] == 1
    counts = cast(dict[str, object], result['counts'])
    assert counts['updates'] == 1
    assert counts['panels'] == {'train/SIMON': 2, 'dev/SIMON': 2}


def test_payload_nonstorage_failure_fixed_failed_terminal(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prediction, 'train_step', fake_step)
    store = RunStore.claim(tmp_path, 'payload', output_bytes=10000000, needs_models=True)
    def fail(*args: object, **kwargs: object) -> str:
        raise RuntimeError('PRIVATE-CANARY')
    monkeypatch.setattr(store, 'write_payload', fail)
    result = run_prediction(recipe(), ArtificialPrediction(), make_meter(), store,
                            backend_identity='demo')
    assert result['status'] == 'FAILED'
    assert 'PRIVATE' not in repr(result)


def test_strict_sha_mismatch_stays_failed_without_test(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab.evaluation.checkpoints import load_checkpoint
    from cipher_lab.training import steps
    original = load_checkpoint
    def wrong_sha(content: bytes, expected: str) -> object:
        return original(content, '0'*64)
    monkeypatch.setattr(steps, 'load_checkpoint', wrong_sha)
    monkeypatch.setattr(prediction, 'train_step', fake_step)
    data = ArtificialPrediction()
    store = RunStore.claim(tmp_path, 'sha', output_bytes=10000000, needs_models=True)
    result = run_prediction(recipe(), data, make_meter(), store, backend_identity='demo')
    assert result['status'] == 'FAILED' and not data.released


def test_release_observes_encoded_saved_and_strictly_loaded_selected_identity(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab.evaluation.checkpoints import load_checkpoint
    from cipher_lab.training import steps
    loaded = False
    store = RunStore.claim(tmp_path, 'gate', output_bytes=10000000, needs_models=True)
    def tracked_load(content: bytes, expected: str) -> object:
        nonlocal loaded
        saved = read_saved(tmp_path, 'gate')
        artifact = cast(dict[str, dict[str, object]], saved['artifacts'])['model/best.pt']
        assert artifact['sha256'] == expected
        assert (tmp_path / 'models/gate/best.pt').read_bytes() == content
        value = load_checkpoint(content, expected)
        loaded = True
        return value
    class Gated(ArtificialPrediction):
        def release_test(self) -> None:
            assert loaded
            super().release_test()
    monkeypatch.setattr(steps, 'load_checkpoint', tracked_load)
    monkeypatch.setattr(prediction, 'train_step', fake_step)
    data = Gated()
    result = run_prediction(recipe(), data, make_meter(), store, backend_identity='demo')
    assert result['status'] == 'SUCCEEDED' and data.released


def test_checkpoint_storage_failure_preserves_running_owner_without_second_finalize(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prediction, 'train_step', fake_step)
    store = RunStore.claim(tmp_path, 'disk', output_bytes=10000000, needs_models=True)
    def fail(*args: object, **kwargs: object) -> str:
        # Emulate a storage owner that cannot even persist a FAILED record.
        raise ArtifactError('WRITE_FAILED')
    monkeypatch.setattr(store, 'write_model', fail)
    meter = make_meter()
    with pytest.raises(ArtifactError):
        run_prediction(recipe(), ArtificialPrediction(), meter, store, backend_identity='demo')
    saved = read_saved(tmp_path, 'disk')
    assert saved['status'] == 'RUNNING'  # UNKNOWN, no invented successful terminal
    assert cast(dict[str, object], saved['counts'])['updates'] == 1
    assert meter.snapshot()['completed_optimizer_updates'] == 1


def test_claim_source_read_failure_is_fixed_failed_and_does_not_touch_data(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    store = RunStore.claim(tmp_path, 'claimread', output_bytes=10000000, needs_models=True)
    def fail() -> dict[str, str]:
        raise RuntimeError('PRIVATE-CANARY')
    monkeypatch.setattr(store, 'source_snapshot', fail)
    data = ArtificialPrediction()
    result = run_prediction(recipe(), data, make_meter(), store, backend_identity='demo')
    assert result['status'] == 'FAILED' and not data.calls
    assert 'PRIVATE' not in repr(result)


def test_data_failure_keeps_initial_identity_and_completed_train_panel(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prediction, 'train_step', fake_step)
    class Broken(ArtificialPrediction):
        def labels(self, split: str) -> Tensor:
            if split == 'dev':
                raise RuntimeError('PRIVATE-CANARY')
            return super().labels(split)
    store = RunStore.claim(tmp_path, 'datafail', output_bytes=10000000, needs_models=True)
    result = run_prediction(recipe(), Broken(), make_meter(), store, backend_identity='demo')
    saved = read_saved(tmp_path, 'datafail')
    assert result['status'] == saved['status'] == 'FAILED'
    assert cast(dict[str, object], saved['counts'])['panels'] == {'train/SIMON': 2}
    assert summary(saved)['task_declarations'] == [
        {'cipher': 'SIMON', 'rounds': 3, 'target_bit': 0, 'label_mode': 'null'}]
    assert summary(saved)['backend_identity'] == 'demo'
    assert 'PRIVATE' not in repr(saved)


def test_model_owner_is_required_and_stop_preserves_meter(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prediction, 'train_step', fake_step)
    store = RunStore.claim(tmp_path, 'nomodels', output_bytes=10000000)
    data, meter = ArtificialPrediction(), make_meter()
    with pytest.raises(ArtifactError):
        run_prediction(recipe(), data, meter, store, backend_identity='demo')
    assert not data.released and not (tmp_path / 'models/nomodels').exists()
    assert meter.snapshot()['completed_optimizer_updates'] == 1

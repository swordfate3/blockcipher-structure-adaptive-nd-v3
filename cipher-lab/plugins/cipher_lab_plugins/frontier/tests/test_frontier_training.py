"""All ledger units in this module are artificial; never fitted scientific work."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_training_steps import make_meter
from torch import Tensor

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Meter, Stopped
from cipher_lab_plugins.frontier.tests.frontier_fakes import plan, poison, recipe
from cipher_lab_plugins.frontier.tests.test_frontier_formats import bundle
from cipher_lab_plugins.prediction_common.checkpoints import load_checkpoint

__all__ = ['poison']


class Panels:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.released = False

    def inputs(self, split: str) -> Tensor:
        if split == 'test':
            assert self.released
        x = bundle().panels['SIMON'][0].clone()
        x[1, 0] = 1
        return cast(Tensor, x)

    def labels(self, split: str) -> Tensor:
        if split == 'test':
            assert self.released
        self.events.append('labels/' + split)
        return cast(Tensor, bundle().panels['SIMON'][1])

    def reference(self, split: str) -> Tensor:
        if split == 'test':
            assert self.released
        return cast(Tensor, bundle().references['SIMON'])

    def release_test(self) -> None:
        assert self.events.count('strictload') == 3
        self.events.append('release')
        self.released = True

    def test_bundle(self) -> Any:
        return replace(bundle(), panels={'SIMON': (self.inputs('test'), self.labels('test'))})


def fake_step(model: Any, x: Tensor, y: Tensor, context: object,
              optimizer: object, meter: Meter) -> float:
    meter.reserve('optimizer_updates', 1)
    meter.reserve('forward_rows', len(x))
    meter.complete('forward_rows', len(x))
    meter.complete('optimizer_updates', 1)
    return .7


def prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any, Any]:
    from cipher_lab_plugins.frontier import training as feature_probe
    events: list[str] = []
    monkeypatch.setattr(feature_probe, 'make_optimizer', lambda *args: object())
    monkeypatch.setattr(feature_probe, 'train_step', fake_step)
    store = RunStore.claim(tmp_path, 'probe', output_bytes=10000000, needs_models=True)

    def strict(content: bytes, sha: str) -> Any:
        loaded = load_checkpoint(content, sha)
        events.append('strictload')
        return loaded
    monkeypatch.setattr(feature_probe, 'load_checkpoint', strict)
    return feature_probe, Panels(events), store, make_meter()


def test_three_durable_selections_pairing_and_no_repeat_forward(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    probe, data, store, meter = prepare(tmp_path, monkeypatch)
    orders: list[Tensor] = []
    labels: list[Tensor] = []
    initial: list[Any] = []
    original = probe.initialized_arm

    def initialize(*args: Any) -> Any:
        model = original(*args)
        initial.append({k: v.clone() for k, v in model.state_dict().items()})
        return model

    def step(model: Any, x: Tensor, y: Tensor, context: object,
             optimizer: object, ledger: Meter) -> float:
        orders.append(x.clone())
        labels.append(y.clone())
        return fake_step(model, x, y, context, optimizer, ledger)
    monkeypatch.setattr(probe, 'initialized_arm', initialize)
    monkeypatch.setattr(probe, 'train_step', step)
    result = probe.run_frontier(recipe(), data, meter, store, backend_identity='demo', plan=plan())
    assert result['status'] == 'SUCCEEDED'
    summary = result['summary']
    assert set(summary['arms']) == {'A', 'B', 'C'}
    assert all(torch.equal(initial[0][k], initial[1][k]) for k in initial[0])
    assert all(torch.equal(initial[0][k], initial[2][k]) for k in initial[0]
               if not k.startswith('layers.0.'))
    assert initial[2]['layers.0.weight'].shape == (2, 89)
    assert all(torch.equal(orders[i], orders[i+2]) and torch.equal(orders[i], orders[i+4])
               for i in range(2))
    assert all(torch.equal(y, torch.ones(2, 1)) for y in labels[2:])
    assert all(int(y.sum()) == 1 for y in labels[:2])
    assert result['counts']['updates'] == 6
    assert result['counts']['train_presentations'] == 12
    assert result['counts']['dev_evaluations'] == 6
    assert meter.snapshot()['completed_forward_rows'] == 42  # artificial ledger only
    assert meter.snapshot()['completed_real_blocks'] == 0
    assert sorted(p.name for p in (tmp_path/'models/probe').iterdir()) == [
        'A-best.pt', 'B-best.pt', 'C-best.pt']
    assert all(summary['arms'][a]['selected']['epoch'] == 1 for a in 'ABC')
    assert [summary['arms'][a]['parameter_count'] for a in 'ABC'] == [139, 139, 189]
    assert set(summary['comparisons']) == {'C-A', 'C-B'}
    assert all(set(v) == {'train', 'dev', 'test'} for v in summary['comparisons'].values())
    assert set(summary['pairing']['panels']) == {'train', 'dev', 'test'}
    full = summary['pairing']['initial_state_sha256']
    assert full['A'] == full['B'] != full['C']
    shared = summary['pairing']['shared_layers_sha256']
    assert len(set(shared.values())) == 1
    assert set(summary['arm_resources']) == {'A', 'B', 'C'}
    assert data.events.count('labels/train') == data.events.count('labels/dev') == 1


@pytest.mark.parametrize('failure', ['third-load', 'sha', 'spec', 'selection', 'recipe',
                                     'baseline', 'source', 'write'])
def test_third_failure_never_releases_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          failure: str) -> None:
    probe, data, store, meter = prepare(tmp_path, monkeypatch)
    original = probe.load_checkpoint
    count = 0

    def broken(content: bytes, sha: str) -> Any:
        nonlocal count
        count += 1
        if count != 3:
            return original(content, sha)
        if failure == 'third-load':
            raise ValueError('private canary')
        if failure == 'sha':
            return original(content, 'b'*64)
        loaded = original(content, sha)
        if failure == 'spec':
            from cipher_lab_plugins.frontier.tests.test_frontier_formats import spec
            return replace(loaded, public_model_spec=spec('B'))
        if failure == 'selection':
            return replace(loaded, selection=replace(loaded.selection, epoch=2))
        if failure == 'recipe':
            return replace(loaded, recipe_sha='b'*64)
        if failure == 'baseline':
            from cipher_lab_plugins.prediction_common.metrics import TrainBaseline
            return replace(loaded, baseline={'SIMON': TrainBaseline(2, 0)})
        if failure == 'source':
            from cipher_lab_plugins.prediction_common import steps as steps
            monkeypatch.setattr(
                    RunStore, 'current_source_identity', lambda self: {'tamper': 'yes'})
        return loaded
    monkeypatch.setattr(probe, 'load_checkpoint', broken)
    if failure == 'write':
        write = store.write_model

        def fail(name: str, content: bytes, *, replace: bool = False) -> str:
            if name.startswith('C-'):
                raise ArtifactError('WRITE_FAILED')
            return cast(str, write(name, content))
        monkeypatch.setattr(store, 'write_model', fail)
        with pytest.raises(ArtifactError):
            probe.run_frontier(recipe(), data, meter, store, backend_identity='demo', plan=plan())
    else:
        result = probe.run_frontier(recipe(), data, meter, store,
                                    backend_identity='demo', plan=plan())
        assert result['status'] == ('STOPPED' if failure == 'source' else 'FAILED')
        assert 'private canary' not in repr(result)
    assert not data.released
    assert meter.snapshot()['completed_optimizer_updates'] == (5 if failure == 'write' else 6)


@pytest.mark.parametrize('arm', ['A', 'B', 'C'])
@pytest.mark.parametrize('failure', ['after-forward', 'reserve'])
def test_failed_step_keeps_rows_without_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               arm: str, failure: str) -> None:
    probe, data, store, meter = prepare(tmp_path, monkeypatch)
    before = 'ABC'.index(arm)*2
    calls = 0
    if failure == 'reserve':
        meter = make_meter(rows=before*4)

    def fail(model: Any, x: Tensor, y: Tensor, context: object,
             optimizer: object, ledger: Meter) -> float:
        nonlocal calls
        calls += 1
        if calls <= before:
            return fake_step(model, x, y, context, optimizer, ledger)
        ledger.reserve('optimizer_updates', 1)
        ledger.reserve('forward_rows', len(x))
        ledger.complete('forward_rows', len(x))
        raise ValueError('manual post-forward failure')
    monkeypatch.setattr(probe, 'train_step', fail)
    result = probe.run_frontier(recipe(), data, meter, store, backend_identity='demo', plan=plan())
    assert result['status'] == ('FAILED' if failure == 'after-forward' else 'STOPPED')
    assert result['counts']['updates'] == before
    assert result['counts']['train_presentations'] == before*2 + (2 if failure == 'after-forward'
                                                                else 0)
    assert not data.released


def test_meter_cumulative_budget_excludes_wait_and_covers_final() -> None:
    from cipher_lab.execution import Clocks
    t = [0., 0.]
    meter = make_meter()
    meter._clocks = Clocks(lambda: t[0], lambda: t[1], meter._clocks.utc)
    # Reset absolute origin for this artificial clock.
    meter._start_wall = meter._start_cpu = 0
    meter._policy = replace(meter._policy, wall_seconds=1800., cpu_seconds=1800.)
    budget = meter.cumulative_budget(wall_seconds=600, cpu_seconds=600)
    with budget:
        t[:] = [300., 200.]
        meter.check()
    t[:] = [900., 800.]  # waiting for other arms is not charged to this one
    with budget:
        t[:] = [1199., 1000.]
        meter.check()
    assert budget.snapshot()['wall_seconds'] == 599
    assert budget.snapshot()['cpu_seconds'] == 400
    with pytest.raises(Stopped, match='ARM_WALL_LIMIT'), budget:
        t[0] += 1
        meter.check()
    assert budget.snapshot()['wall_seconds'] == 600


@pytest.mark.parametrize('phase', ['train', 'final'])
@pytest.mark.parametrize('kind', ['wall', 'cpu'])
def test_arm_limits_cover_training_and_final(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            phase: str, kind: str) -> None:
    from cipher_lab.execution import Clocks
    probe, data, store, meter = prepare(tmp_path, monkeypatch)
    t = [0., 0.]
    meter._clocks = Clocks(lambda: t[0], lambda: t[1], meter._clocks.utc)
    meter._start_wall = meter._start_cpu = 0
    meter._policy = replace(meter._policy, wall_seconds=1800., cpu_seconds=1800.)
    index = 0 if kind == 'wall' else 1
    if phase == 'train':
        def slow(*args: Any) -> float:
            value = fake_step(*args)
            t[index] += 600
            return value
        monkeypatch.setattr(probe, 'train_step', slow)
    else:
        original = probe.evaluate_batches

        def final(*args: Any, **kwargs: Any) -> Any:
            value = original(*args, **kwargs)
            if 'outputs' in kwargs:
                t[index] += 600
            return value
        monkeypatch.setattr(probe, 'evaluate_batches', final)
    result = probe.run_frontier(recipe(), data, meter, store, backend_identity='demo', plan=plan())
    assert result['status'] == 'STOPPED'
    assert result['summary']['reason'] == f'ARM_{kind.upper()}_LIMIT'
    assert result['summary']['arm_resources']['A'][kind+'_seconds'] == 600
    assert data.released == (phase == 'final')


def test_record_accepts_raw_panel_without_shared_m_dispatch(tmp_path: Path) -> None:
    from cipher_lab_plugins.frontier.tests.test_frontier_formats import spec
    from cipher_lab_plugins.prediction_common.steps import TrainingRecord
    store = RunStore.claim(tmp_path, 'panel', output_bytes=10000000, needs_models=True)
    record = TrainingRecord(recipe(), spec('A'), make_meter(), store, 'demo')
    record.begin()
    record.panel('train', 'SIMON', torch.zeros(2, 64), torch.zeros(2, 1))
    assert record.baseline['SIMON'].rows == 2


def test_budget_does_not_mask_sealed_storage_failure() -> None:
    from cipher_lab.execution import Clocks
    t = [0.]
    meter = make_meter()
    meter._clocks = Clocks(lambda: t[0], lambda: 0., meter._clocks.utc)
    budget = meter.cumulative_budget(wall_seconds=600, cpu_seconds=600)
    with pytest.raises(ArtifactError, match='WRITE_FAILED'), budget:
        t[0] = 600.
        raise ArtifactError('WRITE_FAILED')
    assert budget.snapshot()['wall_seconds'] == 600
    assert meter._active_budget is None


def test_comparison_names_absolute_bce_direction(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    probe, data, store, meter = prepare(tmp_path, monkeypatch)
    result = probe.run_frontier(recipe(), data, meter, store, backend_identity='demo', plan=plan())
    assert result['status'] == 'SUCCEEDED'
    for name in ('C-A', 'C-B'):
        for values in result['summary']['comparisons'][name].values():
            assert values['loss_difference']['direction'] == name


@pytest.mark.parametrize('bad', ['unavailable', 'backwards', 'cycle-limit'])
def test_cumulative_clock_fail_closed(bad: str) -> None:
    from cipher_lab.execution import Clocks
    t = [0.]
    meter = make_meter()
    meter._clocks = Clocks(lambda: t[0], lambda: 0., meter._clocks.utc)
    meter._policy = replace(meter._policy, wall_seconds=1800.)
    budget = meter.cumulative_budget(wall_seconds=600, cpu_seconds=600)
    with budget:
        t[0] = 1.
        meter.check()
    if bad == 'cycle-limit':
        t[0] = 1800.  # all inactive time still belongs to the parent meter
        expected = 'WALL_LIMIT'
    else:
        expected = 'CLOCK_UNAVAILABLE'
    with pytest.raises(Stopped, match=expected), budget:
        t[0] = float('nan') if bad == 'unavailable' else -1.
    assert meter._active_budget is None


def test_initialization_streams_fan_in_rng_and_independent_storage() -> None:
    import hashlib

    from cipher_lab_plugins.frontier.tests.test_frontier_formats import spec
    from cipher_lab_plugins.frontier.training import initialized_arm
    before = torch.get_rng_state().clone()
    models = [initialized_arm(spec(name), 31) for name in 'ABC']
    assert torch.equal(before, torch.get_rng_state())
    for model, fan_in in zip(models, (64, 64, 89), strict=True):
        state = model.state_dict()
        domain = f'feature-probe-v1/31/layers.0/{fan_in}'.encode('ascii')
        seed = int.from_bytes(hashlib.sha256(domain).digest()[:8], 'big')
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            expected = torch.nn.Linear(fan_in, 2)
        assert torch.equal(state['layers.0.weight'], expected.weight)
        assert torch.equal(state['layers.0.bias'], expected.bias)
    for key in models[0].state_dict():
        assert models[0].state_dict()[key].data_ptr() != models[1].state_dict()[key].data_ptr()


def test_one_prediction_owner_synthetic_key_and_cached_reference(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.prediction_common.data import PredictionData
    probe, _, store, meter = prepare(tmp_path, monkeypatch)
    meter._policy = replace(meter._policy, real_blocks=12)
    draws: list[int] = []
    keys: list[tuple[int, ...]] = []
    references: list[tuple[int, ...]] = []

    def finite(size: int) -> bytes:
        draws.append(size)
        return b'\x01'*size  # finite artificial fixture, not private RNG

    def target(points: tuple[int, ...], key: tuple[int, ...], rounds: int) -> tuple[int, ...]:
        keys.append(key)
        return tuple(0 for _ in points)  # no cipher

    def reference(points: tuple[int, ...], key: tuple[int, ...], rounds: int) -> tuple[int, ...]:
        assert key == (0, 0, 0, 0)
        references.append(points)
        return tuple(1 << 63 for _ in points)  # no cipher

    data = PredictionData(recipe(), target, finite, meter, reference_oracle=reference)
    release = data.release_test
    loads: list[str] = []
    strict = probe.load_checkpoint

    def load(*args: Any) -> Any:
        value = strict(*args)
        loads.append(value.public_model_spec.arm_id)
        return value

    def gated() -> None:
        assert loads == ['A', 'B', 'C']
        assert len(keys) == len(references) == 2
        release()
    monkeypatch.setattr(probe, 'load_checkpoint', load)
    monkeypatch.setattr(data, 'release_test', gated)
    result = probe.run_frontier(recipe(), data, meter, store, backend_identity='demo', plan=plan())
    assert result['status'] == 'SUCCEEDED'
    assert draws == [16] and len(keys) == 3 and len(set(keys)) == 1
    assert len(references) == 3
    assert meter.snapshot()['completed_real_blocks'] == 12  # artificial ledger, no real cipher
    assert result['counts']['train_presentations'] == 12


def test_absolute_final_logits_reused_with_fixed_candidate_direction(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    probe, data, store, meter = prepare(tmp_path, monkeypatch)
    initialize = probe.initialized_arm

    def constant(*args: Any) -> Any:
        model = initialize(*args)
        with torch.no_grad():
            model.state_dict()['layers.4.bias'].fill_(2.)
        return model
    monkeypatch.setattr(probe, 'initialized_arm', constant)
    result = probe.run_frontier(recipe(), data, meter, store, backend_identity='demo', plan=plan())
    assert result['status'] == 'SUCCEEDED'
    summary = result['summary']
    for split in ('train', 'dev', 'test'):
        assert summary['arms']['A']['metrics'][split]['tasks']['SIMON']['acc'] == .5
        assert summary['arms']['B']['metrics'][split]['tasks']['SIMON']['acc'] == 1.
        assert summary['arms']['C']['metrics'][split]['tasks']['SIMON']['acc'] == 1.
        assert summary['comparisons']['C-A'][split]['delta'] == .5
        assert summary['comparisons']['C-B'][split]['delta'] == 0
        assert summary['comparisons']['C-B'][split]['loss_difference']['mean'] == 0
    assert meter.snapshot()['attempted_forward_rows'] == 42
    assert meter.snapshot()['completed_forward_rows'] == 42


def test_independent_dev_selection_and_stable_files(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    probe, data, store, meter = prepare(tmp_path, monkeypatch)
    original = probe.evaluate_batches
    curves = {'A': iter([.8, .7]), 'B': iter([.6, .6]), 'C': iter([.9, .5])}

    def dev(model: Any, panels: Any, spec: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(model, panels, spec, *args, **kwargs)
        if 'outputs' not in kwargs:
            result['mean_bce'] = next(curves[spec.arm_id])
        return result
    monkeypatch.setattr(probe, 'evaluate_batches', dev)
    result = probe.run_frontier(recipe(), data, meter, store, backend_identity='demo', plan=plan())
    assert result['status'] == 'SUCCEEDED'
    assert sorted(p.name for p in (tmp_path/'models/probe').iterdir()) == [
        'A-best.pt', 'B-best.pt', 'C-best.pt']
    for arm, epoch in zip('ABC', (2, 1, 2), strict=True):
        selected = result['summary']['arms'][arm]['selected']
        assert selected['epoch'] == epoch
        assert selected['ref'].endswith(f'{arm}-best.pt')
        assert len(result['summary']['arms'][arm]['learning_curve']) == 2


def test_nonreturning_step_does_not_credit_owner_update(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    probe, data, store, meter = prepare(tmp_path, monkeypatch)

    def fail_after_completion(*args: Any) -> float:
        fake_step(*args)
        raise ValueError('synthetic post-step failure')
    monkeypatch.setattr(probe, 'train_step', fail_after_completion)
    result = probe.run_frontier(recipe(), data, meter, store, backend_identity='demo', plan=plan())
    assert result['status'] == 'FAILED'
    assert result['counts']['updates'] == 0
    assert result['counts']['train_presentations'] == 2
    assert meter.snapshot()['completed_optimizer_updates'] == 1  # fake work is never refunded
    assert not data.released


@pytest.mark.parametrize('damage', ['delete', 'truncate', 'same-size'])
def test_selected_c_disk_damage_blocks_test_without_memory_damage(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    probe, data, store, meter = prepare(tmp_path, monkeypatch)
    selected_model = probe._Arm.selected_model
    finalize_calls: list[object] = []
    finalize = store.finalize

    def finalized(value: Any) -> None:
        finalize_calls.append(value)
        finalize(value)

    def damage_c(arm: Any, record: Any) -> Any:
        if arm.spec.arm_id == 'C':
            path = tmp_path/'models/probe/C-best.pt'
            original = path.read_bytes()
            assert arm.content == original  # memory remains intact, only disk is damaged
            if damage == 'delete':
                path.unlink()
            elif damage == 'truncate':
                path.write_bytes(original[:len(original)//2])
            else:
                path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            assert arm.content == original
        return selected_model(arm, record)
    monkeypatch.setattr(probe._Arm, 'selected_model', damage_c)
    monkeypatch.setattr(store, 'finalize', finalized)
    with pytest.raises(ArtifactError):
        probe.run_frontier(recipe(), data, meter, store, backend_identity='demo', plan=plan())
    assert not data.released and data.events.count('strictload') == 2
    assert finalize_calls == []
    assert meter.snapshot()['completed_optimizer_updates'] == 6
    assert meter.snapshot()['completed_forward_rows'] == 24


def test_selected_checkpoint_reads_owner_ref_not_memory(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    probe, data, store, meter = prepare(tmp_path, monkeypatch)
    read_model = store.read_model
    refs: list[str] = []

    def read(ref: str, sha: str, max_bytes: int) -> bytes:
        refs.append(ref)
        return cast(bytes, read_model(ref, sha, max_bytes))
    selected_model = probe._Arm.selected_model

    def no_memory(arm: Any, record: Any) -> Any:
        arm.content = b'not a checkpoint: persistence is the authority'
        return selected_model(arm, record)
    monkeypatch.setattr(store, 'read_model', read)
    monkeypatch.setattr(probe._Arm, 'selected_model', no_memory)
    result = probe.run_frontier(recipe(), data, meter, store, backend_identity='demo', plan=plan())
    assert result['status'] == 'SUCCEEDED'
    assert refs == ['model/A-best.pt', 'model/B-best.pt', 'model/C-best.pt']
    assert refs == [result['summary']['arms'][name]['selected']['ref'] for name in 'ABC']


@pytest.mark.parametrize('arm', ['A', 'B', 'C'])
def test_failed_replacement_does_not_commit_selection_or_finalize(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, arm: str) -> None:
    probe, data, store, meter = prepare(tmp_path, monkeypatch)
    evaluate = probe.evaluate_batches
    write = store.write_model
    chosen: list[Any] = []
    choose = probe._Arm.choose
    finalized: list[object] = []

    def dev(*args: Any, **kwargs: Any) -> Any:
        value = evaluate(*args, **kwargs)
        value['mean_bce'] = .8 if not chosen or chosen[-1].best.epoch != 1 else .4
        return value

    def capture(owner: Any, *args: Any) -> None:
        if owner.spec.arm_id == arm:
            chosen.append(owner)
        choose(owner, *args)

    def fail(name: str, raw: bytes, *, replace: bool = False) -> str:
        if name == arm+'-best' and replace:
            raise ArtifactError('WRITE_FAILED')
        return cast(str, write(name, raw, replace=replace))

    monkeypatch.setattr(probe, 'evaluate_batches', dev)
    monkeypatch.setattr(probe._Arm, 'choose', capture)
    monkeypatch.setattr(store, 'write_model', fail)
    monkeypatch.setattr(store, 'finalize', lambda value: finalized.append(value))
    with pytest.raises(ArtifactError, match='WRITE_FAILED'):
        probe.run_frontier(recipe(), data, meter, store, backend_identity='demo', plan=plan())
    owner = chosen[-1]
    assert owner.best.epoch == owner.selection.epoch == owner.evidence['selected']['epoch'] == 1
    assert not data.released and not finalized


@pytest.mark.parametrize('rounds,peak', [(4, 22874080), (5, 32550764)])
def test_output_bound_fixed_scientific_recipe(rounds: int, peak: int) -> None:
    from cipher_lab.training_config import MLPSpec, TrainSpec
    from cipher_lab_plugins.frontier.training import output_bound
    p = plan(rounds)
    r = replace(recipe(), identity=p.identity, rounds=rounds, model=MLPSpec(128),
        training=TrainSpec(16384, 4096, 16384, 256, 32, .001, 31,
                           196905+rounds, 296905+rounds))
    bound = output_bound(r, p)
    assert bound['peak_bytes'] == peak < 128*1024*1024
    assert bound['maximum_temporary_bytes'] > len(__import__(
        'cipher_lab_plugins.frontier.simon_frontier', fromlist=['encode_plan']).encode_plan(p))


def test_output_bound_rejects_before_panels_or_initialization(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab.training_config import MLPSpec
    probe, data, store, meter = prepare(tmp_path, monkeypatch)
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('over-budget run must not initialize or touch panels')
    monkeypatch.setattr(data, 'inputs', forbidden)
    monkeypatch.setattr(probe, 'initialized_arm', forbidden)
    with pytest.raises(ValueError, match='ENGINEERING_LIMIT_OUTPUT'):
        probe.run_frontier(replace(recipe(), model=MLPSpec(4096)), data, meter,
                           store, backend_identity='demo', plan=plan())
    assert meter.snapshot()['attempted_optimizer_updates'] == 0


def test_output_bound_does_not_claim_unbounded_metadata_is_safe() -> None:
    from cipher_lab_plugins.frontier.training import output_bound
    r = recipe()
    with pytest.raises(ValueError, match='ENGINEERING_LIMIT_OUTPUT'):
        output_bound(replace(r, training=replace(r.training, epochs=100000)), plan())


def test_model_plan_tamper_stops_before_test(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    probe, data, store, meter = prepare(tmp_path, monkeypatch)
    step = probe.train_step
    p = plan()
    def mutate(*args: Any) -> float:
        value = step(*args)
        object.__setattr__(p, 'features', ((99999,),))
        return cast(float, value)
    monkeypatch.setattr(probe, 'train_step', mutate)
    result = probe.run_frontier(recipe(), data, meter, store, backend_identity='demo', plan=p)
    assert result['status'] == 'STOPPED'
    assert result['summary']['reason'] == 'IDENTITY_CHANGED'
    assert not data.released


def test_actual_owner_replace_failure_cannot_commit_selection(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    probe, data, store, meter = prepare(tmp_path, monkeypatch)
    evaluate = probe.evaluate_batches
    losses = iter([.8, .4])
    selected: list[Any] = []
    choose = probe._Arm.choose
    atomic = store._atomic
    model_writes = 0
    finalized: list[object] = []
    def dev(*args: Any, **kwargs: Any) -> Any:
        value = evaluate(*args, **kwargs)
        value['mean_bce'] = next(losses)
        return value
    def capture(owner: Any, *args: Any) -> None:
        selected.append(owner)
        choose(owner, *args)
    def fail(target: Path, raw: bytes) -> None:
        nonlocal model_writes
        if target.name == 'A-best.pt':
            model_writes += 1
            if model_writes == 2:
                raise OSError('synthetic atomic write fault')
        atomic(target, raw)
    monkeypatch.setattr(probe, 'evaluate_batches', dev)
    monkeypatch.setattr(probe._Arm, 'choose', capture)
    monkeypatch.setattr(store, '_atomic', fail)
    monkeypatch.setattr(store, 'finalize', lambda value: finalized.append(value))
    with pytest.raises(ArtifactError, match='WRITE_FAILED'):
        probe.run_frontier(recipe(), data, meter, store, backend_identity='demo', plan=plan())
    owner = selected[-1]
    assert owner.selection.epoch == owner.best.epoch == 1
    assert owner.evidence['selected']['sha256'] == __import__('hashlib').sha256(
        (tmp_path/'models/probe/A-best.pt').read_bytes()).hexdigest()
    assert store._sealed and not finalized and not data.released
    assert meter.snapshot()['completed_optimizer_updates'] == 2  # artificial ledger, not refunded

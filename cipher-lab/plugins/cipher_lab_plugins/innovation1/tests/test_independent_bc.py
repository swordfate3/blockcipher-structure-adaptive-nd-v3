"""Independent B/C admission and one-fit lifecycle, exclusively on guarded fixtures."""
from __future__ import annotations

import builtins
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path, PureWindowsPath
from typing import Any
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from cipher_lab_plugins.innovation1 import application, cli, resources, source
from cipher_lab_plugins.innovation1.checkpoint import encode
from cipher_lab_plugins.innovation1.config import TASKS, Task, load_config
from cipher_lab_plugins.innovation1.four_single import config as old
from cipher_lab_plugins.innovation1.four_single import replay
from cipher_lab_plugins.innovation1.four_single.models import build
from cipher_lab_plugins.innovation1.layout_probe import reuse
from cipher_lab_plugins.innovation1.layout_probe.reuse import PoolSourceError
from cipher_lab_plugins.innovation1.models import TorchRuntime
from cipher_lab_plugins.innovation1.resources import Resources
from cipher_lab_plugins.innovation1.serial_bc import config, data, models, training
from cipher_lab_plugins.innovation1.single_train.config import learning_rate
from cipher_lab_plugins.innovation1.single_train.data import PairBatch
from cipher_lab_plugins.innovation1.tests.test_four_single_replay import GPU, SOURCE, _archive
from cipher_lab_plugins.innovation1.tests.test_serial_bc import _tiny
from plugin_isolation import forbidden
from torch import nn

from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.execution import Clocks, Meter, Stopped

ROOT = r'G:\lxy\shared-m-cipher-nd-runs\20261124-independent-bc-30epoch-01'
NOW = datetime(2026, 11, 24, 11, tzinfo=UTC)
DEADLINE = '2026-11-24T11:01:00Z'


def _path(task: Task, arm: str) -> Path:
    return Path(f'configs/innovation1-independent-{task.cipher.lower()}-{arm}-30epoch.json')


def _arms(task: Task) -> tuple[config.SerialRecipe, config.SerialRecipe]:
    b, c = (load_config(_path(task, arm)) for arm in ('b', 'c'))
    assert isinstance(b, config.IndependentRecipe) and isinstance(c, config.IndependentRecipe)
    return b, c


def test_exact_matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                      capsys: pytest.CaptureFixture[str]) -> None:
    matrix = config.INDEPENDENT_RECIPES
    assert len(matrix) == 8 and len({r.run_id for r in matrix}) == 8
    assert {p.name for p in Path('configs').glob('innovation1-independent-*.json')} == {
        _path(task, arm).name for task in TASKS for arm in ('b', 'c')}
    for i, task in enumerate(TASKS):
        b, c = _arms(task)
        assert matrix[2*i:2*i+2] == (b, c)
        assert replace(b, routing_mode=c.routing_mode, run_id=c.run_id) == c
        b_dict, c_dict = b.to_dict(), c.to_dict()
        assert {k for k in b_dict if b_dict[k] != c_dict[k]} == {'run_id', 'routing'}
        for recipe, arm, mode in ((b, 'b', 'public-card'), (c, 'c', 'constant-no-card')):
            assert recipe.run_id == f'{_path(task, arm).stem}-20261124-01'
            assert recipe.tasks == (task,) and recipe.stage_slots == (old.SLOT,)
            assert recipe.archive_reader_recipe() == config.SERIAL_B.archive_reader_recipe()
            assert recipe.namespace == config.SERIAL_B.namespace
            assert recipe.streams.seeds == config.SERIAL_B.streams.seeds
            assert recipe.routing_mode == mode and recipe.to_dict()['output_root'] == ROOT
            tr = recipe.training
            assert (tr.epochs, tr.batches_per_epoch, tr.batch_groups) == (30, 180, 5000)
            assert (tr.dev_groups, tr.test_groups, recipe.resources.wall_seconds) == (
                100_000, 262_144, 5400)
            n = config.estimate(recipe)
            assert tuple(n[k] for k in ('fits', 'retries', 'stages', 'optimizer_updates',
                'train_rows', 'dev_rows', 'test_rows', 'forward_rows', 'real_blocks',
                'null_words')) == (1, 0, 1, 5400, 27_000_000, 100_000, 262_144, 27_362_144, 0, 0)
            assert n['artifact_peak_bound_bytes'] <= recipe.resources.artifact_bytes
            assert [learning_rate(recipe, cycle) for cycle in (0, 5399)] == [.001, .0001]
            config.official(recipe, recipe.run_id)
            config.validate_root(PureWindowsPath(ROOT), expected=ROOT)
            with pytest.raises(ValueError):
                config.validate_root(PureWindowsPath(config.ROOT), expected=ROOT)
    bad = tmp_path / 'bad.json'
    for key, value in (('method', config.METHOD), ('output_root', config.ROOT),
                       ('namespace', 'changed'), ('run_id', matrix[1].run_id),
                       ('tasks', matrix[-1].to_dict()['tasks']), ('unknown', None)):
        bad.write_text(json.dumps({**matrix[0].to_dict(), key: value}))
        with pytest.raises(ValueError, match='CONFIG_EXACT'):
            load_config(bad)
    with pytest.raises(ValueError, match='CONFIG_EXACT'):
        config.official(replace(matrix[0], tasks=matrix[-1].tasks), matrix[0].run_id)
    # Catch direct scientific imports even when pytest has loaded those modules.
    # Fresh-process import isolation is also checked by the standalone CLI export.
    real_import = builtins.__import__

    def static_import(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name.split('.')[0] not in ('torch', 'numpy')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', static_import)
    monkeypatch.setattr(application, 'make_resources', forbidden)
    monkeypatch.setattr(source, 'no_reparse', forbidden)
    for i, task in enumerate(TASKS):
        for j, arm in enumerate(('b', 'c')):
            assert cli.main(['check', '--config', str(_path(task, arm))]) == 0
            assert json.loads(capsys.readouterr().out)['recipe'] == matrix[2*i+j].to_dict()


def test_deadline_required_before_claim(monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    b, _ = _arms(TASKS[0])
    current = NOW
    clocks = Clocks(wall=lambda: 0., cpu=lambda: 0., utc=lambda: current)
    monkeypatch.setattr(application, 'Clocks', lambda: clocks)
    monkeypatch.setattr(resources, 'production_rss_bytes', lambda: 0)
    # Windows-native owner validation is proven with PureWindowsPath above.
    monkeypatch.setattr(cli, 'validate_serial_root', lambda *a, **kw: None)
    monkeypatch.setattr(source, 'no_reparse', forbidden)
    monkeypatch.setattr(RunStore, 'claim', forbidden)
    for value in (None, '', 'broken', '2026-11-24T11:01:00', '2026-11-25T11:01:00+01:00',
                  NOW.isoformat(), (NOW - timedelta(seconds=1)).isoformat()):
        if value is None:
            monkeypatch.delenv(config.DEADLINE_ENV, raising=False)
        else:
            monkeypatch.setenv(config.DEADLINE_ENV, value)
        assert cli.main(['run', '--config', str(_path(TASKS[0], 'b')), '--root', ROOT,
                         '--id', b.run_id, '--gpu-uuid', GPU]) == 3
        assert json.loads(capsys.readouterr().out)['reason'] == 'DEADLINE'
    monkeypatch.setenv(config.DEADLINE_ENV, DEADLINE)
    r = application.make_resources(b)
    assert r.meter._policy.deadline_utc == DEADLINE
    r.check()
    current += timedelta(seconds=60)
    with pytest.raises(Stopped, match='DEADLINE'):
        r.check()
    current = NOW
    monkeypatch.setenv(config.DEADLINE_ENV, '2026-11-24T14:00:00Z')
    assert application.make_resources(b).meter._policy.deadline_utc == (
        NOW + timedelta(seconds=5400)).isoformat()
    # Expire the deadline during pre-claim path checks, not just at policy creation.
    def expire(path: Path) -> Path:
        nonlocal current
        current = NOW + timedelta(seconds=60)
        return path

    monkeypatch.setattr(source, 'no_reparse', expire)
    monkeypatch.setenv(config.DEADLINE_ENV, DEADLINE)
    args = ['run', '--config', str(_path(TASKS[0], 'b')), '--root', ROOT,
            '--id', b.run_id, '--gpu-uuid', GPU]
    assert cli.main(args) == 3
    assert json.loads(capsys.readouterr().out)['reason'] == 'DEADLINE'
    # A fresh valid deadline reaches the existing claim boundary with one slot.
    current = NOW
    root = Path('/artificial-owner')
    monkeypatch.setattr(source, 'no_reparse', lambda _: root)
    claims: list[tuple[Path, str, tuple[str, ...]]] = []

    def claim(path: Path, run_id: str, *, model_slots: tuple[str, ...], **kw: Any) -> object:
        claims.append((path, run_id, model_slots))
        return object()

    monkeypatch.setattr(RunStore, 'claim', claim)
    monkeypatch.setattr(application, 'execute', lambda *a, **kw: None)
    monkeypatch.setattr(cli, '_finish', lambda *a, **kw: 0)
    assert cli.main(args) == 0 and claims == [(root, b.run_id, (old.SLOT,))]


def test_reads_only_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive, original, _ = _archive(tmp_path, 'early')
    legacy_recipe = _tiny(config.SERIAL_B, archive)
    detached = replace(archive, archive=replace(archive.archive, run_id='different-archive'))
    with pytest.raises(ValueError, match='ARCHIVE_RECIPE_IDENTITY'):
        data.SerialPools(_tiny(_arms(TASKS[0])[0], archive), forbidden, archive=detached)
    legacy = data.SerialPools(legacy_recipe, lambda: None, archive=archive)
    dev, bindings = legacy.bind(lambda _: None)
    prior = {t: [legacy.train(t, i).digest for i in range(18)] for t in TASKS}
    pin = reuse._pinned
    seen: list[Path] = []

    def track(path: Path, *args: Any) -> Any:
        seen.append(path)
        return pin(path, *args)

    with monkeypatch.context() as m:
        m.setattr(replay, '_pinned', track)
        m.setattr(reuse, '_pinned', track)
        for target in (TASKS[0], TASKS[-1]):
            name = target.cipher
            for base in _arms(target):
                seen.clear()
                visits: list[str] = []
                pools = data.SerialPools(_tiny(base, archive), lambda: None, archive=archive)
                selected, evidence = pools.bind(visits.append)
                assert evidence == {name: bindings[name]}  # includes split index SHA
                assert set(selected) == {name} and selected[name].digest == dev[name].digest
                with pytest.raises(ValueError, match='POOL_TASK'):
                    pools.train(replace(target, rounds=target.rounds + 1), 0)
                assert [pools.train(target, i).digest for i in range(18)] == prior[target]
                assert visits == [name]
                assert [(p.parent.name, p.name) for p in seen] == (
                    [('archive-four', 'run.json')]
                    + ([(archive.skinny_source.run_id, 'run.json')]
                       if name == 'SKINNY' and archive.skinny_source else [])
                    + [('dataset', f'{split}-{name}.npz') for split in ('train', 'dev')])
                with pytest.raises(ValueError, match='TEST_ORDER'):
                    pools.load_tests(forbidden)
                pools.release_train()
                with pytest.raises(ValueError, match='TEST_ORDER'):
                    pools.load_tests(forbidden, only=TASKS[1])
                tests = pools.load_tests(visits.append)
                assert visits == [name, name] and set(tests) == {name}
                assert seen[-1].name == f'test-{name}.npz'
                assert tests[name][1]['source_record']['record']['sha256'] == (
                    archive.archive.record.sha256)
    assert all(path.read_bytes() == raw for path, raw in original.items())
    # Losing any unrelated on-disk panel cannot cause a read or fallback. Losing
    # the selected panel must stop with the archive owner's fixed failure type.
    base = _arms(TASKS[0])[0]
    for name in ('dev-SIMON.npz', 'test-SKINNY.npz', 'train-SIMECK.npz'):
        victim = next(p for p in original if p.name == name)
        victim.unlink()
        pools = data.SerialPools(_tiny(base, archive), lambda: None, archive=archive)
        if name == 'train-SIMECK.npz':
            with pytest.raises(PoolSourceError):
                pools.bind(lambda _: None)
        else:
            pools.bind(lambda _: None)
            pools.release_train()
            assert set(pools.load_tests(lambda _: None)) == {'SIMECK'}


@pytest.mark.parametrize('case',
                         ['ok-b', 'ok-c', 'strict', 'disk', 'deadline', 'test-missing', 'atomic'])
def test_fit_one_from_scratch(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive, _, _ = _archive(tmp_path / 'prior', 'early')
    task = TASKS[-1] if case == 'ok-c' else TASKS[0]
    name = task.cipher
    if case == 'test-missing':
        (Path(archive.archive.root) / 'data' / archive.archive.run_id / 'dataset'
         / f'test-{name}.npz').unlink()
    recipe = _tiny(_arms(task)[int(case == 'ok-c')], archive)
    # Keep the real epoch count; only the fixture's per-epoch split geometry scales.
    recipe = replace(recipe, training=replace(recipe.training, epochs=30))
    clock_now = NOW
    clocks = Clocks(wall=lambda: 0., cpu=lambda: 0., utc=lambda: clock_now)
    res = Resources(Meter(config.make_policy(recipe, clocks, DEADLINE), clocks, lambda: 0))
    count = recipe.training.epochs * recipe.training.batches_per_epoch
    root = tmp_path / 'out'
    run_id = recipe.run_id
    store = RunStore.claim(root, run_id, output_bytes=recipe.resources.artifact_bytes,
                           needs_models=True, model_slots=recipe.stage_slots,
                           source_supplier=lambda: dict(SOURCE))
    store.write_snapshot(dict(recipe=recipe.to_dict()))
    events: list[str] = []
    updated: list[int] = []
    initial: list[nn.Module] = []
    adams: list[object] = []
    identities: list[dict[str, Any]] = []
    strict_restore = models.SerialRuntime.restore
    build_for_runtime = models.SerialRuntime._build
    path_open, zip_open = Path.open, ZipFile.open
    run_path = root / 'data' / run_id / 'run.json'

    def read_path(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.is_relative_to(archive.archive.root):
            assert path.suffix != '.pt'
            if path.name.startswith('test-'):
                assert events[-1] == 'strict'
                events.append(f'open:{path.name}')
        return path_open(path, *args, **kwargs)

    def read_member(z: ZipFile, member: Any, mode: Any = 'r', **kw: Any) -> Any:
        if mode == 'r' and getattr(member, 'filename', member) == 'logits.npy':
            forbidden()
        return zip_open(z, member, mode, **kw)

    def build_model(self: models.SerialRuntime) -> nn.Module:
        result = build_for_runtime(self)
        initial.append(result)
        return result

    def fake_runtime(self: Any, recipe: Any, device: str, *, inference_only: bool) -> None:
        assert device == 'cuda:0' and inference_only
        self.recipe, self.device = recipe, torch.device('cpu')
        self.models, self.optimizers = {}, {}

    def adam(parameters: Any, **kwargs: Any) -> object:
        assert list(parameters) and kwargs['lr'] == .001
        optimizer = object()
        adams.append(optimizer)
        return optimizer

    def update(self: models.SerialRuntime, slot: str, target: Task, x: PairBatch,
               y: Any, lr: float, meter: Resources) -> float:
        nonlocal clock_now
        assert target == task and self.optimizers[slot] is adams[0]
        assert x.index == len(updated) and lr == learning_rate(recipe, x.index)
        assert not events
        updated.append(x.index)
        meter.reserve('forward_rows', x.groups)
        meter.complete('forward_rows', x.groups)
        if case == 'deadline' and x.index == 4:
            clock_now += timedelta(seconds=60)
        # Store a deterministic marker to prove the reload used the last update.
        with torch.no_grad():
            next(self.models[slot].parameters()).fill_(x.index)
        return .5

    def predict(self: models.SerialRuntime, slot: str, target: Task, x: PairBatch) -> Any:
        assert len(updated) == count and target == task
        assert (not self.optimizers) == (x.split == 'test')
        assert next(self.models[slot].parameters()).flatten()[0].item() == count - 1
        events.append(x.split)
        return np.full(x.labels.shape, .1 if x.split == 'dev' else .2)

    def restore(self: models.SerialRuntime, slot: str, raw: bytes,
                identity: dict[str, Any]) -> None:
        assert events == ['dev'] and self.models == self.optimizers == {}
        assert identity['epoch'] == 30 and identity['stage'] == 1
        assert identity['recipe'] == recipe.to_dict() and identity['routing_mode'] == (
            recipe.routing_mode)
        assert identity['stage_task'] == name and identity['source'] == SOURCE
        assert set(identity['pools']) == {name} and identity['pools'][name]['train']['index_sha256']
        identities.append(identity)
        assert hashlib.sha256(raw).hexdigest() == json.loads(run_path.read_bytes())['artifacts'][
            f'model/{old.SLOT}.pt']['sha256']
        if case == 'strict':
            wrong = {**identity, 'stage_task': TASKS[1].cipher}
            with pytest.raises(ValueError, match='CHECKPOINT_IDENTITY'):
                strict_restore(self, slot, raw, wrong)
            # Same tensor shapes and valid archive, but wrong bound data identity:
            # failure must come from strict reload, not store hash or ZIP preflight.
            raw = encode({**identity, 'pools': {}}, initial[0].state_dict())
        strict_restore(self, slot, raw, identity)
        events.append('strict')

    read_model, finalize = store.read_model, store.finalize

    def disk_read(ref: str, digest: str, limit: int) -> bytes:
        if case == 'disk':
            (root / 'models' / run_id / f'{old.SLOT}.pt').write_bytes(b'damaged on disk')
        return read_model(ref, digest, limit)

    def final_record(state: dict[str, Any]) -> None:
        if case == 'atomic' and state['status'] == 'SUCCEEDED':
            assert json.loads(run_path.read_bytes())['status'] == 'RUNNING'
            assert json.loads(run_path.read_bytes())['summary']['test'] == {}
        finalize(state)

    monkeypatch.setattr(TorchRuntime, '__init__', fake_runtime)
    monkeypatch.setattr(torch.optim, 'Adam', adam)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(models.SerialRuntime, '_build', build_model)
    monkeypatch.setattr(models.SerialRuntime, 'prepare', lambda self, b: (b, b.labels))
    monkeypatch.setattr(models.SerialRuntime, 'routes', lambda self: {name: {'local': 1.}})
    monkeypatch.setattr(models.SerialRuntime, 'update', update)
    monkeypatch.setattr(models.SerialRuntime, 'predict', predict)
    monkeypatch.setattr(models.SerialRuntime, 'restore', restore)
    monkeypatch.setattr(store, 'read_model', disk_read)
    monkeypatch.setattr(store, 'finalize', final_record)
    pools = data.SerialPools(recipe, res.check, archive=archive)
    with monkeypatch.context() as m:
        m.setattr(Path, 'open', read_path)
        m.setattr(ZipFile, 'open', read_member)
        training._execute(recipe, store, res, launch={}, pools=pools)
    assert store._sealed and len(adams) == 1
    assert store._frozen_sha == hashlib.sha256(run_path.read_bytes()).hexdigest()
    saved: dict[str, Any] = json.loads(run_path.read_bytes())
    fitted = case in ('ok-b', 'ok-c', 'atomic')
    assert saved['status'] == ('SUCCEEDED' if fitted else
                               'STOPPED' if case == 'deadline' else 'FAILED')
    assert saved['counts']['completed_real_blocks'] == 0
    assert saved['counts']['completed_optimizer_updates'] == (5 if case == 'deadline' else count)
    assert saved['counts']['completed_forward_rows'] == (
        20 if case == 'deadline' else 4 * count + 4 + 4 * fitted)
    assert sorted(k for k in saved['artifacts'] if k.startswith('model/')) == (
        [] if case == 'deadline' else [f'model/{old.SLOT}.pt'])
    if fitted:
        assert events == ['dev', 'strict', f'open:test-{name}.npz', 'test']
        assert len(initial) == 2 and initial[0] is not initial[1]
        assert set(saved['summary']['test']) == {name}
        assert [s['stage'] for s in saved['summary']['stages']] == [1]
        assert set(saved['summary']['stages'][0]['dev']) == {name}
        assert read_saved(root, run_id)['status'] == 'SUCCEEDED'
        assert cli._finish(store, root, run_id, resources=res) == 0
        report = (root / 'result' / run_id / 'report.md').read_text(encoding='utf-8')
        assert '共用一个 M 与 Adam' not in report and '不检验' in report
    else:
        assert not any(e.startswith('open:') for e in events)
        assert not saved['summary'].get('test')
        if case == 'deadline':
            assert not events and saved['summary']['reason'] == 'DEADLINE'
        elif case == 'strict':
            assert len(identities) == 1 and events == ['dev']
        elif case == 'test-missing':
            assert events == ['dev', 'strict']
            assert saved['summary']['reason'] == 'TEST_SOURCE_INVALID'


def test_paired_initialization_identity() -> None:
    # Vary the ambient CPU RNG to catch any dependence on invocation order or
    # the process default seed; don't manually seed CUDA or sample private entropy.
    with torch.random.fork_rng(devices=[]):
        reference = build(config.SERIAL_B).state_dict()
        for task in TASKS:
            left, right = _arms(task)
            built = []
            for ambient, recipe in enumerate((left, right)):
                torch.default_generator.manual_seed(ambient)
                rng = torch.random.get_rng_state()
                fit = models.SerialRuntime.__new__(models.SerialRuntime)
                fit.contract = recipe
                built.append(fit._build())
                assert torch.equal(rng, torch.random.get_rng_state())
            assert isinstance(built[1], models.NoCardM)
            assert not isinstance(built[0], models.NoCardM)
            for model in built:
                torch.testing.assert_close(model.state_dict(), reference, rtol=0, atol=0)
            for p, q in zip(built[0].parameters(), built[1].parameters(), strict=True):
                assert p is not q and p.data_ptr() != q.data_ptr()

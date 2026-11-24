"""The exact TRAIN-volume pair and owner lifecycle on artificial inputs only."""
from __future__ import annotations

import builtins
import json
import struct
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from typing import Any, cast
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from cipher_lab_plugins.innovation1 import application, cli, resources, source
from cipher_lab_plugins.innovation1.config import Task, load_config
from cipher_lab_plugins.innovation1.data import derive_seed
from cipher_lab_plugins.innovation1.four_single import replay
from cipher_lab_plugins.innovation1.layout_probe import reuse
from cipher_lab_plugins.innovation1.resources import Resources
from cipher_lab_plugins.innovation1.serial_bc import config, data, models, training
from cipher_lab_plugins.innovation1.single_train.config import learning_rate
from cipher_lab_plugins.innovation1.single_train.data import FixedPairReplay, PairBatch, Provider
from cipher_lab_plugins.innovation1.tests.test_four_single_replay import SOURCE
from cipher_lab_plugins.innovation1.tests.test_innovation1_fixed_bce import _fixture
from plugin_isolation import forbidden

from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.execution import Clocks, Meter, Stopped

ROOT = r'G:\lxy\shared-m-cipher-nd-runs\20261124-skinny-train-volume-01'
GPU = 'GPU-171dfbbe-c1a7-d93b-549d-8824cb8a18c7'
NOW = datetime(2026, 11, 24, 11, tzinfo=UTC)
DEADLINE = '2026-11-24T11:01:00Z'
PRIOR = Path('configs/innovation1-independent-skinny-b-30epoch.json')
PATHS = tuple(Path(f'configs/innovation1-skinny-train-volume-{arm}.json')
              for arm in ('small', 'large'))


def test_static_exact_and_full_cosine_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    prior = load_config(PRIOR)
    assert isinstance(prior, config.IndependentRecipe)
    trace = b''.join(struct.pack('<d', learning_rate(prior, i)) for i in range(5400))
    assert len(config.VOLUME_RECIPES) == 2
    real_import = builtins.__import__

    def stdlib_check(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name.split('.')[0] not in ('numpy', 'torch')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', stdlib_check)
    monkeypatch.setattr(application, 'make_resources', forbidden)
    monkeypatch.setattr(source, 'no_reparse', forbidden)
    monkeypatch.setattr(RunStore, 'claim', forbidden)
    monkeypatch.setenv(config.DEADLINE_ENV, 'ignored by static check')
    for path, exact, (epochs, per_epoch) in zip(PATHS, config.VOLUME_RECIPES,
                                                ((30, 180), (3, 1800)), strict=True):
        recipe = load_config(path)
        assert type(recipe) is config.VolumeRecipe and recipe == exact
        config.official(recipe, recipe.run_id)
        value = recipe.to_dict()
        assert recipe.run_id == f'{path.stem}-20261124-01'
        assert recipe.method == 'innovation1-skinny-train-volume-v1'
        assert recipe.streams == prior.streams and recipe.stage_slots == ()
        assert recipe.routing_mode == prior.routing_mode == 'public-card'
        assert recipe.archive_reader_recipe() == prior.archive_reader_recipe()
        for key in ('model', 'optimizer', 'seeds', 'tasks', 'device', 'routing'):
            assert value[key] == prior.to_dict()[key]
        tr = recipe.training
        assert (tr.epochs, tr.batches_per_epoch, tr.batch_groups, tr.dev_groups,
                tr.test_groups) == (epochs, per_epoch, 5000, 100_000, 0)
        assert b''.join(struct.pack('<d', learning_rate(recipe, i)) for i in range(5400)) == trace
        assert recipe.dev_updates == (1800, 3600, 5400)
        assert value['output_root'] == ROOT and 'archive' not in value['data']
        assert recipe.prior_record_sha256 == (
            '3843f9a08b9b40c3b0e1a5ad2f66f316b549229d9845d1b93044a1eb852193e0')
        assert recipe.small_train.digest == (
            '471c7276d81be465dc45db88f6922884a08133cd29d51053501764b15645ff8e')
        assert recipe.small_train.index_sha256 == (
            '5c8137d9db18f90d38910e81041062b7b5d8e6c43196a947b2d59a9a5752a52f')
        assert recipe.common_dev.digest == (
            '22d91a93a58e381788c49856585062e3d5a59b7f763bda53bd627b1935fc7bf2')
        assert recipe.common_dev.index_sha256 == (
            '6c0e924d75810df8dc91c5bf7c0737f107bd62f11dfb56138710fdb5cc3d1c19')
        assert recipe.source_train_digest == (
            'f2fa2651f899a04ee079458fa026cedaff75c1f1f979afc12362d9065603c39d')
        assert cli.main(['check', '--config', str(path)]) == 0
        saved = json.loads(capsys.readouterr().out)
        assert saved['scope'] == 'STATIC_ONLY_NO_DEVICE_ADMISSION'
        assert saved['recipe'] == value
        totals = saved['estimate']
        assert [totals[k] for k in ('fits', 'retries', 'optimizer_updates', 'train_rows',
            'dev_rows', 'test_rows', 'forward_rows', 'real_blocks', 'null_words',
            'fixed_train_rows_each', 'fixed_dev_rows_each', 'dev_tail_rows')] == [
                1, 0, 5400, 27_000_000, 300_000, 0, 27_300_000, 0, 0,
                5000 * per_epoch, 100_000, 160]
        bad = tmp_path / 'bad.json'
        for key, new in (('output_root', config.INDEPENDENT_ROOT), ('unknown', None),
                        ('training', {**value['training'], 'test_groups': False}),
                        ('data', {**value['data'], 'source_train_digest': '0' * 64})):
            bad.write_text(json.dumps({**value, key: new}))
            with pytest.raises(ValueError, match='CONFIG_EXACT'):
                load_config(bad)


def test_cpu_initial_identity_without_forward() -> None:
    prior = load_config(PRIOR)
    assert isinstance(prior, config.IndependentRecipe)
    fits = []
    with torch.random.fork_rng(devices=[]):
        for seed, recipe in enumerate((prior, *config.VOLUME_RECIPES)):
            torch.default_generator.manual_seed(seed)
            rng = torch.random.get_rng_state()
            runtime = models.SerialRuntime.__new__(models.SerialRuntime)
            runtime.contract = recipe
            fits.append(runtime._build())
            assert torch.equal(rng, torch.random.get_rng_state())
    assert all(not isinstance(m, models.NoCardM) for m in fits)
    for fit in fits[1:]:
        torch.testing.assert_close(fits[0].state_dict(), fit.state_dict(), rtol=0, atol=0)
        assert all(a.data_ptr() != b.data_ptr()
                   for a, b in zip(fits[0].parameters(), fit.parameters(), strict=True))


def test_admission_before_owner(monkeypatch: pytest.MonkeyPatch,
                                capsys: pytest.CaptureFixture[str]) -> None:
    exact, other = config.VOLUME_RECIPES
    config.validate_root(PureWindowsPath(ROOT), expected=ROOT)
    with pytest.raises(ValueError):
        config.validate_root(PureWindowsPath(config.INDEPENDENT_ROOT), expected=ROOT)
    now = NOW
    clocks = Clocks(wall=lambda: 0., cpu=lambda: 0., utc=lambda: now)
    monkeypatch.setattr(application, 'Clocks', lambda: clocks)
    monkeypatch.setattr(resources, 'production_rss_bytes', lambda: 0)
    monkeypatch.setattr(source, 'no_reparse', forbidden)
    monkeypatch.setattr(RunStore, 'claim', forbidden)

    def invoke(*, root: str = ROOT, run_id: str = exact.run_id, gpu: str = GPU) -> int:
        return cli.main(['run', '--config', str(PATHS[0]), '--root', root,
                         '--id', run_id, '--gpu-uuid', gpu])

    assert invoke(root=config.INDEPENDENT_ROOT) == 2
    # Exercise native Windows root semantics without claiming on a Windows host.
    validate = config.validate_root
    monkeypatch.setattr(cli, 'validate_serial_root', lambda p, **kw: validate(
        PureWindowsPath(str(p)), **kw))
    assert invoke(root=config.INDEPENDENT_ROOT) == 2
    assert invoke(run_id=other.run_id) == 2
    assert invoke(gpu='GPU-00000000-0000-0000-0000-000000000000') == 2
    for cutoff in (None, '', 'broken', NOW.isoformat(), '2026-11-24T11:01:00+01:00'):
        if cutoff is None:
            monkeypatch.delenv(config.DEADLINE_ENV, raising=False)
        else:
            monkeypatch.setenv(config.DEADLINE_ENV, cutoff)
        assert invoke() == 3
        assert json.loads(capsys.readouterr().out)['reason'] == 'DEADLINE'
    monkeypatch.setenv(config.DEADLINE_ENV, '2026-11-24T14:00:00Z')
    assert application.make_resources(exact).meter._policy.deadline_utc == (
        NOW + timedelta(seconds=3600)).isoformat()
    monkeypatch.setenv(config.DEADLINE_ENV, DEADLINE)
    assert application.make_resources(other).meter._policy.deadline_utc == DEADLINE

    def expire(path: Path) -> Path:
        nonlocal now
        now = NOW + timedelta(seconds=60)
        return path

    monkeypatch.setattr(source, 'no_reparse', expire)
    assert invoke() == 3
    assert json.loads(capsys.readouterr().out)['reason'] == 'DEADLINE'
    now = NOW
    probes = []

    def no_reparse(path: Path) -> Path:
        probes.append(path)
        assert 'models' not in path.parts
        return path

    def claim(root: Path, run_id: str, *, needs_models: bool, model_slots: tuple[str, ...],
              **kwargs: Any) -> object:
        assert str(root) == ROOT and run_id == exact.run_id
        assert not needs_models and model_slots == ()
        return object()

    monkeypatch.setattr(source, 'no_reparse', no_reparse)
    monkeypatch.setattr(RunStore, 'claim', claim)
    monkeypatch.setattr(application, 'execute', lambda *a, **kw: None)
    monkeypatch.setattr(cli, '_finish', lambda *a, **kw: 0)
    assert invoke() == 0 and len(probes) == 3


def test_deadline_crossed_during_claim_source_scan(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / 'out'
    run_id = config.VOLUME_RECIPES[0].run_id
    now = NOW
    scans = 0
    writes: list[Path] = []
    dispatched: list[bool] = []
    clocks = Clocks(wall=lambda: 0., cpu=lambda: 0., utc=lambda: now)
    real_identity, real_mkdir, real_atomic = source.source_identity, Path.mkdir, RunStore._atomic

    def slow_identity() -> dict[str, str]:
        nonlocal now, scans
        assert now == NOW and not root.exists()
        identity = real_identity()
        scans += 1
        now = NOW + timedelta(seconds=60)
        return identity

    def mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
        writes.append(path)
        real_mkdir(path, *args, **kwargs)

    def atomic(store: RunStore, target: Path, content: bytes) -> None:
        writes.append(target)
        real_atomic(store, target, content)

    monkeypatch.setenv(config.DEADLINE_ENV, DEADLINE)
    monkeypatch.setattr(application, 'Clocks', lambda: clocks)
    monkeypatch.setattr(resources, 'production_rss_bytes', lambda: 0)
    monkeypatch.setattr(cli, 'validate_serial_root', lambda *a, **kw: None)
    monkeypatch.setattr(source, 'source_identity', slow_identity)
    monkeypatch.setattr(Path, 'mkdir', mkdir)
    monkeypatch.setattr(RunStore, '_atomic', atomic)
    monkeypatch.setattr(application, 'execute', lambda *a, **kw: dispatched.append(True))
    code = cli.main(['run', '--config', str(PATHS[0]), '--root', str(root),
                     '--id', run_id, '--gpu-uuid', GPU])
    assert scans == 1 and now > NOW
    assert not writes and not root.exists() and not dispatched
    assert code == 3
    result = json.loads(capsys.readouterr().out)
    assert (result['scientific_status'], result['command_status'], result['report_status'],
            result['reason']) == ('UNKNOWN', 'STOPPED', 'NOT_ATTEMPTED', 'DEADLINE')


def _tiny(tmp_path: Path, mp: pytest.MonkeyPatch, arm: int) -> tuple[
        config.VolumeRecipe, PairBatch, PairBatch, dict[Path, bytes]]:
    skinny, _, originals = _fixture(tmp_path / 'prior')
    tr, dv, _ = reuse.load_reused_pool(skinny, lambda: None)
    base = config.VOLUME_RECIPES[arm]
    fixture_reader = replace(base.archive_reader_recipe(), skinny_source=skinny.pool_source,
                                training=skinny.training)
    mp.setattr(config.VolumeRecipe, 'archive_reader_recipe', lambda _: fixture_reader)
    task = base.tasks[0]

    def pins(panel: PairBatch, n: int) -> config.SubsetPin:
        seed = derive_seed(base.streams, task, panel.split, 0, 'sample')
        subset, index = data.index_subset(panel, n, seed, base.namespace)
        return config.SubsetPin(subset.digest, sha256(index).hexdigest())

    recipe = replace(base, skinny_source=skinny.pool_source, dev_updates=(9, 18, 27),
                     source_train_digest=tr.digest, small_train=pins(tr, 12),
                     common_dev=pins(dv, 4),
                     training=replace(base.training, epochs=(9, 3)[arm],
                                      batches_per_epoch=(3, 9)[arm], batch_groups=4, dev_groups=4,
                                      eval_batch_groups=4))
    return recipe, tr, dv, originals


@pytest.mark.parametrize('arm', [0, 1])
def test_only_three_pinned_files_membership_and_old_shuffle(
        arm: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recipe, original, valid, originals = _tiny(tmp_path, monkeypatch, arm)
    task = recipe.tasks[0]
    seen: list[Path] = []
    path_open = Path.open
    zip_open = ZipFile.open
    subset = data.index_subset

    def opened(path: Path, *args: Any, **kwargs: Any) -> Any:
        assert path in originals
        seen.append(path)
        return path_open(path, *args, **kwargs)

    def member(z: ZipFile, item: Any, *args: Any, **kwargs: Any) -> Any:
        assert getattr(item, 'filename', item) in ('ciphertext.npy', 'labels.npy', 'metadata.npy')
        return zip_open(z, item, *args, **kwargs)

    def check_subset(panel: PairBatch, *args: Any) -> Any:
        assert not (arm and panel.split == 'train')
        return subset(panel, *args)

    monkeypatch.setattr(replay.ReplayPools, '__init__', forbidden)
    monkeypatch.setattr(Provider, '__init__', forbidden)
    monkeypatch.setattr(data, 'index_subset', check_subset)
    expected_dev, _ = subset(valid, 4, derive_seed(recipe.streams, task, 'dev', 0, 'sample'),
                             recipe.namespace)
    expected_train, indices = subset(original, 12,
        derive_seed(recipe.streams, task, 'train', 0, 'sample'), recipe.namespace)
    prior = FixedPairReplay(expected_train, recipe.training, recipe.streams, task,
                            recipe.namespace, lambda: None)
    pools = data.VolumePools(recipe, lambda: None)
    visited: list[str] = []
    with monkeypatch.context() as m:
        m.setattr(Path, 'open', opened)
        m.setattr(ZipFile, 'open', member)
        dev, bindings = pools.bind(visited.append)
        assert visited == ['SKINNY']
        assert not hasattr(pools, 'load_tests')
        assert [p.name for p in seen] == ['run.json', 'train-SKINNY.npz', 'dev-SKINNY.npz']
        assert dev[task.cipher].digest == expected_dev.digest == recipe.common_dev.digest
        evidence = bindings['SKINNY']
        assert evidence['prior_small']['record_sha256'] == recipe.prior_record_sha256
        assert evidence['train']['source_digest'] == original.digest
        assert evidence['train']['source_groups'] == 36
        assert evidence['dev']['index_sha256'] == recipe.common_dev.index_sha256
        with pytest.raises(ValueError, match='POOL_TASK'):
            pools.train(replace(task, rounds=task.rounds - 1), 0)
        size = recipe.training.batches_per_epoch
        source_rows = original.ciphertext if arm else original.ciphertext[indices]
        for epoch in range(recipe.training.epochs):
            cycles = range(epoch * size, (epoch + 1) * size)
            rows = [pools.train(task, i) for i in cycles]
            assert all(r.namespace == recipe.namespace for r in rows)
            assert sorted(b.tobytes() for r in rows for b in r.ciphertext) == sorted(
                b.tobytes() for b in source_rows)
            if not arm:
                assert [r.digest for r in rows] == [prior.train(i).digest for i in cycles]
        if not arm:
            assert evidence['train']['index_sha256'] == recipe.small_train.index_sha256
        else:
            assert evidence['train']['digest'] == original.digest
            assert evidence['train']['index_sha256'] is None
        pools.release_train()
        with pytest.raises(ValueError, match='POOL_NOT_READY'):
            pools.train(task, 0)
    assert all(path.read_bytes() == raw for path, raw in originals.items())


@pytest.mark.parametrize('case', ['ok-small', 'ok-large', 'interrupt', 'pre-source', 'setup-stop',
                                  'bad-pin', 'bad-source-digest', 'tamper', 'missing-payload',
                                  'summary-write-fault', 'summary-write-fault-tampered'])
def test_dev_only_owner(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                        capsys: pytest.CaptureFixture[str]) -> None:
    recipe, _, _, originals = _tiny(tmp_path, monkeypatch, int(case == 'ok-large'))
    if case == 'bad-pin':
        recipe = replace(recipe, common_dev=replace(recipe.common_dev, index_sha256='0' * 64))
    if case == 'bad-source-digest':
        recipe = replace(recipe, source_train_digest='0' * 64)
    if case == 'tamper':
        path = next(p for p in originals if p.name == 'train-SKINNY.npz')
        raw = originals[path]
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    clocks = Clocks(wall=lambda: 0., cpu=lambda: 0., utc=lambda: NOW)
    res = Resources(Meter(config.make_policy(recipe, clocks, DEADLINE), clocks, lambda: 0))
    root = tmp_path / 'out'
    run_id = recipe.run_id
    source_checks = 0

    def source_identity() -> dict[str, str]:
        nonlocal source_checks
        source_checks += 1
        return dict(SOURCE)

    store = RunStore.claim(root, run_id, output_bytes=recipe.resources.artifact_bytes,
                           source_supplier=source_identity)
    if case == 'pre-source':
        store.write_snapshot(dict(recipe=recipe.to_dict()))
    initial_source_checks = source_checks
    seen: list[Path] = []
    path_open = Path.open

    def opened(path: Path, *args: Any, **kwargs: Any) -> Any:
        assert path in originals or path.is_relative_to(root)
        assert path.suffix != '.pt' and not path.name.startswith('test-')
        if path in originals:
            seen.append(path)
        return path_open(path, *args, **kwargs)

    checks = 0
    real_check = res.check

    def interrupt_before_source() -> None:
        # Allow the assembly check, then stop inside the source gate before rehash.
        nonlocal checks
        checks += 1
        if checks == 2:
            raise KeyboardInterrupt
        real_check()

    updates: list[float] = []
    observations: list[int] = []

    def update(self: models.SerialRuntime, slot: str, task: Task, x: PairBatch, y: Any,
               lr: float, meter: Resources) -> float:
        assert task == recipe.tasks[0] and x.index == len(updates) and x.split == 'train'
        updates.append(lr)
        meter.reserve('forward_rows', x.groups)
        meter.complete('forward_rows', x.groups)
        return .5

    def predict(self: models.SerialRuntime, slot: str, task: Task, x: PairBatch) -> Any:
        assert task == recipe.tasks[0] and x.split == 'dev' and x.groups == 4
        observations.append(len(updates))
        if case == 'interrupt' and observations == [9, 18]:
            raise KeyboardInterrupt
        return np.zeros(x.labels.shape)

    monkeypatch.setattr(Path, 'open', opened)
    monkeypatch.setattr(replay.ReplayPools, '__init__', forbidden)
    monkeypatch.setattr(replay.ReplayPools, 'load_tests', forbidden)
    monkeypatch.setattr(application, '_production_provider', forbidden)
    # Bypass only the exact native admission for the scaled archive/owner fixture.
    monkeypatch.setattr(config, 'VOLUME_RECIPES', (recipe,))
    monkeypatch.setattr(application, 'validate_serial_root', lambda *a, **kw: None)

    def fake_device(exact: config.VolumeRecipe, gpu: str) -> None:
        assert exact is recipe and gpu == GPU and not seen
        if case == 'setup-stop':
            raise Stopped('DEADLINE')

    monkeypatch.setattr(application, 'prepare_device', fake_device)
    monkeypatch.setattr(Provider, '__init__', forbidden)
    monkeypatch.setattr(RunStore, 'write_model', forbidden)
    monkeypatch.setattr(RunStore, 'read_model', forbidden)
    monkeypatch.setattr(RunStore, 'write_dataset', forbidden)
    monkeypatch.setattr(models.SerialRuntime, 'checkpoint', forbidden)
    monkeypatch.setattr(models.SerialRuntime, 'restore', forbidden)
    monkeypatch.setattr(models.SerialRuntime, '__init__', lambda *a: None)
    monkeypatch.setattr(models.SerialRuntime, 'metadata', lambda _: {})
    monkeypatch.setattr(models.SerialRuntime, 'prepare', lambda _, b: (b, b.labels))
    monkeypatch.setattr(models.SerialRuntime, 'routes', lambda _: {'SKINNY': {'local': 1.}})
    monkeypatch.setattr(models.SerialRuntime, 'predict', predict)
    monkeypatch.setattr(models.SerialRuntime, 'update', update)
    dev_ref = 'payload/dev-9.json'
    run_path = root / 'data' / run_id / 'run.json'
    faulted = False
    real_atomic = store._atomic

    def fail_first_observation_snapshot(target: Path, raw: bytes) -> None:
        nonlocal faulted
        if target == run_path:
            proposed = json.loads(raw)
            if proposed['summary'].get('dev_observations') and not faulted:
                committed = json.loads(run_path.read_bytes())
                assert dev_ref in committed['artifacts']
                assert committed['summary']['dev_observations'] == []
                faulted = True
                raise OSError('artificial post-DEV summary snapshot fault')
        real_atomic(target, raw)

    if case.startswith('summary-write-fault'):
        monkeypatch.setattr(store, '_atomic', fail_first_observation_snapshot)
    if case == 'pre-source':
        with monkeypatch.context() as m:
            m.setattr(res, 'check', interrupt_before_source)
            training._execute(recipe, store, res, launch={})
    else:
        application.execute(recipe, store, res, gpu_uuid=GPU, run_id=run_id)
    assert store._sealed and not (root / 'models').exists()
    if case in ('pre-source', 'setup-stop'):
        assert source_checks == initial_source_checks and not seen
    elif case == 'tamper':
        assert [p.name for p in seen] == ['run.json', 'train-SKINNY.npz']
    else:
        assert [p.name for p in seen] == ['run.json', 'train-SKINNY.npz', 'dev-SKINNY.npz']
    saved = cast(dict[str, Any], read_saved(root, run_id))
    summary, counts = saved['summary'], saved['counts']
    assert 'test' not in summary and 'persisted_reload' not in summary
    assert counts['completed_real_blocks'] == counts['attempted_real_blocks'] == 0
    assert counts['completed_optimizer_updates'] == len(updates)
    if case.startswith('summary-write-fault'):
        assert faulted and saved['status'] == 'FAILED' and summary['reason'] == 'WRITE_FAILED'
        assert 'dev_observations' not in summary and summary['pending'] == []
        assert observations == [9] and len(updates) == 9
        assert counts['completed_forward_rows'] == 40
        assert [ref for ref in saved['artifacts'] if ref.startswith('payload/dev-')] == [dev_ref]
        payload_path = run_path.parent / dev_ref
        raw = payload_path.read_bytes()
        digest = sha256(raw).hexdigest()
        assert saved['artifacts'][dev_ref] == dict(kind='payload', bytes=len(raw), sha256=digest)
        frozen = run_path.read_bytes()
        assert saved['run_sha256'] == sha256(frozen).hexdigest() == store._frozen_sha
        # A plausible-looking next point on disk is not registered by the seal.
        orphan = {**saved['payloads'][dev_ref], 'updates': 18, 'epoch': 6}
        payload_path.with_name('dev-18.json').write_text(json.dumps(orphan), encoding='utf-8')
        tampered = case.endswith('tampered')
        if tampered:
            payload_path.write_bytes(raw.replace(b'"mean_online_train_bce":0.5',
                                                  b'"mean_online_train_bce":0.9'))
            assert payload_path.read_bytes() != raw and payload_path.stat().st_size == len(raw)
        monkeypatch.setattr(store, 'finalize', forbidden)
        monkeypatch.setattr(training, 'score', forbidden)
        assert cli._finish(store, root, run_id, resources=res) == (6 if tampered else 4)
        result = json.loads(capsys.readouterr().out)
        assert result['scientific_status'] == result['command_status'] == 'FAILED'
        assert result['artifact_integrity'] == ('FAILED' if tampered else 'VERIFIED')
        assert run_path.read_bytes() == frozen
        report_dir = root / 'result' / run_id
        if tampered:
            assert not result['report_files_written'] and not list(report_dir.iterdir())
            return
        assert result['report_status'] == 'SAVED' and result['report_files_written']
        report = (report_dir / 'report.md').read_text(encoding='utf-8')
        html = (report_dir / 'index.html').read_text(encoding='utf-8')
        for rendered in (report, html):
            assert '状态：FAILED' in rendered
            assert '|9|3|SKINNY|4|0.693147|50.0000|0.500000|local: 1|' in rendered
            assert '|18|' not in rendered and '|27|' not in rendered
            assert '计划轮数：9' in rendered and '已记录完成更新：9' in rendered
            assert dev_ref in rendered and digest in rendered
        assert json.loads((report_dir / 'source.json').read_bytes()) == dict(
            schema=1, run_id=run_id, run_ref=f'data/{run_id}/run.json',
            run_sha256=saved['run_sha256'])
        return
    fitted = case.startswith('ok') or case == 'missing-payload'
    completed = [9, 18, 27] if fitted else [9] if case == 'interrupt' else []
    assert [o['updates'] for o in summary['dev_observations']] == completed
    assert observations == ([9, 18, 27] if fitted else [9, 18] if case == 'interrupt' else [])
    assert counts['completed_forward_rows'] == 4 * (len(updates) + len(completed))
    assert counts['attempted_forward_rows'] == 4 * (len(updates) + len(observations))
    for ref in saved['artifacts']:
        assert ref.startswith(('payload/dev-', 'payload/batches-'))
    ledger = [row for ref in summary.get('batch_refs', []) for row in saved['payloads'][ref]]
    assert [row['lr'] for row in ledger] == updates
    assert [row['cycle'] for row in ledger] == list(range(len(updates)))
    for observation in summary['dev_observations']:
        ref = observation['evidence_ref']
        assert ref == f"payload/dev-{observation['updates']}.json"
        payload = json.loads((root / 'data' / run_id / ref).read_bytes())
        assert payload == {k: v for k, v in observation.items() if k != 'evidence_ref'}
        assert payload['epoch'] == payload['updates'] // recipe.training.batches_per_epoch
        assert payload['task'] == 'SKINNY' and payload['dev']['groups'] == 4
        assert payload['mean_online_train_bce'] == .5
        assert payload['routes'] == {'SKINNY': {'local': 1.}}
    bad_pool = case in ('bad-pin', 'bad-source-digest', 'tamper')
    assert saved['status'] == ('SUCCEEDED' if fitted else 'FAILED' if bad_pool else 'STOPPED')
    if case != 'pre-source':
        assert saved['policy']['deadline_utc'] == DEADLINE
        assert saved['policy']['request']['gpu_uuid'] == GPU
    if bad_pool:
        assert summary['reason'] == 'POOL_SOURCE_INVALID' and not updates
    if case == 'missing-payload':
        (root / 'data' / run_id / 'payload/dev-18.json').unlink()
        assert cli._finish(store, root, run_id, resources=res) == 6
        assert not (root / 'result' / run_id / 'report.md').exists()
        return
    exit_code = 0 if fitted else 4 if bad_pool else 3
    assert cli._finish(store, root, run_id, resources=res) == exit_code
    report = (root / 'result' / run_id / 'report.md').read_text(encoding='utf-8')
    html = (root / 'result' / run_id / 'index.html').read_text(encoding='utf-8')
    assert 'DEV BCE' in report and 'DEV BCE' in html
    assert '|TEST' not in report and '已保存 TEST' not in html
    assert '"test":' not in report and '"persisted_reload"' not in report
    assert ('|27|' in report) == fitted


@pytest.mark.parametrize('case', ['wrong-point', 'wrong-epoch', 'wrong-task', 'wrong-groups',
                                  'invalid-scalar', 'oversized', 'count-short', 'gap', 'missing',
                                  'success-short'])
def test_finish_rejects_invalid_registered_dev(
        case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    recipe = config.VOLUME_RECIPES[0]
    run_id = recipe.run_id
    root = tmp_path / 'out'
    clocks = Clocks(wall=lambda: 0., cpu=lambda: 0., utc=lambda: NOW)
    res = Resources(Meter(config.make_policy(recipe, clocks, DEADLINE), clocks, lambda: 0))
    store = RunStore.claim(root, run_id, output_bytes=recipe.resources.artifact_bytes,
                           source_supplier=lambda: dict(SOURCE))
    # Hash-valid registrations alone cannot establish a valid point.
    updates = 3600 if case == 'gap' else 1800
    dev = dict(groups=100_000, bce=.5, accuracy=.5)
    row = dict(updates=updates, epoch=updates // recipe.training.batches_per_epoch,
               task='SKINNY', dev=dev, routes={'SKINNY': {'local': 1.}}, mean_online_train_bce=.5)
    if case == 'wrong-point':
        row['updates'] = 3600
    if case == 'wrong-epoch':
        row['epoch'] = 1
    if case == 'wrong-task':
        row['task'] = 'SIMON'
    if case == 'wrong-groups':
        dev['groups'] = 1
    if case == 'invalid-scalar':
        dev['accuracy'] = True
    if case == 'oversized':
        row['routes'] = {'SKINNY': {'a' * config.DEV_OBSERVATION_BYTES: 1.}}
    ref = store.write_payload(f'dev-{updates}', row)
    forward = updates * recipe.training.batch_groups + recipe.training.dev_groups
    counts = res.snapshot()
    for key in ('attempted', 'completed'):
        counts[f'{key}_optimizer_updates'] = updates
        counts[f'{key}_forward_rows'] = forward - int(case == 'count-short')
    status = 'SUCCEEDED' if case == 'success-short' else 'FAILED'
    store.finalize(dict(status=status, counts=counts, recipe=recipe.to_dict(),
                        summary=dict(dev_observations=[dict(row, evidence_ref=ref)])))
    if case == 'missing':
        (root / 'data' / run_id / ref).unlink()
    assert cli._finish(store, root, run_id, resources=res) == 6
    result = json.loads(capsys.readouterr().out)
    assert result['scientific_status'] == status and result['command_status'] == 'FAILED'
    assert result['reason'] == 'SAVED_ARTIFACT_INTEGRITY_FAILED'
    assert not result['report_files_written'] and not list((root / 'result' / run_id).iterdir())


@pytest.mark.parametrize('ref', ['model/M-best.pt', 'dataset/test-SKINNY.npz'])
def test_finish_rejects_nonpayload_before_any_open(
        ref: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab import artifacts

    monkeypatch.setattr(source, 'no_reparse', forbidden)
    monkeypatch.setattr(artifacts, '_stream', forbidden)
    recipe = config.VOLUME_RECIPES[0]
    record = dict(recipe=recipe.to_dict(), status='FAILED',
                  artifacts={'payload/dev-1800.json': {}, ref: {}})
    with pytest.raises(ValueError, match='VOLUME_ARTIFACT_KIND'):
        cli._verified_artifacts(Path('/never-open'), record, recipe.run_id)

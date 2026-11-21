"""The one product entry: use fake device/assemblers and the real artifact owner."""
from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest
from plugin_isolation import forbidden

from cipher_lab.cli import main

PLUGIN = 'cipher_lab_plugins.innovation1.cli'
FRESH = Path('configs/innovation1-fresh.json')
EARLY = Path('configs/innovation1-skinny-layout-early-stop.json')
FIXED = Path('configs/innovation1-skinny-layout-fixed-mse.json')
UUID = 'GPU-00000000-1111-2222-3333-444444444444'


def run_args(config: Path, root: Path, run_id: str = 'artificial-new') -> list[str]:
    return ['--plugin', PLUGIN, 'run', '--config', str(config), '--root', str(root),
            '--id', run_id, '--gpu-uuid', UUID]


@pytest.mark.parametrize('path', [FRESH, EARLY, FIXED])
def test_top_level_check(path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['--plugin', PLUGIN, 'check', '--config', str(path)]) == 0
    value = json.loads(capsys.readouterr().out)
    epochs, updates = {FRESH: (24, 196608), EARLY: (100, 204800), FIXED: (50, 180000)}[path]
    assert value['recipe']['training']['epochs'] == epochs
    assert value['estimate']['optimizer_updates'] == updates


@pytest.mark.parametrize('path', [FRESH, EARLY, FIXED])
def test_only_exact_and_bounded_config(path: Path, tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.config import load_config

    raw = json.loads(path.read_bytes())
    changed = tmp_path / 'changed.json'
    raw['training']['epochs'] -= 1
    for content in (json.dumps(raw), '{"a":1,"a":1}', ' ' * 16385,
                    '[' * 4000 + ']' * 4000, '{"method":NaN}'):
        changed.write_text(content)
        with pytest.raises(ValueError):
            load_config(changed)


@pytest.mark.parametrize('subpackage', ['single_train', 'single_pair', 'layout_probe'])
def test_old_cli_is_gone(subpackage: str) -> None:
    assert find_spec(f'cipher_lab_plugins.innovation1.{subpackage}.cli') is None


@pytest.mark.parametrize('path', [FRESH, EARLY, FIXED])
def test_device_failure_is_sealed_before_science(path: Path, tmp_path: Path,
                                                  capsys: pytest.CaptureFixture[str],
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1 import application, resources

    from cipher_lab.artifacts import read_saved

    def stop(*args: Any, **kwargs: Any) -> None:
        raise resources.ResourceStop('CUDA_COUNT_MISMATCH')

    monkeypatch.setattr(resources, 'production_rss_bytes', lambda: 7)
    monkeypatch.setattr(application, 'prepare_device', stop, raising=False)
    monkeypatch.setattr(application, '_production_provider', forbidden)
    assert main(run_args(path, tmp_path)) == 3
    result = json.loads(capsys.readouterr().out)
    saved: Any = read_saved(tmp_path, 'artificial-new')
    assert result['scientific_status'] == saved['status'] == 'STOPPED'
    assert saved['summary']['detail_code'] == 'CUDA_COUNT_MISMATCH'
    assert saved['counts']['completed_optimizer_updates'] == 0
    assert not saved['artifacts']


@pytest.mark.parametrize('path', [FRESH, EARLY, FIXED])
def test_run_reads_once_and_dispatches_with_same_meter_and_owner(
    path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.innovation1 import application, resources, training
    from cipher_lab_plugins.innovation1.layout_probe import application as layout
    from cipher_lab_plugins.innovation1.layout_probe.config import LayoutRecipe

    from cipher_lab.artifacts import RunStore, read_saved
    from cipher_lab.execution import Stopped

    events: list[str] = []
    owners: list[RunStore] = []
    meters: list[resources.Resources] = []
    open_file, claim, make_resources = Path.open, RunStore.claim, application.make_resources

    def open_once(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == path:
            events.append('config')
        return open_file(self, *args, **kwargs)

    def tracked_claim(*args: Any, **kwargs: Any) -> RunStore:
        store = claim(*args, **kwargs)
        owners.append(store)
        events.append('claim')
        return store

    def tracked_resources(recipe: Any) -> resources.Resources:
        meter = make_resources(recipe)
        meters.append(meter)
        return meter

    def device(recipe: Any, uuid: str) -> None:
        assert uuid == UUID
        assert isinstance(recipe, LayoutRecipe) == (path != FRESH)
        events.append('device')

    provider, runtime = object(), object()

    def assemble(recipe: Any, ledger: resources.Resources) -> tuple[object, object]:
        assert ledger is meters[0] and events[-1] == 'device'
        events.append('assemble')
        if path != FRESH:
            assert recipe.run_id == 'artificial-new'
        return provider, runtime

    def train_four(recipe: Any, p: object, r: object, store: RunStore,
                   ledger: resources.Resources, *, backend: str, launch: Any) -> None:
        assert not isinstance(recipe, LayoutRecipe)
        assert launch['gpu_uuid'] == UUID
        assert p is provider and r is runtime and store is owners[0] and ledger is meters[0]
        assert backend == 'ciphers'
        events.append('four')
        # A dispatched return value is not an owner seal.

    def train_layout(recipe: LayoutRecipe, p: object, r: object, store: RunStore,
                     ledger: resources.Resources, summary: Any, backend: str) -> None:
        assert recipe.run_id == 'artificial-new' and recipe.fixed_pool == (path == FIXED)
        assert recipe.training.epochs == (50 if path == FIXED else 100)
        assert summary['launch']['gpu_uuid'] == UUID
        assert p is provider and r is runtime and store is owners[0] and ledger is meters[0]
        assert backend == 'ciphers'
        events.append('layout')
        raise Stopped('WALL_LIMIT')

    monkeypatch.setattr(Path, 'open', open_once)
    monkeypatch.setattr(RunStore, 'claim', tracked_claim)
    monkeypatch.setattr(resources, 'production_rss_bytes', lambda: 7)
    monkeypatch.setattr(application, 'make_resources', tracked_resources)
    monkeypatch.setattr(application, 'prepare_device', device)
    monkeypatch.setattr(application, '_assemble_four', assemble if path == FRESH else forbidden)
    monkeypatch.setattr(layout, 'assemble', assemble if path != FRESH else forbidden)
    monkeypatch.setattr(training, 'run', train_four if path == FRESH else forbidden)
    monkeypatch.setattr(layout, '_train', train_layout if path != FRESH else forbidden)
    is_layout = path != FRESH
    assert main(run_args(path, tmp_path)) == (3 if is_layout else 7)
    assert events == ['config', 'claim', 'device', 'assemble', 'layout' if is_layout else 'four']
    assert len(owners) == len(meters) == 1
    output = json.loads(capsys.readouterr().out)
    assert output['scientific_status'] == ('STOPPED' if is_layout else 'UNKNOWN')
    saved: Any = read_saved(tmp_path, 'artificial-new')
    result = tmp_path / 'result' / 'artificial-new'
    if is_layout:
        assert saved['recipe']['run_id'] == saved['run_id'] == 'artificial-new'
        assert saved['policy']['request']['gpu_uuid'] == UUID
        assert json.loads((result / 'source.json').read_bytes())['run_id'] == saved['run_id']
    else:
        assert not list(result.iterdir())


def test_claim_does_not_adopt_existing_or_reparse_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stat
    from types import SimpleNamespace

    from cipher_lab_plugins.innovation1 import resources

    monkeypatch.setattr(resources, 'production_rss_bytes', lambda: 7)
    old = tmp_path / 'models' / 'artificial-new'
    old.mkdir(parents=True)
    marker = old / 'marker'
    marker.write_bytes(b'old')
    assert main(run_args(FRESH, tmp_path)) == 2
    assert marker.read_bytes() == b'old' and not (tmp_path / 'data').exists()
    symlink = tmp_path / 'symlink'
    symlink.symlink_to(old, target_is_directory=True)
    assert main(run_args(FRESH, symlink)) == 2
    lstat = Path.lstat

    def fake_reparse(path: Path) -> Any:
        if path == tmp_path / 'junction':
            return SimpleNamespace(st_mode=stat.S_IFDIR,
                                   st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return lstat(path)

    monkeypatch.setattr(Path, 'lstat', fake_reparse)
    assert main(run_args(FRESH, tmp_path / 'junction' / 'never-created')) == 2
    assert not (tmp_path / 'junction').exists()


@pytest.mark.parametrize('run_id', ['../escape', 'NUL',
                                  'innovation1-skinny-layout-train-20261119-01',
                                  'INNOVATION1-SKINNY-LAYOUT-EARLY-STOP-20261120-01'])
def test_reserved_or_old_id_rejected(tmp_path: Path, run_id: str) -> None:
    assert main(run_args(EARLY, tmp_path, run_id)) == 2
    assert not list(tmp_path.iterdir())


def test_flags_and_windows_root_contract(tmp_path: Path) -> None:
    from pathlib import PureWindowsPath

    from cipher_lab_plugins.innovation1.cli import _windows_root

    args = run_args(FRESH, tmp_path)
    assert main(args[:-2]) == 2
    assert main([*args[:-1], 'GPU-prefix']) == 2
    assert main([*args, '--device', 'cpu']) == 2
    assert main(['--plugin', PLUGIN, 'evaluate']) == 2
    assert not list(tmp_path.iterdir())
    assert _windows_root(PureWindowsPath(r'g:\LXY\candidate'))
    for bad in (r'G:\lxy-elsewhere', r'C:\lxy', r'G:\lxy\..\escape', r'G:\lxy\name:ads',
                r'G:\lxy\CON', r'G:\lxy\name.', r'G:lxy', r'\\?\G:\lxy'):
        assert not _windows_root(PureWindowsPath(bad))


def test_production_exact_recipe_rechecked_before_device(tmp_path: Path) -> None:
    from dataclasses import replace

    from cipher_lab_plugins.innovation1 import application
    from cipher_lab_plugins.innovation1.config import Recipe
    from cipher_lab_plugins.innovation1.layout_probe.config import EARLY_STOP, SLOTS, official
    from cipher_lab_plugins.innovation1.layout_probe.weights import identity

    from cipher_lab.artifacts import RunStore

    run_id = 'fresh-id'
    source = {'artificial/fixture.py': 'a'*64}
    store = RunStore.claim(tmp_path, run_id, output_bytes=2**20,
                           needs_models=True, model_slots=SLOTS, source_supplier=lambda: source)
    layout = replace(EARLY_STOP, run_id=run_id)
    assert official(layout, run_id) == layout  # equal resolved value, not singleton identity
    four = Recipe()
    for bad in (replace(layout, training=replace(layout.training, epochs=99)),
                replace(layout, cosine_epochs=25), replace(layout, run_id='different'),
                replace(layout, resources=replace(layout.resources, intra_threads=2)),
                replace(four, optimizer=replace(four.optimizer, weight_decay=.1))):
        ledger = application.make_resources(four)
        with pytest.raises(ValueError, match='CONFIG_EXACT'):
            application.execute(bad, store, ledger, gpu_uuid=UUID, run_id=run_id)
        if not isinstance(bad, Recipe) and bad.run_id == run_id:
            with pytest.raises(ValueError, match='CONFIG_EXACT'):
                identity(bad, source, 'ciphers', SLOTS[0], 1, .5)
    with pytest.raises(ValueError, match='RUN_IDENTITY'):
        application.execute(layout, store, application.make_resources(layout), gpu_uuid=UUID,
                            run_id='different')


def test_first_snapshot_storage_failure_is_artifact_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.innovation1 import application, resources

    from cipher_lab.artifacts import RunStore, read_saved

    monkeypatch.setattr(resources, 'production_rss_bytes', lambda: 7)
    monkeypatch.setattr(application, 'prepare_device', forbidden)
    monkeypatch.setattr(RunStore, 'finalize', forbidden)
    atomic, run_writes = RunStore._atomic, []

    def fault(store: RunStore, target: Path, content: bytes) -> None:
        if target.name == 'run.json':
            run_writes.append(target)
            if len(run_writes) == 2:  # claim's RUNNING record was already committed
                raise OSError('injected first business snapshot failure')
        atomic(store, target, content)

    monkeypatch.setattr(RunStore, '_atomic', fault)
    assert main(run_args(FRESH, tmp_path)) == 4
    output = json.loads(capsys.readouterr().out)
    saved: Any = read_saved(tmp_path, 'artificial-new')
    assert saved['status'] == 'FAILED' and saved['recipe'] == {}
    assert saved['summary']['reason'] == 'WRITE_FAILED'
    assert output['scientific_status'] == 'FAILED'
    assert output['scientific_reason'] == 'WRITE_FAILED'
    assert output['command_status'] == 'FAILED' and output['exit_code'] == 4
    assert output['report_status'] == 'NOT_ATTEMPTED'
    assert output['report_files_written'] is False
    assert output['reason'] == 'ARTIFACT_WRITE_FAILED'
    assert output['request_identity'] == {
        'gpu_uuid': UUID, 'scope': 'REQUESTED_ONLY_NOT_PERSISTED'}
    assert not list((tmp_path / 'result' / 'artificial-new').iterdir())


def test_training_storage_failure_keeps_request_identity_and_saved_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.innovation1 import application, resources, training
    from cipher_lab_plugins.innovation1.config import Recipe
    from cipher_lab_plugins.innovation1.single_train.config import DEVICE

    from cipher_lab.artifacts import ArtifactError, RunStore, read_saved

    monkeypatch.setattr(resources, 'production_rss_bytes', lambda: 7)
    monkeypatch.setattr(application, 'prepare_device', lambda recipe, uuid: None)
    monkeypatch.setattr(application, '_assemble_four',
                        lambda recipe, ledger: (object(), object()))
    claims, finalizes = [], []
    claim, finalize = RunStore.claim, RunStore.finalize

    def tracked_claim(*args: Any, **kwargs: Any) -> RunStore:
        store = claim(*args, **kwargs)
        claims.append(store)
        return store

    def tracked_finalize(store: RunStore, value: Any) -> None:
        finalizes.append(dict(value))
        finalize(store, value)

    def train(recipe: Any, provider: Any, runtime: Any, store: RunStore,
              ledger: resources.Resources, *, backend: str, launch: Any) -> None:
        assert store is claims[0] and launch['gpu_uuid'] == UUID
        ledger.reserve('optimizer_updates', 2)
        ledger.complete('optimizer_updates', 2)
        store.write_snapshot({'summary': {'method': recipe.method, 'reason': 'PAID'},
                              'counts': ledger.snapshot()})
        atomic = store._atomic

        def fault(target: Path, content: bytes) -> None:
            if target.name == 'dev-SIMECK.npz':
                raise OSError('injected training artifact failure')
            atomic(target, content)

        monkeypatch.setattr(store, '_atomic', fault)
        try:
            store.write_dataset('dev-SIMECK', b'artificial')
        except ArtifactError:
            return  # the real loop returns after the owner sealed itself

    monkeypatch.setattr(RunStore, 'claim', tracked_claim)
    monkeypatch.setattr(RunStore, 'finalize', tracked_finalize)
    monkeypatch.setattr(training, 'run', train)
    assert main(run_args(FRESH, tmp_path)) == 4
    output = json.loads(capsys.readouterr().out)
    saved: Any = read_saved(tmp_path, 'artificial-new')
    assert finalizes == []  # storage already sealed; no second finalize
    assert saved['status'] == 'FAILED' and saved['status'] != 'SUCCEEDED'
    assert saved['summary']['reason'] == 'WRITE_FAILED'
    assert saved['counts']['completed_optimizer_updates'] == 2
    assert saved['recipe']['method'] == Recipe().method
    request = {'gpu_uuid': UUID, 'device': DEVICE, 'gpu_memory_scope': 'process_allocator'}
    assert saved['policy']['request'] == request
    assert output['scientific_status'] == 'FAILED' and output['scientific_reason'] == 'WRITE_FAILED'
    assert output['command_status'] == 'FAILED' and output['exit_code'] == 4
    assert output['report_status'] == 'SAVED'
    assert output['request_identity'] == {**request, 'scope': 'PERSISTED_POLICY'}


@pytest.mark.parametrize('sealed', [False, True])
def test_terminal_text_is_not_a_seal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                    capsys: pytest.CaptureFixture[str], sealed: bool) -> None:
    from cipher_lab_plugins.innovation1.cli import _finish
    from cipher_lab_plugins.innovation1.tests.test_task2 import setup

    _, _, _, store, ledger = setup(tmp_path)
    path = tmp_path / 'data' / 'manual' / 'run.json'
    if sealed:
        store.finalize({'status': 'FAILED', 'summary': {'reason': 'FIXTURE_FAILURE'}})
    fake = json.loads(path.read_bytes())
    fake['status'] = 'SUCCEEDED'
    path.write_text(json.dumps(fake))
    monkeypatch.setattr(store, 'finalize', forbidden)
    assert _finish(store, tmp_path, 'manual', resources=ledger) == 7
    status = json.loads(capsys.readouterr().out)
    assert status['scientific_status'] == 'UNKNOWN' and status['report_status'] == 'NOT_ATTEMPTED'
    assert not list((tmp_path / 'result' / 'manual').iterdir())

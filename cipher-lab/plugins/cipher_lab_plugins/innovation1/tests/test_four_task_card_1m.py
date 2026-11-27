"""Exercise the fixed four-task owner without encryption, training or GPU use."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path, PureWindowsPath
from typing import Any, Literal
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from cipher_lab_plugins.innovation1 import application, cli
from cipher_lab_plugins.innovation1.config import Task, canonical, load_config
from cipher_lab_plugins.innovation1.four_single import config, models, replay, training
from cipher_lab_plugins.innovation1.four_single.data import encode_panel
from cipher_lab_plugins.innovation1.layout_probe.reuse import _pinned
from cipher_lab_plugins.innovation1.resources import Resources
from cipher_lab_plugins.innovation1.serial_bc.config import DEADLINE_ENV, SERIAL_B
from cipher_lab_plugins.innovation1.serial_bc.data import SerialPools
from cipher_lab_plugins.innovation1.single_train.config import BACKEND, learning_rate
from cipher_lab_plugins.innovation1.single_train.data import PairBatch
from cipher_lab_plugins.innovation1.tests.test_four_single_replay import SOURCE, _archive, _batch
from plugin_isolation import forbidden
from torch import Tensor

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Clocks, Meter, Stopped
from cipher_lab.representations.tasks import public_features

CONFIG = Path('configs/innovation1-four-task-card-1m.json')


def test_fixed_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                      capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab_plugins.innovation1.four_single.interleave_config import (
        GPU_UUID,
        INTERLEAVE,
        ROOT,
        make_policy,
        validate_launch,
    )

    recipe = load_config(CONFIG)
    assert recipe == INTERLEAVE
    assert recipe.training == replace(SERIAL_B.training, epochs=30)
    assert recipe.subset_recipe() == replace(SERIAL_B, training=recipe.training)
    assert recipe.cosine_epochs == 30
    assert recipe.early_stopping.warmup_epochs == 10 and recipe.early_stopping.patience == 8
    assert recipe.early_stopping.min_delta == 1e-5
    estimates = config.estimate(recipe)
    assert [estimates[key] for key in ('fits', 'optimizer_updates', 'forward_rows',
                                      'real_blocks', 'null_words', 'retries')] == [
        1, 21600, 121048576, 0, 0, 0]
    validate_launch(PureWindowsPath(ROOT), GPU_UUID)
    with pytest.raises(ValueError):
        validate_launch(PureWindowsPath(ROOT), 'GPU-' + '0' * 36)
    with pytest.raises(ValueError):
        validate_launch(PureWindowsPath(recipe.archive.root), GPU_UUID)
    now = datetime(2026, 11, 27, tzinfo=UTC)
    clocks = Clocks(utc=lambda: now)
    end = (now + timedelta(seconds=7200)).isoformat()
    assert make_policy(recipe, clocks, end).deadline_utc == end
    for bad in (None, now.isoformat(), (now + timedelta(seconds=7201)).isoformat()):
        with pytest.raises(Stopped, match='DEADLINE'):
            make_policy(recipe, clocks, bad)
    monkeypatch.setenv(DEADLINE_ENV, 'bad')
    monkeypatch.setattr(application, 'make_resources', forbidden)
    assert cli.main(['check', '--config', str(CONFIG)]) == 0
    assert json.loads(capsys.readouterr().out)['recipe'] == recipe.to_dict()
    value = recipe.to_dict()
    value['selection']['early_stopping']['patience'] = 7
    invalid = tmp_path / 'invalid.json'
    invalid.write_text(canonical(value))
    with pytest.raises(ValueError, match='CONFIG_EXACT'):
        load_config(invalid)


def test_test_panel_keeps_original_identity() -> None:
    from cipher_lab_plugins.innovation1.four_single.interleave_config import INTERLEAVE

    task = INTERLEAVE.tasks[1]
    recipe = replace(INTERLEAVE, training=replace(INTERLEAVE.training, test_groups=4))
    old = _batch(4, 'test', config.NAMESPACE, 9)
    source = dict(root=recipe.archive.root, run_id=recipe.archive.run_id,
                  record_ref=f'data/{recipe.archive.run_id}/run.json',
                  record=vars(recipe.archive.record))
    prior = dict(origin='reused', ref='dataset/test-PRESENT.npz', groups=4, digest=old.digest,
                 namespace=config.NAMESPACE, bytes=999, sha256='b' * 64, source_record=source,
                 source_best=dict(ref='model/M-best.pt', sha256='d' * 64, epoch=9))
    logits = np.zeros((4, 1), dtype=np.float64)
    best = {'epoch': 1}
    assert encode_panel(old, task, recipe, SOURCE, BACKEND,
                        logits=logits, best=best, reused_input=prior)
    with pytest.raises(ValueError, match='PANEL_IDENTITY'):
        encode_panel(old, task, recipe, SOURCE, BACKEND, logits=logits, best=best,
                     reused_input={**prior, 'digest': '0' * 64})


def test_public_lookup_is_buffered_and_branch_local(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.four_single.interleave_config import INTERLEAVE
    from cipher_lab_plugins.innovation1.four_single.interleave_runtime import InterleaveM
    from cipher_lab_plugins.innovation1.present_neighborhood.layout import STRUCTURE_SHA256

    rng = torch.random.get_rng_state()
    model = models.build(INTERLEAVE, InterleaveM)
    assert torch.equal(torch.random.get_rng_state(), rng)
    saved = models.geometry(model)
    assert saved['present_structure'] == STRUCTURE_SHA256
    assert set(saved) == models.FIXED_BUFFERS | {'present_structure'}
    bits = torch.zeros((4, 1, 2, 64), dtype=torch.float32)
    calls: list[Any] = []

    def inspect(shared: Any, slots: Tensor, public: Tensor, *, rounds: int,
                nibble: Any, route_for: Any) -> Tensor:
        assert shared is model.shared and slots.shape == (1, 4, 2, 64)
        assert type(rounds) is int and callable(route_for)
        calls.append(nibble)
        return torch.zeros((4, 1))

    monkeypatch.setattr(models, 'shared_logits', inspect)
    for task in INTERLEAVE.tasks:
        model(bits, torch.tensor(public_features(task.cipher)), rounds=task.rounds)
    assert [v is not None for v in calls] == [False, True, False, True]
    assert models.geometry(model) == saved


@pytest.mark.parametrize(('case', 'v2'), [
    ('early', False), ('test-missing', False), ('seal-cutoff', False),
    ('early', True), ('test-missing', True), ('seal-cutoff', True),
])
def test_four_task_owner(case: str, v2: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                         capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab_plugins.innovation1.four_single import interleave_runtime as runtime
    from cipher_lab_plugins.innovation1.four_single.interleave_config import (
        GPU_UUID,
        INTERLEAVE,
        STRUCTURE_INTERLEAVE,
        make_policy,
    )

    original, _, _ = _archive(tmp_path / 'archive', case)
    base = STRUCTURE_INTERLEAVE if v2 else INTERLEAVE
    recipe = replace(base, archive=original.archive, skinny_source=original.skinny_source,
                     training=replace(original.training, epochs=30))
    now = datetime(2026, 11, 27, tzinfo=UTC)
    tick = [0.]
    clocks = Clocks(wall=lambda: tick[0], cpu=lambda: 0.,
                    utc=lambda: now + timedelta(seconds=tick[0]))
    policy = make_policy(recipe, clocks, (now + timedelta(seconds=7200)).isoformat())
    resources = Resources(Meter(policy, clocks, lambda: 0))
    root = tmp_path / 'run'
    store = RunStore.claim(root, recipe.run_id, output_bytes=2**30, needs_models=True,
                           model_slots=(config.SLOT,), source_supplier=lambda: SOURCE)
    record = root / 'data' / recipe.run_id / 'run.json'
    names = [task.cipher for task in recipe.tasks]
    reference = SerialPools(recipe.subset_recipe(), lambda: None, archive=original)
    _, original_bindings = reference.bind(lambda _: None)
    reference_batches = {task.cipher: [reference.train(task, i).digest for i in range(9)]
                         for task in recipe.tasks}
    updates: list[str] = []
    tests_read: list[str] = []
    tests_scored: list[str] = []
    restored = False

    def read(path: Path, *args: Any) -> Any:
        assert path.suffix != '.pt'
        if path.name.startswith('test-'):
            assert restored
            tests_read.append(path.name)
        return _pinned(path, *args)

    open_member = ZipFile.open

    def member(archive: ZipFile, key: Any, mode: Literal['r', 'w'] = 'r', **kw: Any) -> Any:
        assert mode != 'r' or getattr(key, 'filename', key) != 'logits.npy'
        return open_member(archive, key, mode, **kw)

    class FakeRuntime:
        def __init__(self, rec: Any, check: Any) -> None:
            assert rec is recipe

        def metadata(self) -> dict[str, Any]:
            return {config.SLOT: {'geometry': {'artificial': '1' * 64}}}

        def prepare(self, batch: PairBatch) -> tuple[PairBatch, Any]:
            return batch, batch.labels

        def update(self, slot: str, task: Task, batch: PairBatch, labels: Any, lr: float,
                   budget: Resources) -> float:
            cycle, offset = divmod(len(updates), 4)
            assert task.cipher == names[offset] and batch.index == cycle and not tests_read
            if cycle < 9:
                assert batch.digest == reference_batches[task.cipher][cycle]
            assert slot == config.SLOT and lr == learning_rate(recipe, cycle, cosine_epochs=30)
            budget.reserve('forward_rows', batch.groups)
            budget.complete('forward_rows', batch.groups)
            updates.append(task.cipher)
            return .5

        def predict(self, slot: str, task: Task, batch: PairBatch) -> Any:
            assert slot == config.SLOT
            if batch.split == 'test':
                assert restored and len(tests_read) == 4
                tests_scored.append(task.cipher)
            return np.zeros(batch.labels.shape, dtype=np.float64)

        def routes(self) -> dict[str, dict[str, float]]:
            return {t: {'word': .75, 'nibble' if t == 'PRESENT' else 'local': .25} for t in names}

        def checkpoint(self, slot: str, meta: dict[str, Any]) -> bytes:
            assert meta['pools'] == original_bindings and meta['epoch'] == 1
            return canonical(meta).encode()

        def release(self) -> None:
            assert len(updates) == 18 * 36 and not tests_read

        def restore(self, slot: str, raw: bytes, meta: dict[str, Any]) -> None:
            nonlocal restored
            assert json.loads(raw) == meta and not tests_read
            restored = True

    def pools(rec: Any, check: Any) -> SerialPools:
        assert rec is recipe
        return SerialPools(rec.subset_recipe(), check, archive=original, defer_activate=True)

    atomic = store._atomic

    def late_seal(path: Path, content: bytes) -> None:
        if case == 'seal-cutoff' and path == record:
            if json.loads(content)['status'] == 'SUCCEEDED':
                tick[0] = 7200.
        atomic(path, content)

    monkeypatch.setattr(store, '_atomic', late_seal)
    monkeypatch.setattr(replay, '_pinned', read)
    monkeypatch.setattr(ZipFile, 'open', member)
    monkeypatch.setattr(runtime, 'assemble_pools', pools)
    monkeypatch.setattr(runtime, 'InterleaveRuntime', FakeRuntime)
    monkeypatch.setattr(training, 'assemble', forbidden)
    monkeypatch.setattr(training, 'official', lambda rec, _: rec)
    monkeypatch.setattr(application, 'official_four_single', lambda rec, _: rec)
    monkeypatch.setattr(application, 'validate_interleave_launch', lambda *_, **__: None)
    monkeypatch.setattr(application, 'prepare_device', lambda *_: None)
    application.execute(recipe, store, resources, gpu_uuid=GPU_UUID, run_id=recipe.run_id)
    saved = json.loads(record.read_bytes())
    s = saved['summary']
    assert saved['recipe']['structure_card'] == recipe.to_dict()['structure_card']
    assert s['progress']['completed_dev_epochs'] == 18 and s['counts_scope'] == 'PRE_OWNER_SEAL'
    assert s['training_termination'] == dict(reason='EARLY_STOPPING', completed_epochs=18)
    assert s['best'] == s['persisted_reload'] and s['best']['epoch'] == 1
    assert s['persisted_reload']['nibble_lookup']['PRESENT'] == 'present_structure'
    assert s['persisted_reload']['nibble_lookup']['SKINNY'] is None
    assert saved['counts']['completed_optimizer_updates'] == 18 * 36
    assert saved['counts']['completed_real_blocks'] == 0
    assert s['pools'] == original_bindings
    rc = cli._finish(store, root, recipe.run_id, resources=resources)
    outcome = json.loads(capsys.readouterr().out)
    assert rc == {'early': 0, 'test-missing': 4, 'seal-cutoff': 3}[case]
    if case == 'test-missing':
        assert saved['status'] == 'FAILED' and s['reason'] == 'TEST_SOURCE_INVALID'
        assert not tests_scored and not s['test']
    else:
        assert saved['status'] == 'SUCCEEDED' and tests_scored == names
        assert set(s['test']) == set(names)
        assert all(x['panel']['digest'] == x['panel']['reused_input']['digest']
                   for x in s['test'].values())
    if case == 'seal-cutoff':
        assert outcome['command_status'] == 'STOPPED' and outcome['reason'] == 'DEADLINE'
        assert not outcome['report_files_written']
    else:
        assert outcome['report_files_written']
        text = (root / 'result' / recipe.run_id / 'report.md').read_text(encoding='utf-8')
        assert '完整 DEV 轮数：18' in text and '终态写盘' in text

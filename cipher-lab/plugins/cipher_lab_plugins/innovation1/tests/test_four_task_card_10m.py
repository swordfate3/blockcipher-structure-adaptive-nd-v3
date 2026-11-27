"""Guarded engineering checks for the original full pools with the v2 model."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path, PureWindowsPath
from typing import Any
from zipfile import ZipFile

import pytest
import torch
from cipher_lab_plugins.innovation1 import cli
from cipher_lab_plugins.innovation1.config import load_config
from cipher_lab_plugins.innovation1.four_single import config, interleave_config, replay, report
from cipher_lab_plugins.innovation1.four_single.interleave_runtime import (
    InterleaveRuntime,
    assemble_pools,
)
from cipher_lab_plugins.innovation1.layout_probe.reuse import _pinned
from cipher_lab_plugins.innovation1.tests.test_four_single_replay import _archive
from plugin_isolation import forbidden

CONFIG = Path('configs/innovation1-four-task-card-10m.json')


def test_fixed_full_recipe(capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab_plugins.innovation1.four_single.interleave_config import FULL_INTERLEAVE

    recipe = FULL_INTERLEAVE
    assert load_config(CONFIG) is recipe
    assert interleave_config.official(recipe, recipe.run_id) is recipe
    tr = recipe.training
    assert (tr.batch_groups, tr.batches_per_epoch, tr.dev_groups, tr.epochs) == (
        5000, 1800, 1000000, 30)
    assert (recipe.cosine_epochs, recipe.resources.wall_seconds) == (30, 16 * 3600)
    assert recipe.archive_reader_recipe().namespace == config.NAMESPACE
    assert cli.main(['check', '--config', str(CONFIG)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['recipe'] == recipe.to_dict()
    assert [result['estimate'][key] for key in (
        'optimizer_updates', 'forward_rows', 'real_blocks', 'null_words', 'retries',
    )] == [216000, 1201048576, 0, 0, 0]
    small = interleave_config.STRUCTURE_INTERLEAVE
    for key in ('model', 'geometry', 'structure_card', 'routing', 'tasks', 'seeds'):
        assert result['recipe'][key] == small.to_dict()[key]
    interleave_config.validate_launch(PureWindowsPath(recipe.output_root),
                                       interleave_config.GPU_UUID, recipe=recipe)
    with pytest.raises(ValueError):
        interleave_config.validate_launch(PureWindowsPath(small.output_root),
                                           interleave_config.GPU_UUID, recipe=recipe)
    saved = dict(recipe=recipe.to_dict(), run_id=recipe.run_id, run_ref='data/test/run.json',
                 run_sha256='a' * 64, status='STOPPED', summary={}, counts={}, source={},
                 artifacts={})
    assert 'TRAIN 900 万对和 DEV 100 万对' in report.render(saved)['report.md'].decode()


def test_full_replay_preserves_archive_and_train_order(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.four_single.interleave_config import FULL_INTERLEAVE

    original, _, batches = _archive(tmp_path, 'ok')
    recipe = replace(FULL_INTERLEAVE, archive=original.archive,
                     skinny_source=original.skinny_source,
                     training=replace(original.training, epochs=30))
    test_reads: list[str] = []
    read_member = ZipFile.open

    def member(self: ZipFile, name: Any, *args: Any, **kwargs: Any) -> Any:
        assert getattr(name, 'filename', name) != 'logits.npy'
        return read_member(self, name, *args, **kwargs)

    def read(path: Path, *args: Any) -> Any:
        assert path.suffix != '.pt'
        if path.name.startswith('test-'):
            test_reads.append(path.name)
        return _pinned(path, *args)

    monkeypatch.setattr(ZipFile, 'open', member)
    monkeypatch.setattr(replay, '_pinned', read)
    pool = assemble_pools(recipe, lambda: None)
    assert type(pool) is replay.ReplayPools
    dev, bindings = pool.bind(lambda _: None)
    pool.activate()
    for task in recipe.tasks:
        name = task.cipher
        assert dev[name].groups == 4 and bindings[name]
        assert [pool.train(task, c).digest for c in range(18)] == batches[name][:18]
    assert not test_reads
    pool.release_train()
    tests = pool.load_tests(lambda _: None)
    assert test_reads == [f'test-{task.cipher}.npz' for task in recipe.tasks]
    for panel, identity in tests.values():
        assert panel.groups == 4 and panel.digest == identity['digest']


def test_full_and_small_build_same_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.four_single.interleave_config import FULL_INTERLEAVE

    runtime = object.__new__(InterleaveRuntime)
    runtime.contract = interleave_config.STRUCTURE_INTERLEAVE
    first = runtime._build()
    monkeypatch.setattr(replay.ReplayPools, 'bind', forbidden)
    runtime.contract = FULL_INTERLEAVE
    second = runtime._build()
    assert type(first) is type(second)
    assert first.state_dict().keys() == second.state_dict().keys()
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name

"""Guarded checks for the four-task full-pool card ablation."""
from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import pytest
import torch
from cipher_lab_plugins.innovation1 import cli
from cipher_lab_plugins.innovation1.config import load_config
from cipher_lab_plugins.innovation1.four_single import interleave_config as cfg
from cipher_lab_plugins.innovation1.four_single import models, replay, report
from cipher_lab_plugins.innovation1.four_single.interleave_runtime import (
    InterleaveRuntime,
    assemble_pools,
)
from plugin_isolation import forbidden
from torch import Tensor

from cipher_lab.models.shared_m.spec import STEM_NAMES
from cipher_lab.representations.tasks import public_features

CONFIG = Path('configs/innovation1-four-task-no-card-10m.json')


def test_ablation_contract(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab_plugins.innovation1.four_single.interleave_config import NO_CARD_INTERLEAVE

    base = cfg.FULL_INTERLEAVE
    recipe = NO_CARD_INTERLEAVE
    assert load_config(CONFIG) is recipe
    assert cfg.official(recipe, recipe.run_id) is recipe
    assert recipe.resources == base.resources
    assert recipe.archive_reader_recipe() == base.archive_reader_recipe()
    pool = assemble_pools(recipe, lambda: None)
    assert type(pool) is replay.ReplayPools and pool.recipe == base.archive_reader_recipe()
    assert cli.main(['check', '--config', str(CONFIG)]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked['recipe'] == recipe.to_dict()
    assert checked['estimate']['optimizer_updates'] == 216000
    assert checked['estimate']['real_blocks'] == 0
    value = recipe.to_dict()
    prior = base.to_dict()
    for key in ('tasks', 'training', 'resources', 'seeds', 'optimizer', 'data', 'selection'):
        assert value[key] == prior[key]
    assert 'structure_card' not in value
    assert value['routing']['constant'] == [.5] * 13
    assert set(value['geometry']['nibble_buffers'].values()) == {'shared.stems.nibble.indices'}
    assert value['model']['parameters'] == 1087509
    cfg.validate_launch(PureWindowsPath(recipe.output_root), cfg.GPU_UUID, recipe=recipe)
    with pytest.raises(ValueError):
        cfg.validate_launch(PureWindowsPath(base.output_root), cfg.GPU_UUID, recipe=recipe)
    saved = dict(recipe=value, run_id=recipe.run_id, run_ref='data/sample/run.json',
                 run_sha256='a' * 64, status='STOPPED', summary={}, counts={}, source={},
                 artifacts={})
    assert '撤卡' in report.render(saved)['report.md'].decode()

    value['routing']['constant'][0] = .6
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='CONFIG_EXACT'):
        load_config(bad)


def test_ablation_routes_default_buffers_and_keeps_representation(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.four_single.interleave_config import NO_CARD_INTERLEAVE

    runtime = object.__new__(InterleaveRuntime)
    runtime.contract = cfg.FULL_INTERLEAVE
    base = runtime._build()
    runtime.contract = NO_CARD_INTERLEAVE
    monkeypatch.setattr(models, 'nibble_indices', forbidden)
    ablated = runtime._build()
    assert type(ablated.shared.representation) is type(base.shared.representation)
    assert dict(base.named_parameters()).keys() == dict(ablated.named_parameters()).keys()
    for name, param in ablated.named_parameters():
        assert torch.equal(param, base.get_parameter(name))
    geometry = models.geometry(ablated)
    assert geometry == {name: models.geometry(base)[name] for name in geometry}
    assert set(geometry) == {f'shared.stems.{name}.indices' for name in STEM_NAMES}
    assert geometry == runtime.contract.to_dict()['geometry']['sha256']
    seen: list[Tensor] = []

    def gate(card: Tensor) -> Tensor:
        seen.append(card)
        return torch.tensor([0., 1., 2.], dtype=card.dtype)

    monkeypatch.setattr(ablated.shared.router.gate, 'forward', gate)
    routes = {}
    for task in runtime.contract.tasks:
        public = torch.tensor(public_features(task.cipher))
        assert ablated._nibble(public) is None
        routes[task.cipher] = ablated.route_weights(public)
    assert len(seen) == 4 and all(torch.equal(x, torch.full((13,), .5)) for x in seen)
    assert all(r == routes['SKINNY'] for r in routes.values())
    assert runtime.contract.nibble_lookup(routes) == dict.fromkeys(routes,
                                                                  'shared.stems.nibble.indices')
    assert runtime.contract.nibble_lookup(dict.fromkeys(routes, {'word': .7, 'local': .3})) == (
        dict.fromkeys(routes))

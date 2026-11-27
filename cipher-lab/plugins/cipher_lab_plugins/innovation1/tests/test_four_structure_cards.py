"""Card-generated nibble geometry, checked without scientific execution."""
from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from cipher_lab_plugins.innovation1 import cli
from cipher_lab_plugins.innovation1.config import load_config
from cipher_lab_plugins.innovation1.four_single import interleave_config, models
from cipher_lab_plugins.innovation1.four_single.config import estimate
from cipher_lab_plugins.innovation1.four_single.interleave_runtime import InterleaveM
from cipher_lab_plugins.innovation1.layout_probe.layout import nibble_indices
from cipher_lab_plugins.innovation1.present_neighborhood.layout import remap
from cipher_lab_plugins.innovation1.source import source_identity
from plugin_isolation import forbidden
from torch import Tensor

from cipher_lab.models.shared_m.spec import STEM_NAMES
from cipher_lab.representations.tasks import public_features

CONFIG = Path('configs/innovation1-four-task-card-1m-v2.json')


def test_all_tables_preserve_indices() -> None:
    from cipher_lab_plugins.innovation1.four_single.structure_cards import CARDS, verified_tables

    row = tuple(tuple(indices) for indices in nibble_indices('row').tolist())
    column = tuple(tuple(indices) for indices in nibble_indices('column').tolist())
    tables = verified_tables()
    assert tables == dict(SIMECK=row, PRESENT=remap(row), SIMON=row, SKINNY=column)
    assert tuple(CARDS) == tuple(tables)
    assert tables['SKINNY'][52] == (64, 36, 37, 64, 52, 53, 64, 8, 9)
    bad = replace(CARDS['SKINNY'], block_order=tuple(range(16)))
    with pytest.raises(ValueError, match='STRUCTURE_CARD_TABLE'):
        verified_tables({**CARDS, 'SKINNY': bad})
    bad_rule = replace(CARDS['SIMECK'].gather_rule, sentinel=63)
    with pytest.raises(ValueError, match='STRUCTURE_CARD_RULE'):
        verified_tables({**CARDS, 'SIMECK': replace(CARDS['SIMECK'], gather_rule=bad_rule)})


def test_exact_v2_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab_plugins.innovation1.four_single.interleave_config import STRUCTURE_INTERLEAVE

    recipe = STRUCTURE_INTERLEAVE
    assert load_config(CONFIG) == recipe
    assert cli.main(['check', '--config', str(CONFIG)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['recipe'] == recipe.to_dict()
    assert result['estimate'] == estimate(interleave_config.INTERLEAVE)
    old = interleave_config.INTERLEAVE.to_dict()
    v2 = result['recipe']
    for key in ('tasks', 'data', 'training', 'optimizer', 'selection', 'routing', 'seeds'):
        assert v2[key] == old[key]
    assert set(v2['structure_card']) == {'SIMECK', 'PRESENT', 'SIMON', 'SKINNY'}
    assert v2['structure_card']['PRESENT']['p_layer'] == old['structure_card']['p_layer']
    assert v2['geometry']['sha256']['skinny_column'] == (
        '227afc280e23f090ee6cb1e4a95a83af527bc49f09d852ed783bb8536def2418')
    assert recipe.run_id != interleave_config.RUN_ID
    assert interleave_config.official(recipe, recipe.run_id) == recipe
    with pytest.raises(ValueError, match='CONFIG_EXACT'):
        interleave_config.official(recipe, interleave_config.RUN_ID)
    interleave_config.validate_launch(PureWindowsPath(recipe.output_root),
                                      interleave_config.GPU_UUID, recipe=recipe)
    with pytest.raises(ValueError):
        interleave_config.validate_launch(PureWindowsPath(interleave_config.ROOT),
                                          interleave_config.GPU_UUID, recipe=recipe)
    with pytest.raises(ValueError):
        interleave_config.validate_launch(PureWindowsPath(recipe.output_root),
                                          interleave_config.GPU_UUID)
    v2['structure_card']['SKINNY']['block_order'] = list(range(16))
    changed = tmp_path / 'layout.json'
    changed.write_text(json.dumps(v2), encoding='utf-8')
    with pytest.raises(ValueError, match='CONFIG_EXACT'):
        load_config(changed)
    assert any(name.endswith('/four_single/structure_cards.py') for name in source_identity())


def test_v2_runtime_uses_card_tables_and_respects_top2(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.four_single.interleave_config import STRUCTURE_INTERLEAVE
    from cipher_lab_plugins.innovation1.four_single.interleave_runtime import InterleaveRuntime
    from cipher_lab_plugins.innovation1.four_single.structure_cards import verified_tables

    old = models.build(interleave_config.INTERLEAVE, InterleaveM)
    # The v2 model must obtain SKINNY's indices from the structure card.
    monkeypatch.setattr(models, 'nibble_indices', forbidden)
    recipe = STRUCTURE_INTERLEAVE
    runtime = object.__new__(InterleaveRuntime)
    runtime.contract = recipe
    new = runtime._build()
    buffers = dict(new.named_buffers())
    assert new.state_dict().keys() == old.state_dict().keys()
    assert models.geometry(new) == models.geometry(old)
    for name, tensor in old.state_dict().items():
        assert torch.equal(new.state_dict()[name], tensor), name
    tables = verified_tables()
    for cipher, name in recipe.to_dict()['geometry']['nibble_buffers'].items():
        assert torch.equal(buffers[name], torch.tensor(tables[cipher]))

    word, local, nibble = (STEM_NAMES.index(k) for k in ('word', 'local', 'nibble'))
    bits = torch.zeros((4, 1, 2, 64), dtype=torch.float32)
    calls: list[Tensor] = []

    def gather(raw: Tensor, channel: Tensor, table: Tensor) -> Tensor:
        calls.append(table)
        return raw.new_zeros((1, 4, 96, 64))

    def stem(raw: Tensor, channel: Tensor) -> Tensor:
        return raw.new_zeros((1, 4, 96, 64))

    def logits(shared: Any, slots: Tensor, public: Tensor, *, rounds: int,
               nibble: Any, route_for: Any) -> Tensor:
        from cipher_lab_plugins.innovation1.single_pair.models import shared_logits

        fake = SimpleNamespace(
            representation=lambda *a, **kw: SimpleNamespace(canonical=slots, channel=slots),
            stems=dict(word=stem, local=stem, nibble=forbidden), spec=shared.spec,
            suffix=lambda x: x,
            readout=SimpleNamespace(base=lambda x: x[:, :1], residual=lambda x: x[:, :1]))
        return cast(Tensor, shared_logits(fake, slots, public, rounds=rounds, nibble=nibble,
                                          route_for=route_for))

    monkeypatch.setattr(new, '_gather', gather)
    monkeypatch.setattr(models, 'shared_logits', logits)
    for selected in (True, False):
        indices = torch.tensor([word, nibble if selected else local])
        weights = torch.tensor([.75, .25])
        monkeypatch.setattr(new, '_route', lambda _: SimpleNamespace(indices=indices,
                                                                      weights=weights))
        recorded = recipe.nibble_lookup({t.cipher: new.route_weights(
            torch.tensor(public_features(t.cipher))) for t in recipe.tasks})
        for task in (*recipe.tasks, recipe.tasks[0]):
            calls.clear()
            new(bits, torch.tensor(public_features(task.cipher)), rounds=task.rounds)
            lookup = recorded[task.cipher]
            if selected:
                assert lookup is not None and len(calls) == 1 and calls[0] is buffers[lookup]
            else:
                assert lookup is None and not calls

    bad = new.state_dict()
    bad['skinny_column'] = bad['skinny_column'].clone()
    bad['skinny_column'][52, 1] = 0
    metadata: dict[str, object] = {'geometry': models.geometry(new)}
    stream = io.BytesIO()
    torch.save({'metadata': metadata, 'state': bad}, stream)
    with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
        models.load_state(new, stream.getvalue(), metadata)

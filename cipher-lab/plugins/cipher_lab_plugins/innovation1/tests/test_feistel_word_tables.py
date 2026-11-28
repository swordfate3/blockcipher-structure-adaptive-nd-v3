"""Guarded checks for the two new fixed word tables."""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from cipher_lab_plugins.innovation1.config import load_config
from cipher_lab_plugins.innovation1.four_single import interleave_config as cfg
from cipher_lab_plugins.innovation1.four_single import models
from cipher_lab_plugins.innovation1.four_single.interleave_runtime import InterleaveRuntime
from cipher_lab_plugins.innovation1.single_pair.models import shared_logits
from torch import Tensor

from cipher_lab.models.shared_m.spec import STEM_NAMES
from cipher_lab.representations.tasks import public_features


def test_word_recipe_and_tables() -> None:
    from cipher_lab_plugins.innovation1.four_single.structure_cards import verified_word_tables

    recipe = load_config(Path('configs/innovation1-feistel-word-tables-10m.json'))
    assert recipe is cfg.WORD_INTERLEAVE
    assert cfg.official(cfg.WORD_INTERLEAVE, cfg.WORD_INTERLEAVE.run_id) is recipe
    new, old = recipe.to_dict(), cfg.FULL_INTERLEAVE.to_dict()
    for key in ('training', 'tasks', 'seeds', 'resources', 'optimizer', 'selection', 'data',
                'routing'):
        assert new[key] == old[key]
    tables = verified_word_tables()
    expected = {
        'SIMECK': (0, 5, 1, -8, -5, -2, -1, 2), 'SIMON': (0, 1, 8, 2, -8, -5, -2, -1),
    }
    for name, offsets in expected.items():
        assert new['structure_card'][name]['word']['offsets'] == list(offsets)
        assert tables[name] == tuple(
            tuple((i & 32) + ((i + d) & 31) for d in offsets) + (i ^ 32,) for i in range(64))


def test_model_word_selection_and_strict_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = object.__new__(InterleaveRuntime)
    runtime.contract = cfg.FULL_INTERLEAVE
    old = runtime._build()
    runtime.contract = cfg.WORD_INTERLEAVE
    new = runtime._build()
    assert dict(old.named_parameters()).keys() == dict(new.named_parameters()).keys()
    for name, param in new.named_parameters():
        assert torch.equal(param, old.get_parameter(name))
    expected = cfg.WORD_INTERLEAVE.to_dict()['geometry']['sha256']
    geometry = models.geometry(new)
    assert all(geometry[name] == value for name, value in expected.items())
    assert geometry == models.geometry(old) | {k: v for k, v in expected.items()
                                               if k in ('simeck_word', 'simon_word')}
    chosen: list[tuple[str, Tensor]] = []

    def gather(raw: Tensor, channel: Tensor, table: Tensor, *, stem_name: str) -> Tensor:
        chosen.append((stem_name, table))
        return raw.new_zeros((1, 4, 96, 64))

    def logits(shared: Any, slots: Tensor, public: Tensor, **kwargs: Any) -> Tensor:
        def ordinary(*_: Tensor) -> Tensor:
            return slots.new_zeros((1, 4, 96, 64))
        fake = SimpleNamespace(
            representation=lambda *a, **kw: SimpleNamespace(canonical=slots, channel=slots),
            stems=dict(word=ordinary, local=ordinary, nibble=ordinary), spec=shared.spec,
            suffix=lambda x: x,
            readout=SimpleNamespace(base=lambda x: x[:, :1], residual=lambda x: x[:, :1]))
        # Legacy nibble overrides are validated by the existing wiring tests.
        kwargs['nibble'] = None
        return cast(Tensor, shared_logits(fake, slots, public, **kwargs))

    monkeypatch.setattr(new, '_gather', gather)
    monkeypatch.setattr(models, 'shared_logits', logits)
    bits = torch.zeros((4, 1, 2, 64), dtype=torch.float32)
    for stem in ('word', 'nibble'):
        indices = torch.tensor([STEM_NAMES.index(s) for s in (stem, 'local')])
        monkeypatch.setattr(new, '_route', lambda _: SimpleNamespace(
            indices=indices, weights=torch.tensor([.75, .25])))
        routes = {t.cipher: new.route_weights(torch.tensor(public_features(t.cipher)))
                  for t in cfg.WORD_INTERLEAVE.tasks}
        lookups = cfg.WORD_INTERLEAVE.word_lookup(routes)
        for task in cfg.WORD_INTERLEAVE.tasks:
            chosen.clear()
            new(bits, torch.tensor(public_features(task.cipher)), rounds=task.rounds)
            if stem == 'word' and task.cipher in ('SIMON', 'SIMECK'):
                assert len(chosen) == 1 and chosen[0][0] == 'word'
                assert chosen[0][1] is new.get_buffer(task.cipher.lower() + '_word')
                assert lookups[task.cipher] == task.cipher.lower() + '_word'
            else:
                assert not chosen
                assert lookups[task.cipher] == (
                    'shared.stems.word.indices' if stem == 'word' else None)
    state = new.state_dict()
    state['simon_word'] = state['simon_word'].clone()
    state['simon_word'][0, 2] = 7
    stream = io.BytesIO()
    identity: dict[str, object] = {'geometry': geometry}
    torch.save(dict(metadata=identity, state=state), stream)
    with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
        models.load_state(new, stream.getvalue(), identity)

from __future__ import annotations

from typing import cast

import pytest
import torch
from cipher_lab_plugins.innovation1.prefix_bridge.model import (
    PrefixSharedM,
    PrefixStem,
    build,
    initial_sha256,
)
from torch import Tensor, nn


def test_actual_model_shapes_initial_identity_and_no_alias() -> None:
    # CPU initialization and read-only state inspection only, no forward.
    first, second = build(31), build(31)
    assert isinstance(first, PrefixSharedM)
    assert sum(t.numel() for t in first.parameters()) == 1095285
    first_state, second_state = first.state_dict(), second.state_dict()
    assert initial_sha256(first) == initial_sha256(second)
    assert initial_sha256(first) != initial_sha256(build(32))
    assert all(first_state[k].data_ptr() != second_state[k].data_ptr() for k in first_state)
    card = [0., 0., 1., .5, .625, .125, 1., .5, 0., 0., 0., 1., 0.]
    assert first_state["public"].tolist() == card
    for stem in ('local', 'word', 'nibble'):
        assert first_state[f'stems.{stem}.network.0.weight'].shape == (96, 54, 1)
        assert first_state[f'stems.{stem}.indices'].shape == (64, 9)
        assert torch.equal(first_state[f'stems.{stem}.network.0.weight'],
                           first_state['stems.local.network.0.weight'])
        norm = cast(nn.GroupNorm, cast(PrefixStem, first.stems[stem]).network[1])
        assert norm.num_groups == 8
    assert first_state['readout.base.weight'].shape == (1, 6144)
    assert first_state['router.gate.0.weight'].shape == (16, 13)
    assert [first_state[f'suffix.{i}.network.0.weight'].shape[-1]
            for i in range(5)] == [3, 5, 7, 9, 11]
    print('constructed model:', sum(t.numel() for t in first.parameters()),
          'parameters; initial SHA256:', initial_sha256(first))


def test_stem_double_gather_pre_norm_relation_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from cipher_lab_plugins.innovation1.prefix_bridge import model

    events: list[object] = []
    index_table, zero_relation, gathered_rel = object(), object(), object()
    gathered_s2, gathered_cr, gathered54, output = object(), object(), object(), object()

    class Raw:
        def __getitem__(self, at: tuple[object, ...]) -> object:
            assert at[:2] == (slice(None), slice(None))
            assert at[2] in (slice(None, 3), slice(3, None))
            return gathered_s2 if at[2] == slice(None, 3) else gathered_cr

    class Base:
        def __add__(self, other: object) -> object:
            assert other is gathered_rel
            events.append('before normalization')
            return output

    class Hidden:
        def reshape(self, *shape: int) -> object:
            assert shape == (-1, 4, 96, 64)
            events.append('four pairs')
            return output

    def gather(raw: object, indices: object) -> object:
        assert indices is index_table
        events.append(raw)
        return raw

    def gather_relation(channel: object, indices: object) -> object:
        assert channel is zero_relation and indices is index_table
        return gathered_rel

    def concat(features: tuple[object, object], *, dim: int) -> object:
        assert features == (gathered_s2, gathered_cr) and dim == 1
        return gathered54

    def first_conv(features: object) -> Base:
        assert features is gathered54
        return Base()

    def remainder(features: object) -> Hidden:
        assert features is output
        events.append('normalization and remainder')
        return Hidden()

    class Network:
        def __getitem__(self, at: object) -> object:
            return first_conv if at == 0 else remainder

    stem = SimpleNamespace(network=Network(), indices=index_table, channels=96,
                           relation_projection=lambda x: x)
    monkeypatch.setattr(model, 'gather_neighborhood', gather)
    monkeypatch.setattr(model, 'gather_relation', gather_relation)
    monkeypatch.setattr(model, 'torch', SimpleNamespace(cat=concat))
    assert PrefixStem.forward(cast(PrefixStem, stem), cast(Tensor, Raw()),
                              cast(Tensor, zero_relation)) is output
    assert events == [gathered_s2, gathered_cr, 'before normalization',
                      'normalization and remainder', 'four pairs']

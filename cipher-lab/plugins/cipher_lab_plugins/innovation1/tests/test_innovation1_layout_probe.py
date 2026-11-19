"""Public tables, CPU initialization metadata and explicit fake pair operators only."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest
import torch
from cipher_lab_plugins.innovation1 import models as production
from cipher_lab_plugins.innovation1.config import Recipe, Seeds
from plugin_isolation import forbidden
from torch import Tensor, nn

from cipher_lab.models.shared_m import SharedM
from cipher_lab.models.shared_m.components import GeometryStem
from cipher_lab.models.shared_m.router import PublicRouter
from cipher_lab.models.shared_m.spec import STEM_NAMES, SharedMSpec
from cipher_lab.representations.input import RepresentedInput
from cipher_lab.representations.neighborhoods import neighborhood_indices
from cipher_lab.representations.tasks import public_features


def test_public_tables_at_every_bit_and_slot() -> None:
    from cipher_lab_plugins.innovation1.layout_probe.layout import nibble_indices

    row, column = nibble_indices('row'), nibble_indices('column')
    assert torch.equal(row, neighborhood_indices('nibble'))
    scan = (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15)
    padded = (None, *scan, None)
    for bit in range(64):
        n, offset = divmod(bit, 4)
        k = scan.index(n)
        expected = [4 * neighbor + b if neighbor is not None and 0 <= b <= 3 else 64
                    for neighbor in padded[k:k + 3] for b in range(offset - 1, offset + 2)]
        assert column[bit].tolist() == expected
    for table in (row, column):
        assert table.shape == (64, 9) and table.dtype == torch.long
        assert table.device.type == 'cpu'
        assert table[:, 4].tolist() == list(range(64))
        assert table[:, 3:6].equal(row[:, 3:6])  # original within-nibble slots
    assert torch.equal(row == 64, column == 64)
    assert not torch.equal(row, column)
    # Continuous scanning links the end of one row/column to the next.
    assert row[4 * 3 + 1, 7].item() == 4 * 4 + 1
    assert column[4 * 12 + 1, 7].item() == 4 * 1 + 1
    with pytest.raises(ValueError, match='scan'):
        nibble_indices('grid')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_fixed_route_matches_card_device_and_dtype(dtype: torch.dtype) -> None:
    from cipher_lab_plugins.innovation1.layout_probe.models import FixedSkinnyRouter

    router = FixedSkinnyRouter()
    public = torch.tensor(public_features('SKINNY'), dtype=dtype, requires_grad=True)
    route = router.forward(public)
    assert route.indices.tolist() == [STEM_NAMES.index('nibble'), STEM_NAMES.index('word')]
    assert route.indices.dtype == torch.long
    assert route.weights.tolist() == [0.5, 0.5]
    assert route.weights.dtype == public.dtype
    assert route.indices.device == route.weights.device == public.device
    assert not route.weights.requires_grad
    assert not tuple(router.parameters()) and not router.state_dict()


def test_fixed_route_rejects_non_skinny_and_malformed_cards() -> None:
    from cipher_lab_plugins.innovation1.layout_probe.models import FixedSkinnyRouter

    router = FixedSkinnyRouter()
    for family in ('PRESENT', 'SIMON', 'SIMECK'):
        with pytest.raises(ValueError, match='SKINNY'):
            router.forward(torch.tensor(public_features(family)))
    public = torch.tensor(public_features('SKINNY'))
    near = public.clone()
    near[0] = 0.001
    for bad in (near, public[None, :], public.long(), public * float('nan')):
        with pytest.raises(ValueError, match='public'):
            router.forward(bad)


def test_pair_state_is_fresh_cpu_and_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.layout_probe import models as probe
    from cipher_lab_plugins.innovation1.layout_probe.layout import nibble_indices

    construct = production.construct
    seed = 71
    recipe = replace(Recipe(), seeds=replace(Seeds(), init=seed))
    before = torch.random.get_rng_state().clone()
    template = construct(recipe, 'M-best')  # CPU construction, no forward/load of artifacts
    calls: list[tuple[Recipe, str]] = []

    def counted(recipe: Recipe, slot: str) -> nn.Module:
        calls.append((recipe, slot))
        return construct(recipe, slot)

    monkeypatch.setattr(production, 'construct', counted)
    monkeypatch.setattr(production, 'TorchRuntime', forbidden)
    monkeypatch.setattr(production, 'decode', forbidden)
    monkeypatch.setattr(production, 'encode', forbidden)
    # Explicit CPU factory even when the caller uses a non-CPU default device.
    with torch.device('meta'):
        a, b = probe.build_pair(init_seed=seed)
    assert calls == [(recipe, 'M-best')]
    assert torch.equal(before, torch.random.get_rng_state())
    assert a is not b and a.shared is not b.shared and a.shared.router is not b.shared.router
    for arm in (a, b):
        assert isinstance(arm.shared.router, probe.FixedSkinnyRouter)
        assert sum(p.numel() for p in arm.parameters() if p.requires_grad) == 1_087_234
        assert all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in arm.parameters())
        assert not arm.shared.router.state_dict()
    assert sum(p.numel() for p in template.parameters()) == 1_087_509
    assert sum(p.numel() for p in cast(SharedM, template).router.parameters()) == 275

    left, right = a.state_dict(), b.state_dict()
    original = {f'shared.{name}': t for name, t in template.state_dict().items()
                if not name.startswith('router.')}
    assert left.keys() == right.keys() == original.keys()
    assert all(torch.equal(t, original[name]) for name, t in left.items())
    assert all(t.device.type == 'cpu' for t in (*left.values(), *right.values()))
    assert all(t.shape == right[k].shape and t.dtype == right[k].dtype for k, t in left.items())
    assert {k for k, t in left.items() if not torch.equal(t, right[k])} == {
        'shared.stems.nibble.indices'}
    for arm, direction in ((a, 'row'), (b, 'column')):
        stem = cast(GeometryStem, arm.shared.stems['nibble'])
        assert torch.equal(stem.indices, nibble_indices(direction))

    def storage_ids(model: nn.Module) -> set[int]:
        return {t.untyped_storage().data_ptr() for t in (*model.parameters(), *model.buffers())}

    assert storage_ids(a).isdisjoint(storage_ids(b))


class _Manual(nn.Module):
    def __init__(self, fn: Callable[..., object]) -> None:
        super().__init__()
        self.forward = fn  # explicit fake, not a numerical Torch layer


def test_pair_forward_wires_fixed_stems_and_never_pools() -> None:
    from cipher_lab_plugins.innovation1.layout_probe.models import LayoutProbe

    public = torch.tensor(public_features('SKINNY'))
    ids = torch.arange(8)
    bits = torch.zeros((8, 1, 2, 64), dtype=torch.uint8)
    bits[:, 0, 0, :3] = ((ids[:, None] >> torch.arange(3)) & 1).to(bits.dtype)
    events: list[str] = []

    def represent(slots: Tensor, card: Tensor, *, rounds: int) -> RepresentedInput:
        events.append('representation')
        assert card is public and rounds == 8 and slots.shape == (2, 4, 2, 64)
        assert torch.equal(slots.reshape_as(bits), bits)
        x = (slots[:, :, 0, :3].float() * torch.tensor([1., 2., 4.])).sum(-1)
        x = x.reshape(2, 4, 1, 1).expand(-1, -1, 1, 64)
        return RepresentedInput(x, x + 3)

    def nibble(raw: Tensor, channel: Tensor) -> Tensor:
        events.append('nibble')
        return 2 * raw + channel

    def word(raw: Tensor, channel: Tensor) -> Tensor:
        events.append('word')
        return raw + 2 * channel

    fake = _Manual(forbidden)  # stock M's grouped forward must not be used
    fake.register_parameter('metadata_anchor', nn.Parameter(torch.zeros(())))
    fake.representation = _Manual(represent)
    fake.router = PublicRouter(1)  # constructed, never used
    fake.stems = nn.ModuleDict(dict(nibble=_Manual(nibble), word=_Manual(word),
                                    local=_Manual(forbidden)))
    setattr(fake, 'spec', SharedMSpec(1, 1, 1, (3,), 1, 1))
    fake.suffix = _Manual(lambda x: x + 1)
    fake.readout = _Manual(forbidden)
    fake.readout.base = _Manual(lambda x: 2 * x[:, :1])
    fake.readout.residual = _Manual(lambda x: x[:, :1].square())
    arm = LayoutProbe(cast(SharedM, fake))
    out = arm.forward(bits, public, rounds=8)
    hidden = 3 * ids.float().reshape(8, 1) + 5.5
    assert out.shape == (8, 1) and torch.equal(out, 2 * hidden + hidden.square())
    assert events == ['representation', 'nibble', 'word']
    events.clear()
    for bad in (bits[:0], bits[:3], bits[:, 0], bits.reshape(2, 4, 2, 64)):
        with pytest.raises(ValueError, match='SINGLE_PAIR_SHAPE'):
            arm.forward(bad, public, rounds=8)
    with pytest.raises(ValueError, match='SKINNY'):
        arm.forward(bits, torch.tensor(public_features('PRESENT')), rounds=8)
    with pytest.raises(ValueError, match='dtype'):
        arm.forward(bits, public.double(), rounds=8)
    with pytest.raises(ValueError, match='device'):
        arm.forward(bits.to('meta'), public, rounds=8)
    assert events == []

from __future__ import annotations

from typing import cast

import pytest
import torch


def test_constructor_exact_geometry_and_state() -> None:
    from cipher_lab_plugins.innovation1.config import Recipe, Seeds
    from cipher_lab_plugins.innovation1.four_single.models import SinglePairM
    from cipher_lab_plugins.innovation1.models import construct
    from cipher_lab_plugins.innovation1.present_card_fresh import model

    from cipher_lab.models.shared_m.components import GeometryStem
    from cipher_lab.representations.neighborhoods import neighborhood_indices

    before = torch.random.get_rng_state()
    net = model.build()
    assert torch.equal(torch.random.get_rng_state(), before)
    base = SinglePairM(construct(Recipe(seeds=Seeds(init=26111741)), 'M-best'))
    assert sum(p.numel() for p in net.parameters()) == 1_087_509
    spec = net.pair.shared.spec
    assert (spec.stem_width, spec.channels, spec.norm_groups, spec.suffix_kernels,
            spec.head_hidden, spec.router_hidden) == (96, 96, 8, (3, 5, 7, 9, 11), 64, 16)
    assert all(p.requires_grad == (not name.startswith('pair.shared.router.'))
               for name, p in net.named_parameters())
    assert net.pair.route_weights(net.public) == {'nibble': .5, 'word': .5}
    assert model.geometry(net)['shared.stems.nibble.indices'] == (
        'e3d8578406b90440c74f938c2cb110e12160b8a8342fae18ecb2a5dea0d4f4ae')
    word = cast(GeometryStem, net.pair.shared.stems['word'])
    assert torch.equal(word.indices, neighborhood_indices('word'))
    for name, value in base.state_dict().items():
        if name != 'shared.stems.nibble.indices':
            assert torch.equal(value, net.pair.state_dict()[name]), name


def test_input_domain_with_stubbed_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.four_single import models as pair
    from cipher_lab_plugins.innovation1.present_card_fresh import model

    net = model.build()
    seen: list[int] = []

    def logits(shared: object, slots: torch.Tensor, public: torch.Tensor, **kwargs: object
               ) -> torch.Tensor:
        assert shared is net.pair.shared and public is net.public
        assert kwargs == {'rounds': 7, 'nibble': None, 'route_for': net.pair._route}
        seen.append(slots.shape[0])
        return slots[:, :, 0, 0].reshape(-1, 1)

    monkeypatch.setattr(pair, 'shared_logits', logits)
    bits = torch.zeros((8, 1, 2, 64))
    bits[1::2, 0, 0, 0] = 1
    assert torch.equal(net(bits), bits[:, :, 0, 0])
    assert seen == [2]
    for invalid in (bits[:0], bits[:3], bits.double(), bits.reshape(2, 4, 2, 64),
                    bits[:, :, :, :32], bits + 2, bits + float('nan'), bits.to('meta')):
        with pytest.raises(ValueError):
            net(invalid)
    assert seen == [2]


def test_original_pair_operators_never_pool_or_call_learned_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typing import Any

    from cipher_lab_plugins.innovation1.present_card_fresh.model import build
    from torch import nn

    from cipher_lab.representations.input import RepresentedInput, canonical_input
    from cipher_lab.representations.tasks import public_features

    net = build()
    shared = net.pair.shared
    calls: list[str] = []

    def blocked(*args: object, **kwargs: object) -> Any:
        pytest.fail('unexpected learned route or group readout')

    monkeypatch.setattr(shared.router, 'forward', blocked)
    monkeypatch.setattr(shared.readout, 'forward', blocked)
    monkeypatch.setattr(shared.stems['local'], 'forward', blocked)

    def represent(bits: torch.Tensor, public: torch.Tensor, *, rounds: int) -> RepresentedInput:
        assert bits.shape == (2, 4, 2, 64) and rounds == 7
        assert public.tolist() == pytest.approx(public_features('PRESENT'))
        calls.append('represent')
        return RepresentedInput(canonical_input(bits), bits.new_zeros((2, 4, 1, 64)))

    def fake_stem(name: str, factor: int) -> Any:
        def stem(raw: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
            assert raw.shape == (2, 4, 3, 64) and not bool(relation.any())
            assert torch.equal(raw[:, :, 2], (raw[:, :, 0] != raw[:, :, 1]).float())
            calls.append(name)
            return (factor * raw[:, :, :1]).expand(2, 4, 96, 64)
        return stem

    def suffix(value: torch.Tensor) -> torch.Tensor:
        assert value.shape == (8, 96, 64)
        calls.append('suffix')
        return value + 1

    def fake_head(name: str, factor: int) -> Any:
        def head(value: torch.Tensor) -> torch.Tensor:
            assert value.shape == (8, 96 * 64)
            calls.append(name)
            return factor * value[:, :1]
        return head

    monkeypatch.setattr(shared.representation, 'forward', represent)
    monkeypatch.setattr(shared.stems['nibble'], 'forward', fake_stem('nibble', 1))
    monkeypatch.setattr(shared.stems['word'], 'forward', fake_stem('word', 3))
    monkeypatch.setattr(shared.suffix, 'forward', suffix)
    monkeypatch.setattr(cast(nn.Module, shared.readout.base), 'forward', fake_head('base', 1))
    monkeypatch.setattr(shared.readout.residual, 'forward', fake_head('residual', 2))
    bits = torch.zeros((8, 1, 2, 64))
    bits[3, 0, 0, 0] = 1
    bits[7, 0, 1, 0] = 1
    assert net(bits).flatten().tolist() == [3, 3, 3, 9, 3, 3, 3, 3]
    assert calls == ['represent', 'nibble', 'word', 'suffix', 'base', 'residual']

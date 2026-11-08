"""Public hand vectors and small CPU forward/backward, never encryption or fitting."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor, nn

from cipher_lab.models import SharedM
from cipher_lab.models.shared_m.components import GeometryStem
from cipher_lab.models.shared_m.router import PublicRouter
from cipher_lab.representations.input import InputRepresentation, public_backstep
from cipher_lab.representations.neighborhoods import (
    gather_neighborhood,
    gather_relation,
    neighborhood_indices,
)
from cipher_lab.representations.tasks import public_features, validate_public
from cipher_lab.training_config import TASK_FAMILIES, SharedMSpec, parse_shared_m

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT = SharedMSpec(96, 96, 8, (3, 5, 7, 9, 11), 64, 16)
SMALL = SharedMSpec(8, 16, 2, (1, 3), 4, 3)


def card(cipher: str) -> Tensor:
    return torch.tensor(public_features(cipher))


def bits() -> Tensor:
    return (torch.arange(512).reshape(1, 4, 2, 64) % 3 == 0).float()


def word_bits(word: int) -> Tensor:
    return torch.tensor([(word >> shift) & 1 for shift in range(63, -1, -1)],
                        dtype=torch.float32).repeat(1, 4, 2, 1)


@pytest.mark.parametrize(('cipher', 'expected'), [
    ('SIMECK', (0, 1, 0, .5, 1, 1, .125, 0, 0, .15625, .03125, 0, 0)),
    ('PRESENT', (0, 0, 1, .5, .625, .125, 1, .5, 0, 0, 0, 1, 0)),
    ('SIMON', (0, 1, 0, .5, 1, 1, .125, 0, .03125, .25, .0625, 0, 0)),
    ('SKINNY', (0, 0, 1, .5, .5, .125, 1, .5, 0, 0, 0, 0, 1)),
])
def test_public13_exact_cards(cipher: str, expected: tuple[float, ...]) -> None:
    assert public_features(cipher) == expected
    validate_public(card(cipher))
    assert card(cipher).shape == (13,)


@pytest.mark.parametrize('bad', [torch.zeros(13), torch.zeros(4, 13), torch.zeros(14),
                               torch.full((13,), float('nan')),
                               torch.full((13,), float('inf')),
                               torch.zeros(13, dtype=torch.long), torch.zeros(13).to_sparse()])
def test_reject_public_shape_tampering_nonfinite(bad: Tensor) -> None:
    with pytest.raises(ValueError):
        validate_public(bad)


def test_reject_near_card_and_unknown_family() -> None:
    bad = card('SIMON').double()
    bad[8] += 1e-9
    with pytest.raises(ValueError):
        validate_public(bad)
    with pytest.raises(ValueError):
        public_features('unknown')


@pytest.mark.parametrize(('kind', 'row', 'expected'), [
    ('local', 0, [64, 64, 64, 64, 0, 1, 2, 3, 4]),
    ('word', 0, [24, 27, 30, 31, 0, 1, 2, 5, 32]),
    ('nibble', 0, [64, 64, 64, 64, 0, 1, 64, 4, 5]),
    ('local', 63, [59, 60, 61, 62, 63, 64, 64, 64, 64]),
])
def test_neighborhood_order_padding(kind: str, row: int, expected: list[int]) -> None:
    indices = neighborhood_indices(kind)
    assert indices.shape == (64, 9) and indices[row].tolist() == expected
    raw = torch.arange(3 * 64).reshape(1, 1, 3, 64).repeat(1, 4, 1, 1).float()
    gathered = gather_neighborhood(raw, indices)
    assert gathered.shape == (4, 27, 64)
    assert gathered[0, :9, row].tolist() == [float(i) if i < 64 else 0. for i in expected]
    relation = gather_relation(raw[:, :, :1], indices)
    assert relation.shape == (4, 9, 64)


def test_bad_neighborhood_rejected() -> None:
    with pytest.raises(ValueError):
        neighborhood_indices('unknown')
    with pytest.raises(ValueError):
        gather_neighborhood(torch.zeros(1, 4, 3, 64), torch.zeros(64, 9))


@pytest.mark.parametrize(('rounds', 'zero_word', 'pattern_word'), [
    # rc=01,3d,3b,37,38. For 0123456789abcdef inverse MC gives
    # 456701238888cccc, inverse SR gives 456712308888cccc, then
    # public RC subtraction at cells 0/4/8 and inverse Sbox ONLY cells 8..15.
    # inverse[10]=5, inverse[8]=9, inverse[12]=0; the top half is NOT Sboxed.
    (1, 0x1000000063333333, 0x5567123059990000),
    (7, 0xd000300063333333, 0x9567223059990000),
    (8, 0xb000300063333333, 0xf567223059990000),
    (9, 0x7000300063333333, 0x3567223059990000),
    (32, 0x8000300063333333, 0xc567223059990000),
])
def test_public_backstep_hand_vectors(rounds: int, zero_word: int, pattern_word: int) -> None:
    assert torch.equal(public_backstep(word_bits(0), rounds), word_bits(zero_word))
    assert torch.equal(public_backstep(word_bits(0x0123456789abcdef), rounds),
                       word_bits(pattern_word))
    byte = word_bits(0).to(torch.uint8)
    assert public_backstep(byte, rounds).dtype == torch.uint8


@pytest.mark.parametrize('rounds', [0, 33, True, 1.5])
def test_backstep_invalid_rounds(rounds: int) -> None:
    with pytest.raises(ValueError):
        public_backstep(bits(), rounds)


def test_representation_applies_one_backstep_then_xor_and_relation() -> None:
    represented = InputRepresentation()(word_bits(0), card('SKINNY'), rounds=8)
    assert torch.equal(represented.canonical[:, :, :2], word_bits(0xb000300063333333))
    assert represented.canonical.shape == (1, 4, 3, 64)
    assert not represented.canonical[:, :, 2].any()
    assert represented.channel.shape == (1, 4, 1, 64) and not represented.channel.any()
    # Artificial low word has only MSB set: SIMON f(0x80000000)=rol2=2;
    # before=2, f(before)=8, relation words=2 and 80000008.
    pair = word_bits(0)
    pair[:, :, 0] = word_bits(0x80000000)[:, :, 0]
    simon = InputRepresentation()(pair, card('SIMON'), rounds=13)
    assert torch.equal(simon.canonical[:, :, :2], pair)
    assert torch.equal(simon.canonical[:, :, 2:3], word_bits(0x80000000)[:, :, :1])
    assert torch.equal(simon.channel, word_bits(0x0000000280000008)[:, :, :1])
    present = InputRepresentation()(pair, card('PRESENT'), rounds=7)
    assert not present.channel.any()


def test_public_router_stable_top2_and_gate_gradient() -> None:
    router = PublicRouter(3)
    with torch.no_grad():
        for p in router.parameters():
            p.zero_()
    route = router(card('SIMON'))
    assert route.indices.tolist() == [0, 1]
    assert route.weights.tolist() == [.5, .5]
    route.weights[0].backward()
    final = cast(nn.Linear, router.gate[2])
    assert final.bias is not None and final.bias.grad is not None
    assert final.bias.grad.tolist() == [.25, -.25, 0.]
    with torch.no_grad():
        final.bias.copy_(torch.tensor([1., 3., 2.]))
    assert router(card('SIMON')).indices.tolist() == [1, 2]
    assert cast(nn.Linear, router.gate[0]).in_features == 13


def test_default_parameter_count_initialization_and_example() -> None:
    model = SharedM(DEFAULT)
    assert sum(p.numel() for p in model.parameters()) == 1087509
    assert len(model.suffix) == 5
    local = model.stems['local'].state_dict()
    for name in ('word', 'nibble'):
        for key, value in model.stems[name].state_dict().items():
            if key != 'indices':
                assert torch.equal(value, local[key])
    for stem in model.stems.values():
        assert not cast(GeometryStem, stem).relation_projection.weight.any()
    assert not cast(nn.Linear, model.readout.residual[2]).weight.any()
    recipe = parse_shared_m(json.loads((ROOT / 'configs/shared-m-example.json').read_text()))
    assert recipe.model == DEFAULT
    assert (recipe.training.train, recipe.training.dev, recipe.training.test,
            recipe.training.batch_size, recipe.training.epochs) == (2, 2, 2, 2, 1)


@pytest.mark.parametrize('family', TASK_FAMILIES, ids=lambda x: x.cipher)
def test_shared_forward_backward_no_update(family: object) -> None:
    cipher = cast(str, getattr(family, 'cipher'))
    model = SharedM(SMALL)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    logits = model(bits(), card(cipher), rounds=7)
    assert logits.shape == (1, 1) and torch.isfinite(logits).all()
    logits.sum().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    assert model.router.gate[2].weight.grad is not None
    assert any(p.grad is not None for p in model.stems['local'].parameters()) or any(
        p.grad is not None for p in model.stems['word'].parameters())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    model.eval()
    assert torch.equal(logits, model(bits(), card(cipher), rounds=7))


def test_rounds_affect_representation_not_route_or_parameters() -> None:
    model = SharedM(SMALL)
    public = card('SKINNY')
    indices = model.router(public).indices
    weights = model.router(public).weights
    count = sum(p.numel() for p in model.parameters())
    a = model.representation(bits(), public, rounds=1).canonical
    b = model.representation(bits(), public, rounds=9).canonical
    assert not torch.equal(a, b)
    for rounds in (1, 7, 8, 9, 32):
        assert model(bits(), public, rounds=rounds).shape == (1, 1)
        route = model.router(public)
        assert torch.equal(indices, route.indices) and torch.equal(weights, route.weights)
    assert sum(p.numel() for p in model.parameters()) == count


@pytest.mark.parametrize('bad', [torch.zeros(1, 512), torch.zeros(0, 4, 2, 64),
                               torch.zeros(1, 3, 2, 64), torch.full((1, 4, 2, 64), .2),
                               torch.full((1, 4, 2, 64), float('nan')),
                               torch.zeros(1, 4, 2, 64).to_sparse()])
def test_shared_invalid_bits(bad: Tensor) -> None:
    with pytest.raises(ValueError):
        SharedM(SMALL)(bad, card('SIMON'), rounds=13)


def test_spec_identity_and_family_round_validation() -> None:
    with pytest.raises(ValueError):
        SharedM(cast(SharedMSpec, {}))
    for cipher, rounds in [('PRESENT', 32), ('SIMON', 45), ('SIMECK', 0), ('SKINNY', True)]:
        with pytest.raises(ValueError):
            SharedM(SMALL)(bits(), card(cipher), rounds=rounds)


def test_pair_readout_nonlinearity_precedes_pair_average() -> None:
    model = SharedM(SMALL)
    head = model.readout
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        first = cast(nn.Linear, head.residual[0])
        last = cast(nn.Linear, head.residual[2])
        first.weight[0, 0] = 1
        last.weight[0, 0] = 1
    hidden = torch.zeros(1, 4, SMALL.channels, 64)
    hidden[0, :, 0, 0] = torch.tensor([-2., 2., -2., 2.])
    assert head(hidden).item() == 1.  # group-head would yield 0


def test_model_and_representation_import_boundaries() -> None:
    for directory in ('models', 'representations'):
        for path in (ROOT / 'src/cipher_lab' / directory).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or '']
                    assert not node.level
                for module in modules:
                    assert module.split('.')[0] in {'__future__', 'torch', 'typing', 'copy',
                                                    'dataclasses', 'cipher_lab'}
                    if module.startswith('cipher_lab.'):
                        assert module.split('.')[1] in {'models', 'representations',
                                                       'training_config'}


def test_task_route_sample_independence_batch_alignment_pair_symmetry() -> None:
    model = SharedM(SMALL).eval()
    public = card('SIMON')
    first, second = bits(), 1 - bits()
    route = model.router(public)
    joined = model(torch.cat((first, second)), public, rounds=13)
    separate = torch.cat((model(first, public, rounds=13), model(second, public, rounds=13)))
    assert joined.shape == (2, 1) and torch.allclose(joined, separate, atol=1e-6)
    permuted = model(first[:, [3, 1, 0, 2]], public, rounds=13)
    assert torch.allclose(joined[:1], permuted, atol=1e-6)
    assert torch.equal(route.indices, model.router(public).indices)
    assert torch.equal(route.weights, model.router(public).weights)
    joined.sum().backward()
    selected = {('local', 'word', 'nibble')[i] for i in route.indices.tolist()}
    for name, stem in model.stems.items():
        gradients = [p.grad for p in stem.parameters()]
        if name in selected:
            assert all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
        else:
            assert all(g is None for g in gradients)


def test_default_forward_and_public_dtype_rejection() -> None:
    model = SharedM(DEFAULT)
    with torch.no_grad():
        logits = model(bits(), card('SKINNY'), rounds=8)
    assert logits.shape == (1, 1) and torch.isfinite(logits).all()
    with pytest.raises(ValueError, match='dtype'):
        model(bits(), card('SIMON').double(), rounds=13)


def test_reject_corrupt_router_or_output_without_updates() -> None:
    router = PublicRouter(3)
    with torch.no_grad():
        cast(nn.Linear, router.gate[2]).weight.fill_(float('nan'))
    with pytest.raises(ValueError, match='finite'):
        router(card('SIMON'))
    model = SharedM(SMALL)
    with torch.no_grad():
        model.readout.base.weight.fill_(float('nan'))
    with pytest.raises(ValueError, match='finite'):
        model(bits(), card('SIMON'), rounds=13)

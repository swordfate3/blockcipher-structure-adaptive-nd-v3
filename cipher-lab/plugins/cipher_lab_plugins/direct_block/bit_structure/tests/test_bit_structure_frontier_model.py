import hashlib
import json

import pytest
import torch
from cipher_lab_plugins.direct_block.bit_structure.geometry import (
    LEVELS,
    SHUFFLE,
    SUPPORT,
    graph,
    shuffled_binding,
)
from cipher_lab_plugins.direct_block.bit_structure.model import build


@pytest.mark.parametrize(
    ('rounds', 'leaves', 'nodes', 'width', 'structured_count', 'mlp_count'),
    [
        (4, 24, 20, 187, 28528, 28535),
        (5, 30, 34, 244, 35965, 35945),
        (6, 32, 50, 302, 43476, 43485),
        (7, 32, 66, 360, 50987, 51025),
        (8, 32, 82, 417, 58498, 58435),
    ],
)
def test_frontier_models(
    rounds: int, leaves: int, nodes: int, width: int, structured_count: int, mlp_count: int
) -> None:
    support, levels = graph(rounds)
    binding = shuffled_binding(rounds)
    assert len(support) == leaves
    assert len(levels) == rounds
    assert sum(map(len, levels)) == nodes
    parents = ((rounds - 1, 1), (rounds - 1, 8), (rounds - 1, 2), (rounds - 2, 0))
    assert levels[-1] == (((rounds, 0), parents),)
    assert set(binding) == set(binding.values()) == set(support)
    for word in (0, 1):
        ordered = sorted(
            (bit for bit in support if bit // 16 == word),
            key=lambda bit: hashlib.sha256(
                f'simon32-r{rounds}-msb0-bit0-control-v1/{bit}'.encode()
            ).digest(),
        )
        assert all(binding[bit] != bit for bit in ordered)
        assert [binding[bit] for bit in ordered] == ordered[1:] + ordered[:1]

    rng = torch.get_rng_state()
    mlp = build('mlp', 71, rounds=rounds)
    correct = build('correct', 71, rounds=rounds)
    shuffled = build('shuffled', 71, rounds=rounds)
    assert torch.equal(rng, torch.get_rng_state())
    assert mlp.state_dict()['2.weight'].shape == (width, 128)
    for model, count in (
        (mlp, mlp_count), (correct, structured_count), (shuffled, structured_count)
    ):
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == count
    state = correct.state_dict()
    control_state = shuffled.state_dict()
    assert state.keys() == control_state.keys() == dict(correct.named_parameters()).keys()
    for name, value in state.items():
        assert torch.equal(control_state[name], value)
        assert control_state[name].data_ptr() != value.data_ptr()
    shuffled.load_state_dict(state, strict=True)
    bits = torch.eye(32)[:2]
    rebound = bits.clone()
    rebound[:, list(support)] = bits[:, [binding[bit] for bit in support]]
    with torch.no_grad():
        for model in (mlp, correct, shuffled):
            assert model(bits).shape == (2, 1)
            assert model(bits[:0]).shape == (0, 1)
        assert torch.equal(shuffled(bits), correct(rebound))


@pytest.mark.parametrize(
    ('arm', 'expected'),
    [
        ('mlp', '566b62d346538fb74f3c477075195daf5f18e5af7b8df42b5606b8b2a457b411'),
        ('correct', 'a3f5c44495f8aee3eaed924ac64cd27bfaedb5d1eb07d1c6532761d674e17ee6'),
        ('shuffled', 'a3f5c44495f8aee3eaed924ac64cd27bfaedb5d1eb07d1c6532761d674e17ee6'),
    ],
)
def test_legacy_initial_state(arm: str, expected: str) -> None:
    assert graph(3) == (SUPPORT, LEVELS)
    assert shuffled_binding(3) == SHUFFLE
    for model in (build(arm, 71), build(arm, 71, rounds=3)):
        digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            digest.update(repr((name, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        assert digest.hexdigest() == expected
        if arm != 'mlp':
            wiring = {name: tensor.tolist() for name, tensor in model.named_buffers()}
            raw = json.dumps(wiring, sort_keys=True, separators=(',', ':')).encode()
            expected_wiring = {
                'correct': '44fc2fbefeab49cfcfaee6636234cb92d7aafbe151cbe3090fa4be52703894d5',
                'shuffled': '5b8a632c25ec2ad0b6239be166379ae3c09c31b2f10ac1dd88786bc2ab64cd29',
            }
            assert hashlib.sha256(raw).hexdigest() == expected_wiring[arm]


@pytest.mark.parametrize('rounds', [2, 9])
def test_unsupported_rounds(rounds: int) -> None:
    with pytest.raises(ValueError, match='rounds'):
        graph(rounds)
    with pytest.raises(ValueError, match='rounds'):
        shuffled_binding(rounds)
    for arm in ('mlp', 'correct', 'shuffled'):
        with pytest.raises(ValueError, match='rounds'):
            build(arm, 71, rounds=rounds)

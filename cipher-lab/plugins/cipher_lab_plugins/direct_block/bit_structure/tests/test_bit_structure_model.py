import hashlib
import json

import pytest
import torch
from cipher_lab_plugins.direct_block.bit_structure.geometry import LEVELS, SHUFFLE, SUPPORT
from cipher_lab_plugins.direct_block.bit_structure.model import build


def test_public_graph() -> None:
    assert SUPPORT == (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 16, 18, 19, 20, 25, 26)
    assert tuple(map(len, LEVELS)) == (6, 3, 1)
    # IDs are (round, MSB0 position); each row has four ordered parents.
    assert dict(LEVELS[2]) == {(3, 0): ((2, 1), (2, 8), (2, 2), (1, 0))}
    assert dict(LEVELS[1]) == {
        (2, 1): ((1, 2), (1, 9), (1, 3), (0, 1)),
        (2, 2): ((1, 3), (1, 10), (1, 4), (0, 2)),
        (2, 8): ((1, 9), (1, 0), (1, 10), (0, 8)),
    }
    assert dict(LEVELS[0]) == {
        (1, 0): ((0, 1), (0, 8), (0, 2), (0, 16)),
        (1, 2): ((0, 3), (0, 10), (0, 4), (0, 18)),
        (1, 3): ((0, 4), (0, 11), (0, 5), (0, 19)),
        (1, 4): ((0, 5), (0, 12), (0, 6), (0, 20)),
        (1, 9): ((0, 10), (0, 1), (0, 11), (0, 25)),
        (1, 10): ((0, 11), (0, 2), (0, 12), (0, 26)),
    }


def test_shuffle_identity() -> None:
    assert set(SHUFFLE) == set(SHUFFLE.values()) == set(SUPPORT)
    assert all(src != dst and src // 16 == dst // 16 for src, dst in SHUFFLE.items())
    encoded = json.dumps(dict(SHUFFLE), sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(encoded).hexdigest() == (
        'db685b4b3dbafcbcf33b92ed6a7a1840d692afbfd2456c0cb74381856e10b3eb'
    )


@pytest.mark.parametrize(
    ('arm', 'count'), [('mlp', 21255), ('correct', 21239), ('shuffled', 21239)]
)
def test_model_shape_and_count(arm: str, count: int) -> None:
    model = build(arm, 71)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == count
    with torch.no_grad():
        assert model(torch.zeros(2, 32)).shape == (2, 1)
        assert model(torch.zeros(0, 32)).shape == (0, 1)
        with pytest.raises(ValueError, match='32'):
            model(torch.zeros(2, 33))


def test_initial_weights_and_binding_isolation() -> None:
    rng = torch.get_rng_state()
    correct = build('correct', 71)
    shuffled = build('shuffled', 71)
    other = build('correct', 73)
    assert torch.equal(rng, torch.get_rng_state())
    state = correct.state_dict()
    assert state.keys() == shuffled.state_dict().keys() == dict(correct.named_parameters()).keys()
    for name, value in shuffled.state_dict().items():
        assert torch.equal(value, state[name])
        assert value.data_ptr() != state[name].data_ptr()
    assert any(not torch.equal(value, other.state_dict()[name]) for name, value in state.items())
    shuffled.load_state_dict(state, strict=True)
    bits = torch.zeros(2, 32)
    bits[0, 1] = bits[1, 16] = 1
    rebound = bits.clone()
    rebound[:, list(SUPPORT)] = bits[:, [SHUFFLE[pos] for pos in SUPPORT]]
    unused = bits.clone()
    unused[:, sorted(set(range(32)) - set(SUPPORT))] = 1
    with torch.no_grad():
        assert torch.equal(shuffled(bits), correct(rebound))
        assert torch.equal(correct(bits), correct(unused))
        assert not torch.equal(correct(bits), shuffled(bits))


def test_unknown_arm() -> None:
    with pytest.raises(ValueError, match='unknown'):
        build('unknown', 71)

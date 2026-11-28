"""Small CPU checks for the full-word model and its shuffled control."""
from hashlib import sha256
from typing import cast

import pytest
import torch
from cipher_lab_plugins.direct_block.bit_structure.model import _Level
from torch import Tensor, nn


def _build(arm: str, seed: int, *, rounds: int) -> nn.Module:
    from cipher_lab_plugins.direct_block.full32_frontier.model import build

    return build(arm, seed, rounds=rounds)


@pytest.mark.parametrize(('rounds', 'count'), [(3, 22645), (4, 30156), (5, 37667), (6, 45178)])
@torch.no_grad()
def test_matched_initialization_and_fixed_input_binding(rounds: int, count: int) -> None:
    expected = list(range(32))
    for start in (0, 16):
        order = sorted(range(start, start + 16), key=lambda col: sha256(
            f'simon32-full32-frontier-shuffle-v1/{col}'.encode()
        ).digest())
        for source, target in zip(order, order[1:] + order[:1], strict=True):
            expected[source] = target

    rng = torch.random.get_rng_state()
    correct = _build('correct', 123, rounds=rounds)
    shuffled = _build('shuffled', 123, rounds=rounds)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert sum(p.numel() for p in correct.parameters()) == count
    assert sum(p.numel() for p in shuffled.parameters()) == count
    state = correct.state_dict()
    control_state = shuffled.state_dict()
    assert state.keys() == control_state.keys()
    for name, value in state.items():
        torch.testing.assert_close(value, control_state[name], rtol=0, atol=0)

    bits = torch.arange(32, dtype=torch.float32).reshape(1, 32)
    for model, columns in ((correct, list(range(32))), (shuffled, expected)):
        buffers = dict(model.named_buffers())
        (binding_name,) = [name for name, buf in buffers.items() if buf.shape == (32,)]
        assert binding_name not in model.state_dict()
        assert buffers[binding_name].tolist() == columns
        (embed,) = [m for m in model.modules() if isinstance(m, nn.Linear) and m.in_features == 1]
        seen: list[Tensor] = []
        with embed.register_forward_pre_hook(lambda _, args: seen.append(args[0])):
            assert model(bits).shape == (1, 32)
        (inputs,) = seen
        torch.testing.assert_close(inputs, bits[:, columns].unsqueeze(-1))


@pytest.mark.parametrize('arm', ['correct', 'shuffled'])
@torch.no_grad()
def test_msb0_routing_and_feistel_readout(arm: str) -> None:
    from contextlib import ExitStack

    model = _build(arm, 42, rounds=4)
    levels = [m for m in model.modules() if isinstance(m, _Level)]
    assert len(levels) == 4
    (embed,) = [m for m in model.modules() if isinstance(m, nn.Linear) and m.in_features == 1]
    (head,) = [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == 1]
    embedded: list[Tensor] = []
    parents: list[Tensor] = []
    highs: list[Tensor] = []
    read_inputs: list[Tensor] = []
    scores: list[Tensor] = []

    def record_level(module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> None:
        assert isinstance(module, _Level)
        parents.append(args[0][:, module.indices])
        highs.append(output)

    def record_head(_: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> None:
        read_inputs.append(args[0])
        scores.append(output)

    bits = torch.arange(64, dtype=torch.float32).reshape(2, 32) / 64
    with ExitStack() as hooks:
        hooks.enter_context(embed.register_forward_hook(lambda _, args, out: embedded.append(out)))
        hooks.enter_context(head.register_forward_hook(record_head))
        for level in levels:
            hooks.enter_context(level.register_forward_hook(record_level))
        logits = model(bits)

    (leaves,) = embedded
    high, low = leaves[:, :16], leaves[:, 16:]
    for actual, next_high in zip(parents, highs, strict=True):
        expected = torch.stack([high.roll(-1, 1), high.roll(-8, 1), high.roll(-2, 1), low], dim=2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        high, low = next_high, high

    (representations,) = read_inputs
    (readout,) = scores
    torch.testing.assert_close(representations[:, :16], highs[-1], rtol=0, atol=0)
    torch.testing.assert_close(representations[:, 16:], highs[-2], rtol=0, atol=0)
    assert logits.shape == (2, 32)
    torch.testing.assert_close(logits, readout.squeeze(-1), rtol=0, atol=0)


@pytest.mark.parametrize(('arm', 'rounds'),
                         [('mlp', 3), ('correct', 2), ('shuffled', 7), ('correct', 3.0)])
def test_invalid_model_configuration(arm: str, rounds: object) -> None:
    with pytest.raises(ValueError):
        _build(arm, 0, rounds=cast(int, rounds))

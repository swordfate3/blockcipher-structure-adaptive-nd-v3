"""Synthetic integration coverage for the extracted shared-M model."""

import torch

from cipher_experiments.ciphers.tasks import public_features
from cipher_experiments.models.shared_m import ModelConfig, RouterConfig, SharedDistinguisher


def test_shared_m_forward_and_public_only_router_gradient() -> None:
    torch.manual_seed(7)
    config = ModelConfig(
        stem_width=8,
        channels=8,
        norm_groups=2,
        suffix_kernels=(3,),
        head_hidden=4,
        router=RouterConfig(hidden_width=4),
    )
    model = SharedDistinguisher(config)
    bits = torch.randint(0, 2, (2, 4, 2, 64), dtype=torch.float32)
    public = torch.tensor(public_features("simon64"), dtype=torch.float32)

    suffix_inputs: list[tuple[int, ...]] = []
    handle = model.suffix.register_forward_pre_hook(
        lambda _module, inputs: suffix_inputs.append(tuple(inputs[0].shape))
    )
    logits = model(bits, public)
    handle.remove()
    assert logits.shape == (2,)
    assert suffix_inputs == [(8, config.channels, 64)]
    assert model.last_route is not None
    assert model.last_route.indices.shape == (2,)
    assert model.last_route.weights.shape == (2,)
    # Routing takes only the public card, not ciphertext rows.
    route = model.route(public)
    assert torch.equal(model.route(public).indices, route.indices)
    assert torch.equal(model.route(public).weights, route.weights)

    logits.sum().backward()
    gate_gradient = model.router.gate[0].weight.grad
    assert isinstance(gate_gradient, torch.Tensor)
    assert torch.isfinite(gate_gradient).all()
    assert any(parameter.grad is not None for parameter in model.parameters())

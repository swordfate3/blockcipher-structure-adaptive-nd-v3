"""Synthetic contracts for the key-free convolutional SIMON predictor."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from cipher_experiments.models.simon_conv import (
    CircularTapConv,
    ConvResidualBlock,
    SimonConvPredictor,
)

_LOCAL = (0, 1, 2, 3)
_ROTATION = (0, 1, 2, 8)


def _plaintexts(count: int = 8) -> Tensor:
    generator = torch.Generator().manual_seed(3517)
    return torch.randint(0, 2, (count, 64), generator=generator).to(torch.float32)


def _tensor_bytes(value: Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


@pytest.mark.parametrize("neighborhood,offsets", [("local", _LOCAL), ("rotation", _ROTATION)])
def test_predictor_shape_finite_metadata_and_persistent_wiring(
    neighborhood: str, offsets: tuple[int, ...]
) -> None:
    model = SimonConvPredictor(8, target_bit=47, neighborhood=neighborhood)
    logits = model(_plaintexts(5))

    assert logits.shape == (5, 1)
    assert logits.dtype is torch.float32
    assert bool(torch.isfinite(logits).all())
    assert model.rounds == 6
    assert model.target_bit == 47
    assert model.hidden_width == 8
    assert model.neighborhood == neighborhood
    assert model.offsets == offsets
    assert model.feature_mode == "full"
    assert model.prediction_mode == "direct"
    assert model.checkpoint_config_sha256 == ""
    assert torch.equal(model.indices, torch.arange(64))
    assert "indices" not in model.state_dict()

    tap_modules = [module for module in model.modules() if isinstance(module, CircularTapConv)]
    assert len(tap_modules) == 12
    assert all(module.offsets == offsets for module in tap_modules)
    assert all(module.conv.kernel_size == (4,) for module in tap_modules)
    assert all(module.conv.stride == (4,) for module in tap_modules)
    assert all(module.conv.bias is not None for module in tap_modules)
    assert all(module.gather_indices.dtype is torch.long for module in tap_modules)
    assert all(module.gather_indices.shape == (128,) for module in tap_modules)
    expected_indices = torch.tensor(
        [(position + offset) % 32 for position in range(32) for offset in offsets],
        dtype=torch.long,
    )
    assert all(torch.equal(module.gather_indices, expected_indices) for module in tap_modules)
    assert all(
        f"blocks.{index}.conv{conv}.gather_indices" in model.state_dict()
        for index in range(6)
        for conv in (1, 2)
    )
    model.validate_wiring()


def test_stem_receives_centered_msb_ordered_high_and_low_words() -> None:
    model = SimonConvPredictor(8).eval()
    plaintexts = torch.zeros((3, 64), dtype=torch.float32)
    plaintexts[0, 0] = 1.0
    plaintexts[0, 32] = 1.0
    plaintexts[1, 31] = 1.0
    plaintexts[1, 63] = 1.0
    captured: list[Tensor] = []
    hook = model.stem.register_forward_pre_hook(
        lambda _module, arguments: captured.append(cast(Tensor, arguments[0]).detach().clone())
    )
    try:
        model(plaintexts)
    finally:
        hook.remove()

    assert len(captured) == 1
    expected = plaintexts.reshape(3, 2, 32).mul(2.0).sub(1.0)
    assert torch.equal(captured[0], expected)
    assert torch.equal(captured[0][0, 0], torch.cat((torch.ones(1), -torch.ones(31))))
    assert torch.equal(captured[0][0, 1], torch.cat((torch.ones(1), -torch.ones(31))))
    assert torch.equal(captured[0][1, 0], torch.cat((-torch.ones(31), torch.ones(1))))
    assert torch.equal(captured[0][1, 1], torch.cat((-torch.ones(31), torch.ones(1))))
    assert torch.equal(captured[0][2], -torch.ones((2, 32)))


def test_circular_local_conv_matches_independent_right_circular_padding_oracle() -> None:
    torch.manual_seed(520)
    module = CircularTapConv(channels=3, offsets=_LOCAL)
    inputs = torch.randn(4, 3, 32)

    actual = module(inputs)
    padded = F.pad(inputs, (0, 3), mode="circular")
    expected = F.conv1d(padded, module.conv.weight, module.conv.bias, stride=1)

    assert actual.shape == (4, 3, 32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_circular_local_conv_input_and_parameter_gradients_match_padding_oracle() -> None:
    torch.manual_seed(519)
    module = CircularTapConv(channels=3, offsets=_LOCAL)
    inputs = torch.randn(2, 3, 32, requires_grad=True)
    probe = torch.randn(2, 3, 32)
    bias = module.conv.bias
    assert bias is not None

    actual = module(inputs)
    actual_gradients = torch.autograd.grad(
        actual,
        (inputs, module.conv.weight, bias),
        grad_outputs=probe,
    )
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    reference = F.conv1d(
        F.pad(reference_inputs, (0, 3), mode="circular"),
        module.conv.weight,
        bias,
        stride=1,
    )
    reference_gradients = torch.autograd.grad(
        reference,
        (reference_inputs, module.conv.weight, bias),
        grad_outputs=probe,
    )

    for actual_gradient, reference_gradient in zip(
        actual_gradients, reference_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-6, atol=1e-6)


def test_rotation_conv_matches_manual_per_offset_kernel_one_oracle() -> None:
    torch.manual_seed(521)
    module = CircularTapConv(channels=2, offsets=_ROTATION)
    inputs = torch.randn(3, 2, 32)

    expected = torch.zeros((3, 2, 32), dtype=inputs.dtype)
    for tap_index, offset in enumerate(_ROTATION):
        shifted = torch.cat((inputs[:, :, offset:], inputs[:, :, :offset]), dim=-1)
        expected = expected + F.conv1d(
            shifted,
            module.conv.weight[:, :, tap_index : tap_index + 1],
            bias=None,
        )
    bias = module.conv.bias
    assert bias is not None
    expected = expected + bias.view(1, -1, 1)

    torch.testing.assert_close(module(inputs), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("offset", _ROTATION)
def test_rotation_impulse_reads_msb_position_i_plus_offset_with_wrap(offset: int) -> None:
    module = CircularTapConv(channels=1, offsets=_ROTATION)
    with torch.no_grad():
        module.conv.weight.zero_()
        bias = module.conv.bias
        assert bias is not None
        bias.zero_()
        tap_index = _ROTATION.index(offset)
        module.conv.weight[0, 0, tap_index] = 1.0

    for impulse_position in (0, 31):
        inputs = torch.zeros((1, 1, 32))
        inputs[0, 0, impulse_position] = 1.0
        result = module(inputs)
        expected_output_position = (impulse_position - offset) % 32
        assert torch.equal(result[0, 0, expected_output_position], torch.tensor(1.0))
        assert torch.count_nonzero(result) == 1


def test_constructor_rejects_invalid_values_strictly() -> None:
    for kwargs in (
        {"hidden_width": True},
        {"hidden_width": 7},
        {"hidden_width": 257},
        {"rounds": True},
        {"rounds": 5},
        {"rounds": 7},
        {"target_bit": False},
        {"target_bit": -1},
        {"target_bit": 64},
    ):
        bad_integer = next(iter(kwargs.values()))
        error = TypeError if type(bad_integer) is not int else ValueError
        with pytest.raises(error):
            cast(Any, SimonConvPredictor)(**kwargs)

    with pytest.raises(TypeError, match="neighborhood"):
        cast(Any, SimonConvPredictor)(neighborhood=1)
    with pytest.raises(ValueError, match="neighborhood"):
        SimonConvPredictor(neighborhood="global")

    for channels in (True, 0, -1):
        error = TypeError if type(channels) is not int else ValueError
        with pytest.raises(error):
            CircularTapConv(channels=channels, offsets=_LOCAL)
    for offsets in ([0, 1, 2, 3], (0, 1, 2, True), (0, 1, 2, 4)):
        error = (
            TypeError
            if type(offsets) is not tuple
            or any(type(offset) is not int for offset in cast(tuple[object, ...], offsets))
            else ValueError
        )
        with pytest.raises(error):
            CircularTapConv(2, cast(tuple[int, ...], offsets))


@pytest.mark.parametrize(
    "inputs,error,message",
    [
        ([[0.0] * 64], TypeError, "torch.Tensor"),
        (torch.zeros((0, 64)), ValueError, "shape"),
        (torch.zeros((2, 63)), ValueError, "shape"),
        (torch.zeros((1, 64), dtype=torch.float64), TypeError, "float32"),
        (torch.full((1, 64), float("nan")), ValueError, "finite"),
        (torch.full((1, 64), 0.5), ValueError, "binary"),
    ],
)
def test_forward_rejects_non_plaintext_inputs(
    inputs: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        SimonConvPredictor(8)(cast(Tensor, inputs))


def test_forward_requires_model_and_input_on_same_device() -> None:
    with pytest.raises(ValueError, match="same device"):
        SimonConvPredictor(8).to(device="meta")(_plaintexts(1))


@pytest.mark.parametrize("corruption", ["value", "range", "shape", "dtype"])
def test_validate_wiring_rejects_buffer_tampering(corruption: str) -> None:
    model = SimonConvPredictor(8, neighborhood="rotation")
    tap = next(module for module in model.modules() if isinstance(module, CircularTapConv))
    if corruption == "value":
        tap.gather_indices[0] = (tap.gather_indices[0] + 1) % 32
    elif corruption == "range":
        tap.gather_indices[0] = 32
    elif corruption == "shape":
        tap.gather_indices = torch.zeros((127,), dtype=torch.long)
    else:
        tap.gather_indices = tap.gather_indices.to(torch.int32)

    with pytest.raises(ValueError, match="gather_indices"):
        model.validate_wiring()


def test_neighborhoods_have_identical_trainable_initialization_and_parameter_count() -> None:
    torch.manual_seed(913)
    local = SimonConvPredictor(8, neighborhood="local")
    torch.manual_seed(913)
    rotation = SimonConvPredictor(8, neighborhood="rotation")

    local_parameters = dict(local.named_parameters())
    rotation_parameters = dict(rotation.named_parameters())
    assert local_parameters.keys() == rotation_parameters.keys()
    assert all(
        _tensor_bytes(local_parameters[name]) == _tensor_bytes(rotation_parameters[name])
        for name in local_parameters
    )
    assert sum(parameter.numel() for parameter in local.parameters()) == 3913
    assert sum(parameter.numel() for parameter in rotation.parameters()) == 3913

    local_state = local.state_dict()
    rotation_state = rotation.state_dict()
    assert local_state.keys() == rotation_state.keys()
    differing_keys = {
        name
        for name in local_state
        if _tensor_bytes(local_state[name]) != _tensor_bytes(rotation_state[name])
    }
    expected_gather_keys = {
        f"blocks.{index}.conv{conv}.gather_indices" for index in range(6) for conv in (1, 2)
    }
    assert differing_keys == expected_gather_keys


def test_parameter_count_obeys_convolutional_scale_formula() -> None:
    for width in (8, 16, 64, 256):
        model = SimonConvPredictor(width)
        expected = 48 * width**2 + 105 * width + 1
        assert sum(parameter.numel() for parameter in model.parameters()) == expected


def test_all_residual_blocks_receive_finite_nonzero_gradients_and_adam_updates() -> None:
    torch.manual_seed(7301)
    model = SimonConvPredictor(8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    inputs = _plaintexts(12)
    labels = torch.randint(0, 2, (12, 1), generator=torch.Generator().manual_seed(821))
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("blocks.")
    }

    loss = F.binary_cross_entropy_with_logits(model(inputs), labels.to(torch.float32))
    loss.backward()  # type: ignore[no-untyped-call]

    for block_index, block in enumerate(model.blocks):
        gradients = [parameter.grad for parameter in block.parameters()]
        assert all(gradient is not None for gradient in gradients)
        assert all(bool(torch.isfinite(cast(Tensor, gradient)).all()) for gradient in gradients)
        assert sum(float(cast(Tensor, gradient).abs().sum()) for gradient in gradients) > 0.0
        assert any(
            float(cast(Tensor, parameter.grad).abs().sum()) > 0.0
            for parameter in block.parameters()
        ), f"block {block_index} has no nonzero parameter gradient"

    optimizer.step()
    for name, parameter in model.named_parameters():
        if name.startswith("blocks."):
            assert not torch.equal(parameter, before[name])


def test_residual_block_has_the_fixed_per_position_residual_topology() -> None:
    block = ConvResidualBlock(8, _LOCAL)
    assert isinstance(block.norm1, torch.nn.LayerNorm)
    assert isinstance(block.norm2, torch.nn.LayerNorm)
    assert block.norm1.normalized_shape == (8,)
    assert block.norm2.normalized_shape == (8,)
    assert isinstance(block.conv1, CircularTapConv)
    assert isinstance(block.conv2, CircularTapConv)
    assert not any(isinstance(module, torch.nn.BatchNorm1d) for module in block.modules())
    assert not any(isinstance(module, torch.nn.Dropout) for module in block.modules())
    inputs = torch.randn(3, 8, 32)
    result = block(inputs)
    assert result.shape == inputs.shape
    assert bool(torch.isfinite(result).all())


def test_forward_has_no_cross_sample_dependency() -> None:
    torch.manual_seed(113)
    model = SimonConvPredictor(8).eval()
    inputs = _plaintexts(7)
    with torch.no_grad():
        combined = model(inputs)
        split = torch.cat((model(inputs[:3]), model(inputs[3:])), dim=0)
    torch.testing.assert_close(combined, split, rtol=1e-6, atol=1e-7)


def test_model_module_imports_only_torch_and_prediction_helpers() -> None:
    module_path = (
        Path(__file__).resolve().parents[1] / "src/cipher_experiments/models/simon_conv.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_roots.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imported_roots <= {"torch", "prediction", "typing", "__future__"}

"""Public-only oracles: no cipher, labels, optimizer, or dataset imports.

Each group catches a concrete break: MSB/AND column drift, invalid public
contracts, extra RNG draws, repeated bias compensation, or wrong residual/L1.
"""

from __future__ import annotations

import random
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from cipher_experiments.models.mlp import BitMLP
from cipher_experiments.models.prediction import PlaintextPredictor, public_coordinate_transform
from cipher_experiments.models.public_prediction import PublicPredictor
from cipher_experiments.representations.prediction_public import (
    public_coordinates,
    public_interactions,
    zero_fixed_background,
    zero_round_constant_reference,
)

B = 0xA5A5A5A5
MASK = 0xFFFFFFFF


def bits32(word: int) -> list[int]:
    return [(word >> shift) & 1 for shift in range(31, -1, -1)]


def rol(word: int, count: int) -> int:
    return ((word << count) | (word >> (32 - count))) & MASK


def scalar_f(word: int) -> int:
    return (rol(word, 1) & rol(word, 8)) ^ rol(word, 2)


def scalar_phi(high: int, low: int) -> list[int]:
    z_high = low ^ scalar_f(high)
    return bits32(z_high) + bits32(high)


def scalar_psi(high: int, low: int) -> list[int]:
    z_high = low ^ scalar_f(high)
    return scalar_phi(high, low) + bits32(rol(z_high, 1) & rol(z_high, 8))


def scalar_q(high: int, low: int, rounds: int) -> list[int]:
    # All round CONSTANTS zero; deliberately no master key or key schedule.
    for _ in range(rounds):
        high, low = low ^ scalar_f(high), high
    return [(high >> 31) & 1]


def words() -> list[tuple[int, int]]:
    rng = random.Random(9081)
    return ([(0, 0), (MASK, MASK), (B, 0x12345678), (0x80000000, 1)]
            + [(1 << i, 0) for i in range(32)]
            + [(0, 1 << i) for i in range(32)]
            + [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(129)])


def plaintexts(pairs: list[tuple[int, int]], dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.tensor([bits32(high) + bits32(low) for high, low in pairs], dtype=dtype)


def predictor(**overrides: Any) -> PublicPredictor:
    options: dict[str, Any] = dict(
        model_kind="public-mlp", public_input="identity", public_readout="direct",
        background_init="standard", rounds=6,
    )
    options.update(overrides)
    return PublicPredictor(**options)


def background(**overrides: Any) -> PublicPredictor:
    return predictor(**dict(
        dict(public_input="background-zero", rounds=3, plaintext_space="high32-zero",
             plaintext_fixed_value=B), **overrides,
    ))


def linear(**overrides: Any) -> PublicPredictor:
    return predictor(**dict(
        dict(model_kind="public-linear", public_input="simon-interactions",
             public_readout="zero-round-constant-residual", l1_penalty=1e-4), **overrides,
    ))


@pytest.fixture(autouse=True)
def no_training() -> Any:
    with torch.no_grad():
        yield


@pytest.mark.parametrize("dtype", [torch.float32, torch.bool])
def test_phi_psi_every_column_matches_independent_uint32_oracle(dtype: torch.dtype) -> None:
    pairs = words()
    inputs = plaintexts(pairs, dtype)
    before = inputs.clone()
    phi = public_coordinates(inputs)
    psi = public_interactions(inputs)
    assert phi.dtype == psi.dtype == torch.float32
    assert phi.shape == (len(pairs), 64)
    assert psi.shape == (len(pairs), 96)
    assert torch.equal(phi, torch.tensor([scalar_phi(*pair) for pair in pairs]).float())
    assert torch.equal(psi, torch.tensor([scalar_psi(*pair) for pair in pairs]).float())
    assert torch.equal(phi, public_coordinate_transform(inputs))
    assert torch.equal(inputs, before)


@pytest.mark.parametrize("rounds", [6, 8, 10])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bool])
def test_zero_round_constants_q_matches_scalar_recurrence(rounds: int, dtype: torch.dtype) -> None:
    pairs = words()
    inputs = plaintexts(pairs, dtype)
    before = inputs.clone()
    actual = zero_round_constant_reference(inputs, rounds=rounds)
    expected = torch.tensor([scalar_q(*pair, rounds) for pair in pairs], dtype=torch.float32)
    assert actual.shape == (len(pairs), 1)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert torch.equal(inputs, before)


@pytest.mark.parametrize("fixed", [0, B, MASK])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bool])
def test_background_returns_fresh_zero_high_preserves_low_and_original(
    fixed: int, dtype: torch.dtype,
) -> None:
    inputs = plaintexts([(fixed, 0x12345678), (fixed, MASK)], dtype)
    before = inputs.clone()
    result = zero_fixed_background(inputs, plaintext_space="high32-zero", fixed_value=fixed)
    assert result.dtype == torch.float32
    assert torch.equal(result, plaintexts([(0, 0x12345678), (0, MASK)]))
    assert result.data_ptr() != inputs.data_ptr()
    result.zero_()
    assert torch.equal(inputs, before)


@pytest.mark.parametrize("function", [public_coordinates, public_interactions,
                                     lambda x: zero_round_constant_reference(x, rounds=6),
                                     lambda x: zero_fixed_background(
                                         x, plaintext_space="high32-zero", fixed_value=0)])
@pytest.mark.parametrize("inputs,error", [
    (None, TypeError), (torch.zeros(1, 64, dtype=torch.float64), TypeError),
    (torch.zeros(1, 64, dtype=torch.int64), TypeError),
    (torch.zeros(0, 64), ValueError), (torch.zeros(64), ValueError),
    (torch.zeros(1, 63), ValueError), (torch.zeros(1, 64, 1), ValueError),
    (torch.zeros(1, 64).to_sparse(), ValueError),
    (torch.full((1, 64), float("nan")), ValueError),
    (torch.full((1, 64), float("inf")), ValueError),
    (torch.full((1, 64), 0.5), ValueError),
])
def test_public_helpers_reject_invalid_binary_tensors(
    function: Any, inputs: Any, error: type[Exception],
) -> None:
    with pytest.raises(error):
        function(inputs)


@pytest.mark.parametrize("overrides,error", [
    ({"rounds": True}, TypeError), ({"rounds": 6.0}, TypeError),
    ({"rounds": "6"}, TypeError), ({"rounds": 3}, ValueError),
    ({"rounds": 7}, ValueError), ({"rounds": 44}, ValueError),
    ({"target_bit": False}, TypeError), ({"target_bit": 0.0}, TypeError),
    ({"target_bit": 1}, ValueError), ({"target_bit": -1}, ValueError),
])
def test_q_rejects_non_protocol_rounds_and_bits(overrides: dict[str, Any],
                                               error: type[Exception]) -> None:
    args: dict[str, Any] = {"rounds": 6}
    args.update(overrides)
    with pytest.raises(error):
        zero_round_constant_reference(torch.zeros(1, 64), **args)


@pytest.mark.parametrize("overrides,error", [
    ({"plaintext_space": "full"}, ValueError), ({"plaintext_space": True}, TypeError),
    ({"fixed_value": True}, TypeError), ({"fixed_value": 0.0}, TypeError),
    ({"fixed_value": -1}, ValueError), ({"fixed_value": 1 << 32}, ValueError),
    ({"fixed_value": 1}, ValueError),
])
def test_background_rejects_unverified_fixed_half(overrides: dict[str, Any],
                                                 error: type[Exception]) -> None:
    args: dict[str, Any] = dict(plaintext_space="high32-zero", fixed_value=0)
    args.update(overrides)
    with pytest.raises(error):
        zero_fixed_background(torch.zeros(2, 64), **args)


@pytest.mark.parametrize("key,value,error", [
    ("hidden_width", True, TypeError), ("hidden_width", 128.0, TypeError),
    ("hidden_width", "128", TypeError), ("hidden_width", 0, ValueError),
    ("hidden_width", 4097, ValueError),
    ("rounds", True, TypeError), ("rounds", 6.0, TypeError),
    ("rounds", "6", TypeError), ("rounds", 7, ValueError),
    ("target_bit", False, TypeError), ("target_bit", 0.0, TypeError),
    ("target_bit", 1, ValueError),
    ("plaintext_fixed_value", True, TypeError), ("plaintext_fixed_value", 0.0, TypeError),
    ("plaintext_fixed_value", -1, ValueError), ("plaintext_fixed_value", 1 << 32, ValueError),
    ("plaintext_fixed_value", 1, ValueError),
    ("l1_penalty", True, TypeError), ("l1_penalty", "0", TypeError),
    ("l1_penalty", float("nan"), ValueError), ("l1_penalty", float("inf"), ValueError),
    ("l1_penalty", -1.0, ValueError), ("l1_penalty", 1e-4, ValueError),
])
def test_model_rejects_invalid_public_numeric_parameters(key: str, value: Any,
                                                         error: type[Exception]) -> None:
    with pytest.raises(error):
        predictor(**{key: value})


@pytest.mark.parametrize("key", ["model_kind", "public_input", "public_readout",
                                 "background_init", "plaintext_space"])
@pytest.mark.parametrize("value,error", [(True, TypeError), (1, TypeError),
                                        (None, TypeError), ("not-a-mode", ValueError)])
def test_model_rejects_non_exact_enum_values(key: str, value: Any,
                                            error: type[Exception]) -> None:
    with pytest.raises(error):
        predictor(**{key: value})


@pytest.mark.parametrize("builder,overrides", [
    (predictor, dict(rounds=3)), (predictor, dict(background_init="match-direct")),
    (predictor, dict(plaintext_space="high32-zero")),
    (background, dict(rounds=6)), (background, dict(plaintext_space="full")),
    (background, dict(public_readout="zero-round-constant-residual")),
    (background, dict(l1_penalty=1e-4)),
    (linear, dict(hidden_width=64)), (linear, dict(public_input="identity")),
    (linear, dict(public_readout="direct")), (linear, dict(l1_penalty=0)),
    (linear, dict(background_init="match-direct")),
    (linear, dict(plaintext_space="high32-zero")),
])
def test_model_rejects_illegal_combinations(builder: Any, overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        builder(**overrides)


def test_builtin_subclasses_are_not_silently_accepted() -> None:
    class StringSubclass(str):
        pass

    class IntSubclass(int):
        pass

    with pytest.raises(TypeError):
        predictor(public_input=StringSubclass("identity"))
    with pytest.raises(TypeError):
        predictor(hidden_width=IntSubclass(128))


@pytest.mark.parametrize("rounds", [6, 8, 10])
@pytest.mark.parametrize("public_input,width,count", [("identity", 64, 24961),
                                                     ("simon-interactions", 96, 29057)])
def test_mlp_registration_parameter_count_and_nonpersistent_indices(
    rounds: int, public_input: str, width: int, count: int,
) -> None:
    model = predictor(public_input=public_input, rounds=rounds)
    assert sum(p.numel() for p in model.parameters()) == count
    assert [type(layer) for layer in model.layers] == [nn.Linear, nn.ReLU, nn.Linear,
                                                      nn.ReLU, nn.Linear]
    assert model.layers[0].weight.shape == (128, width)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert torch.equal(model.indices, torch.arange(64))
    assert "indices" not in model.state_dict()
    assert not hasattr(model, "checkpoint_config_sha256")
    assert model.public_input == public_input
    assert model.public_readout == "direct"
    assert model.background_init == "standard"


@pytest.mark.parametrize("hidden_width", [1, 4096])
def test_mlp_width_limits_are_legal_without_changing_old_bitmlp(hidden_width: int) -> None:
    model = predictor(hidden_width=hidden_width)
    assert model.layers[0].weight.shape == (hidden_width, 64)
    with pytest.raises(ValueError):
        BitMLP(input_width=96)


def test_n1_r_z_initial_state_rng_equal_and_no_input_mutation() -> None:
    torch.manual_seed(120)
    original = PlaintextPredictor(128, rounds=3)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(120)
    zeroed = background()
    assert torch.equal(torch.get_rng_state(), rng)
    assert original.state_dict().keys() == zeroed.state_dict().keys()
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, zeroed.state_dict()[name])
    inputs = plaintexts([(B, 0x12345678), (B, MASK)])
    before = inputs.clone()
    actual = zeroed(inputs)
    assert torch.equal(inputs, before)
    assert torch.equal(actual, original(plaintexts([(0, 0x12345678), (0, MASK)])))
    with pytest.raises(ValueError):
        zeroed(plaintexts([(B, 0), (0, 0)]))


def test_n1_m_matched_initial_logits_and_bias_only_change() -> None:
    torch.manual_seed(73)
    original = PlaintextPredictor(128, rounds=3)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(73)
    matched = background(background_init="match-direct")
    assert torch.equal(torch.get_rng_state(), rng)
    for name, tensor in original.state_dict().items():
        if name != "layers.0.bias":
            assert torch.equal(tensor, matched.state_dict()[name])
    original_first = cast(nn.Linear, original.layers[0])
    matched_first = cast(nn.Linear, matched.layers[0])
    expected_bias = original_first.bias + original_first.weight[:, :32] @ torch.tensor(
        bits32(B), dtype=torch.float32)
    assert torch.equal(matched_first.bias, expected_bias)
    inputs = plaintexts([(B, low) for _, low in words()])
    before = inputs.clone()
    torch.testing.assert_close(matched(inputs), original(inputs), rtol=0, atol=1e-5)
    assert torch.equal(inputs, before)
    # Repeated forward and strict load must not apply a second compensation.
    expected = matched(inputs).clone()
    saved = {key: value.clone() for key, value in matched.state_dict().items()}
    restored = background(background_init="match-direct")
    for _ in range(2):
        restored.load_state_dict(saved, strict=True)
        assert torch.equal(restored(inputs), expected)
        assert all(torch.equal(saved[k], v) for k, v in restored.state_dict().items())


def test_independent_fp64_affine_oracle_for_background_bias_identity() -> None:
    # Pure affine oracle, not production FP64: no model or representation helper.
    weights = torch.arange(7 * 64, dtype=torch.float64).reshape(7, 64) / 1024 - 0.2
    bias = torch.arange(7, dtype=torch.float64) / 13
    fixed = torch.tensor(bits32(B), dtype=torch.float64)
    low = torch.tensor([bits32(0), bits32(0x12345678), bits32(MASK)], dtype=torch.float64)
    direct = torch.cat((fixed.expand(3, -1), low), dim=1) @ weights.T + bias
    corrected_bias = bias + weights[:, :32] @ fixed
    centered = torch.cat((torch.zeros(3, 32, dtype=torch.float64), low), dim=1)
    matched = centered @ weights.T + corrected_bias
    assert float((direct - matched).abs().max()) <= 1e-12


@pytest.mark.parametrize("public_input", ["identity", "simon-interactions"])
@pytest.mark.parametrize("rounds", [6, 8, 10])
def test_n2_direct_residual_share_state_rng_and_restore_sign_last(public_input: str,
                                                                 rounds: int) -> None:
    torch.manual_seed(891)
    direct = predictor(public_input=public_input, rounds=rounds)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(891)
    residual = predictor(public_input=public_input, rounds=rounds,
                         public_readout="zero-round-constant-residual")
    assert torch.equal(torch.get_rng_state(), rng)
    assert direct.state_dict().keys() == residual.state_dict().keys()
    for name, tensor in direct.state_dict().items():
        assert torch.equal(tensor, residual.state_dict()[name])
    pairs = words()
    inputs = plaintexts(pairs)
    before_rng = torch.get_rng_state().clone()
    before = inputs.clone()
    sign = 1 - 2 * torch.tensor([scalar_q(*p, rounds) for p in pairs]).float()
    assert torch.equal(residual(inputs), sign * direct(inputs))
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert torch.equal(inputs, before)
    features = inputs if public_input == "identity" else torch.tensor(
        [scalar_psi(*p) for p in pairs], dtype=torch.float32)
    assert torch.equal(direct(inputs), direct.layers(features))
    if public_input == "identity":
        torch.manual_seed(891)
        old = PlaintextPredictor(128, rounds=rounds)
        assert torch.equal(old(inputs), direct(inputs))
        assert torch.equal(torch.get_rng_state(), rng)


def test_linear_has_one_layer_97_parameters_and_weight_only_l1() -> None:
    model = linear()
    assert len(model.layers) == 1
    assert isinstance(model.layers[0], nn.Linear)
    assert sum(p.numel() for p in model.parameters()) == 97
    model.layers[0].weight.fill_(-2)
    model.layers[0].bias.fill_(99)
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    loss = model.regularization_loss()
    assert loss.ndim == 0
    assert loss.dtype == torch.float32
    assert float(loss) == pytest.approx(96 * 2 * 1e-4)
    assert all(torch.equal(saved[k], v) for k, v in model.state_dict().items())
    pairs = words()
    inputs = plaintexts(pairs)
    sign = 1 - 2 * torch.tensor([scalar_q(*p, 6) for p in pairs]).float()
    features = torch.tensor([scalar_psi(*p) for p in pairs], dtype=torch.float32)
    assert torch.equal(model(inputs), sign * (features @ torch.full((96, 1), -2.0) + 99))
    mlp = predictor(l1_penalty=0)
    assert float(mlp.regularization_loss()) == 0.0


@pytest.mark.parametrize("builder", [predictor, background, linear])
@pytest.mark.parametrize("inputs,error", [
    (None, TypeError), (torch.zeros(1, 64, dtype=torch.bool), TypeError),
    (torch.zeros(1, 64, dtype=torch.float64), TypeError),
    (torch.zeros(1, 96), ValueError), (torch.zeros(0, 64), ValueError),
    (torch.zeros(64), ValueError), (torch.zeros(1, 64).to_sparse(), ValueError),
    (torch.full((1, 64), float("nan")), ValueError),
    (torch.full((1, 64), float("inf")), ValueError),
    (torch.full((1, 64), 0.5), ValueError),
])
def test_model_forward_rejects_invalid_observations(builder: Any, inputs: Any,
                                                   error: type[Exception]) -> None:
    with pytest.raises(error):
        builder()(inputs)


def test_model_rejects_device_mismatch_before_layers() -> None:
    model = predictor()
    with pytest.raises(ValueError, match="device"):
        model(torch.empty(1, 64, device="meta"))


def test_model_rejects_nonfinite_logits() -> None:
    model = predictor()
    cast(nn.Linear, model.layers[-1]).bias.fill_(float("nan"))
    with pytest.raises(RuntimeError):
        model(torch.zeros(1, 64))

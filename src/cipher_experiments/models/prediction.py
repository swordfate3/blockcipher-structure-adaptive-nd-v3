"""Key-free public-reference transforms and plaintext bit predictors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .circuit import CircuitPredictor
    from .logic import LogicPredictor
    from .simon_conv import SimonConvPredictor
    from .simon_two_round import (
        TwoRoundLocalPredictor,
        TwoRoundTablePredictor,
        TwoRoundVectorPredictor,
    )
    from .simon_vector import SimonVectorPredictor
    from .structured_simon import StructuredSimonPredictor
    from .unrolled_simon import UnrolledSimonPredictor

from .mlp import BitMLP


def _validate_builtin_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in int")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _validate_plaintexts(inputs: Tensor) -> None:
    if not isinstance(inputs, Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    if (
        inputs.layout != torch.strided
        or inputs.ndim != 2
        or inputs.shape[0] < 1
        or inputs.shape[1] != 64
    ):
        raise ValueError("inputs must have nonempty shape [N,64]")
    if inputs.dtype != torch.float32:
        raise TypeError("inputs must have dtype torch.float32")
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("inputs must be finite")
    if not bool(((inputs == 0) | (inputs == 1)).all()):
        raise ValueError("inputs must contain binary values")


def _validate_reference_inputs(inputs: Tensor) -> None:
    if not isinstance(inputs, Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    if (
        inputs.layout != torch.strided
        or inputs.ndim != 2
        or inputs.shape[0] < 1
        or inputs.shape[1] != 64
    ):
        raise ValueError("inputs must have nonempty shape [N,64]")
    if inputs.dtype not in (torch.float32, torch.bool):
        raise TypeError("inputs must have dtype torch.float32 or torch.bool")
    if inputs.dtype == torch.float32:
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("inputs must be finite")
        if not bool(((inputs == 0) | (inputs == 1)).all()):
            raise ValueError("inputs must contain binary values")


def support_indices(rounds: int, target_bit: int) -> tuple[int, ...]:
    """Return plaintext MSB indices influencing one SIMON64 output bit.

    This propagates sets backward through the public key-free round recurrence;
    it does not inspect actual cipher labels, keys, or data.
    """
    rounds = _validate_builtin_int(rounds, "rounds", 1, 44)
    target_bit = _validate_builtin_int(target_bit, "target_bit", 0, 63)

    # State bit positions use plaintext/output MSB order: x=0..31, y=32..63.
    x_support = [frozenset((index,)) for index in range(32)]
    y_support = [frozenset((index,)) for index in range(32, 64)]
    for _ in range(rounds):
        next_x: list[frozenset[int]] = []
        for bit in range(32):
            dependencies = set(y_support[bit])
            dependencies.update(x_support[(bit + 2) % 32])
            # AND's output can depend on either operand.
            dependencies.update(x_support[(bit + 1) % 32])
            dependencies.update(x_support[(bit + 8) % 32])
            next_x.append(frozenset(dependencies))
        x_support, y_support = next_x, x_support
    return tuple(sorted(x_support[target_bit] if target_bit < 32 else y_support[target_bit - 32]))


def _public_round_function(high: Tensor) -> Tensor:
    """Apply ROL1(X) & ROL8(X) xor ROL2(X) in MSB-indexed tensor order."""
    rotate_one = torch.cat((high[:, 1:], high[:, :1]), dim=1)
    rotate_eight = torch.cat((high[:, 8:], high[:, :8]), dim=1)
    rotate_two = torch.cat((high[:, 2:], high[:, :2]), dim=1)
    return (rotate_one & rotate_eight) ^ rotate_two


def public_coordinate_transform(inputs: Tensor) -> Tensor:
    """Map plaintext (X,Y) to the reversible, key-free coordinate (Y xor F(X), X).

    This is a public coordinate system, not an actual cipher state: it does not
    include round-key material. Boolean operations preserve all 64 bits exactly
    before the result is returned in the model's float32 input dtype.
    """
    _validate_reference_inputs(inputs)
    bits = inputs.to(dtype=torch.bool)
    high, low = bits[:, :32], bits[:, 32:]
    transformed = torch.cat((low ^ _public_round_function(high), high), dim=1)
    return transformed.to(dtype=torch.float32)


def public_coordinate_inverse(coordinates: Tensor) -> Tensor:
    """Invert :func:`public_coordinate_transform` for binary 64-bit rows."""
    _validate_reference_inputs(coordinates)
    bits = coordinates.to(dtype=torch.bool)
    transformed_high, original_high = bits[:, :32], bits[:, 32:]
    original_low = transformed_high ^ _public_round_function(original_high)
    plaintexts = torch.cat((original_high, original_low), dim=1)
    return plaintexts.to(dtype=torch.float32)


def public_reference(inputs: Tensor, rounds: int, target_bit: int) -> Tensor:
    """Compute a zero-master-key SIMON64/128 reference output bit.

    The first four round keys of a zero master key are zero. The pure Boolean
    recurrence therefore defines this public reference for rounds 1..4 only;
    all rounds 1..44 remain valid for plaintext support calculation.
    """
    rounds = _validate_builtin_int(rounds, "rounds", 1, 4)
    target_bit = _validate_builtin_int(target_bit, "target_bit", 0, 63)
    _validate_reference_inputs(inputs)

    # Tensor Boolean operations preserve bool or float32 values without relying
    # on XOR arithmetic that could drift away from exact binary values.
    high, low = inputs[:, :32], inputs[:, 32:]
    for _ in range(rounds):
        rotate_one = torch.cat((high[:, 1:], high[:, :1]), dim=1)
        rotate_eight = torch.cat((high[:, 8:], high[:, :8]), dim=1)
        rotate_two = torch.cat((high[:, 2:], high[:, :2]), dim=1)
        if inputs.dtype == torch.bool:
            next_high = low ^ (rotate_one & rotate_eight) ^ rotate_two
        else:
            and_term = rotate_one * rotate_eight
            next_high = low + and_term - 2.0 * low * and_term
            next_high = next_high + rotate_two - 2.0 * next_high * rotate_two
        low, high = high, next_high
    if target_bit < 32:
        result = high[:, target_bit : target_bit + 1]
    else:
        result = low[:, target_bit - 32 : target_bit - 31]
    return result


def build_predictor(hidden_width: int = 64, *, model_kind: str = "mlp",
                    rounds: int = 3, target_bit: int = 0, feature_mode: str = "full",
                    prediction_mode: str = "direct"
                    ) -> (
                        PlaintextPredictor | LogicPredictor | CircuitPredictor
                        | StructuredSimonPredictor | UnrolledSimonPredictor
                        | TwoRoundTablePredictor | TwoRoundLocalPredictor
                        | TwoRoundVectorPredictor | SimonConvPredictor | SimonVectorPredictor):
    """Explicit small model choices, without experiment config or a registry."""
    if model_kind == "simon-vector":
        if feature_mode != "full" or prediction_mode != "direct":
            raise ValueError("simon-vector requires full/direct")
        from .simon_vector import SimonVectorPredictor
        return SimonVectorPredictor(hidden_width, rounds=rounds, target_bit=target_bit)
    if model_kind in ("simon-cnn", "simon-rotation-cnn"):
        if feature_mode != "full" or prediction_mode != "direct":
            raise ValueError("simon CNN models require full/direct")
        from .simon_conv import SimonConvPredictor
        return SimonConvPredictor(
            hidden_width, rounds=rounds, target_bit=target_bit,
            neighborhood="local" if model_kind == "simon-cnn" else "rotation")
    if model_kind in ("simon-r2-table", "simon-r2-local", "simon-r2-vector"):
        _validate_builtin_int(hidden_width, "hidden_width", 16, 16)
        _validate_builtin_int(rounds, "rounds", 2, 2)
        _validate_builtin_int(target_bit, "target_bit", 0, 31)
        if feature_mode != "full" or prediction_mode != "direct":
            raise ValueError(
                "simon-r2 models require hidden_width=16, full/direct and rounds=2"
            )
        from .simon_two_round import (
            TwoRoundLocalPredictor,
            TwoRoundTablePredictor,
            TwoRoundVectorPredictor,
        )
        if model_kind == "simon-r2-table":
            return TwoRoundTablePredictor(rounds=rounds, target_bit=target_bit)
        if model_kind == "simon-r2-local":
            return TwoRoundLocalPredictor(rounds=rounds, target_bit=target_bit)
        return TwoRoundVectorPredictor(rounds=rounds, target_bit=target_bit)
    xor_modes = ("plaintext-xor", "plaintext-xor-no-final-low")
    if prediction_mode == "public-coordinate" and model_kind != "mlp":
        raise ValueError("public-coordinate requires model_kind=mlp")
    if prediction_mode in xor_modes and model_kind != "simon-unrolled":
        raise ValueError("plaintext-xor modes require model_kind=simon-unrolled")
    if model_kind == "simon-unrolled":
        if feature_mode != "full" or prediction_mode not in ("direct", *xor_modes):
            raise ValueError("simon-unrolled requires full/direct or a full/plaintext-xor mode")
        from .unrolled_simon import UnrolledSimonPredictor
        return UnrolledSimonPredictor(hidden_width, rounds=rounds, target_bit=target_bit,
                                      prediction_mode=prediction_mode)
    if model_kind == "simon-structure":
        from .structured_simon import StructuredSimonPredictor
        return StructuredSimonPredictor(rounds=rounds, target_bit=target_bit)
    if model_kind in ("circuit-aligned", "circuit-random"):
        from .circuit import CircuitPredictor
        return CircuitPredictor(wiring=model_kind.removeprefix("circuit-"))
    if model_kind == "mlp":
        return PlaintextPredictor(hidden_width, rounds=rounds, target_bit=target_bit,
                                  feature_mode=feature_mode, prediction_mode=prediction_mode)
    if model_kind == "logic16":
        from .logic import LogicPredictor
        return LogicPredictor(hidden_width, rounds=rounds, target_bit=target_bit,
                              feature_mode=feature_mode, prediction_mode=prediction_mode)
    raise ValueError("unknown prediction model_kind")


class PlaintextPredictor(BitMLP):
    """Predict an absolute plaintext-conditioned output-bit logit.

    ``support`` mode supplies a conservative structural dependency set, while
    ``public-residual`` learns a residual logit relative to the public
    zero-master reference and restores its sign for absolute-label BCE.
    """

    indices: Tensor
    checkpoint_config_sha256: str

    def __init__(
        self,
        hidden_width: int = 64,
        *,
        rounds: int = 3,
        target_bit: int = 0,
        feature_mode: str = "full",
        prediction_mode: str = "direct",
    ) -> None:
        rounds = _validate_builtin_int(rounds, "rounds", 1, 44)
        target_bit = _validate_builtin_int(target_bit, "target_bit", 0, 63)
        if feature_mode not in ("full", "support"):
            raise ValueError("feature_mode must be 'full' or 'support'")
        if prediction_mode not in ("direct", "public-residual", "public-coordinate"):
            raise ValueError(
                "prediction_mode must be 'direct', 'public-residual', or 'public-coordinate'"
            )
        if prediction_mode == "public-residual" and rounds > 4:
            raise ValueError("public-residual mode supports rounds 1..4")
        if prediction_mode == "public-coordinate" and (
                feature_mode != "full" or not 2 <= rounds <= 44):
            raise ValueError("public-coordinate mode requires full features and rounds 2..44")

        selected = (
            tuple(range(64))
            if feature_mode == "full"
            else support_indices(rounds, target_bit)
        )
        super().__init__(hidden_width, input_width=len(selected))
        self.rounds = rounds
        self.target_bit = target_bit
        self.feature_mode = feature_mode
        self.prediction_mode = prediction_mode
        self.register_buffer(
            "indices", torch.tensor(selected, dtype=torch.long), persistent=False
        )

    def forward(self, inputs: Tensor) -> Tensor:
        _validate_plaintexts(inputs)
        if inputs.device != next(self.parameters()).device:
            raise ValueError("inputs and model must be on the same device")
        predictor_inputs = (public_coordinate_transform(inputs)
                            if self.prediction_mode == "public-coordinate" else inputs)
        features = predictor_inputs.index_select(1, self.indices)
        logits = super().forward(features)
        if self.prediction_mode in ("direct", "public-coordinate"):
            return logits
        reference = public_reference(inputs, self.rounds, self.target_bit)
        reference = reference.to(dtype=logits.dtype)
        absolute_logits = (1.0 - 2.0 * reference) * logits
        if not bool(torch.isfinite(absolute_logits).all()):
            raise RuntimeError("model produced invalid absolute logits")
        return absolute_logits

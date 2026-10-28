"""Position-local convolutional predictors for plaintext-conditioned SIMON bits."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .prediction import _validate_builtin_int, _validate_plaintexts

_LOCAL_OFFSETS = (0, 1, 2, 3)
_ROTATION_OFFSETS = (0, 1, 2, 8)
_WORD_BITS = 32


def _check_channels(channels: int) -> None:
    if type(channels) is not int:
        raise TypeError("channels must be a built-in int")
    if channels < 1:
        raise ValueError("channels must be positive")


def _check_offsets(offsets: tuple[int, ...]) -> None:
    if type(offsets) is not tuple or any(type(offset) is not int for offset in offsets):
        raise TypeError("offsets must be a tuple of built-in ints")
    if offsets not in (_LOCAL_OFFSETS, _ROTATION_OFFSETS):
        raise ValueError("offsets must be (0, 1, 2, 3) or (0, 1, 2, 8)")


class CircularTapConv(nn.Module):
    """Gather four circular taps per position before a stride-four Conv1d."""

    channels: int
    offsets: tuple[int, ...]
    gather_indices: Tensor

    def __init__(self, channels: int, offsets: tuple[int, ...]) -> None:
        super().__init__()
        _check_channels(channels)
        _check_offsets(offsets)

        self.channels = channels
        self.offsets = offsets
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=4, bias=True)
        gather_indices = torch.tensor(
            [
                (position + offset) % _WORD_BITS
                for position in range(_WORD_BITS)
                for offset in offsets
            ],
            dtype=torch.long,
        )
        self.register_buffer("gather_indices", gather_indices, persistent=True)

    def validate_wiring(self) -> None:
        """Reject malformed or altered persistent circular-tap wiring."""
        _check_offsets(self.offsets)
        indices = self.gather_indices
        expected_shape = (_WORD_BITS * len(self.offsets),)
        if indices.dtype is not torch.long:
            raise ValueError("gather_indices must have dtype torch.long")
        if tuple(indices.shape) != expected_shape:
            raise ValueError(f"gather_indices must have shape {expected_shape}")
        if indices.numel() and (
            int(indices.min().item()) < 0 or int(indices.max().item()) >= _WORD_BITS
        ):
            raise ValueError("gather_indices values must be in [0, 32)")
        expected = torch.tensor(
            [
                (position + offset) % _WORD_BITS
                for position in range(_WORD_BITS)
                for offset in self.offsets
            ],
            dtype=torch.long,
            device=indices.device,
        )
        if not torch.equal(indices, expected):
            raise ValueError("gather_indices do not match the configured circular wiring")

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (self.channels, _WORD_BITS):
            raise ValueError(f"inputs must have shape [N,{self.channels},32]")
        gathered = inputs.index_select(-1, self.gather_indices)
        return cast(Tensor, self.conv(gathered))


class ConvResidualBlock(nn.Module):
    """A two-convolution residual block normalized independently per position."""

    def __init__(self, channels: int, offsets: tuple[int, ...]) -> None:
        super().__init__()
        _check_channels(channels)
        _check_offsets(offsets)
        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = CircularTapConv(channels, offsets)
        self.norm2 = nn.LayerNorm(channels)
        self.conv2 = CircularTapConv(channels, offsets)

    @staticmethod
    def _normalize(inputs: Tensor, norm: nn.LayerNorm) -> Tensor:
        return cast(Tensor, norm(inputs.transpose(1, 2)).transpose(1, 2))

    def forward(self, inputs: Tensor) -> Tensor:
        branch = self._normalize(inputs, self.norm1)
        branch = F.gelu(branch)
        branch = self.conv1(branch)
        branch = self._normalize(branch, self.norm2)
        branch = F.gelu(branch)
        branch = self.conv2(branch)
        return cast(Tensor, inputs + branch)


class SimonConvPredictor(nn.Module):
    """Predict one raw output-bit logit from binary 64-bit plaintexts.

    The six learned residual blocks are a convolutional architecture, not six
    simulated SIMON rounds.  No key, cipher state, cipher routine, or label is
    provided to this model.
    """

    indices: Tensor
    checkpoint_config_sha256: str

    def __init__(
        self,
        hidden_width: int = 64,
        *,
        rounds: int = 6,
        target_bit: int = 0,
        neighborhood: str = "local",
    ) -> None:
        super().__init__()
        hidden_width = _validate_builtin_int(hidden_width, "hidden_width", 8, 256)
        if type(rounds) is not int:
            raise TypeError("rounds must be a built-in int")
        if rounds != 6:
            raise ValueError("rounds must be exactly 6")
        target_bit = _validate_builtin_int(target_bit, "target_bit", 0, 63)
        if type(neighborhood) is not str:
            raise TypeError("neighborhood must be a string")
        if neighborhood == "local":
            offsets = _LOCAL_OFFSETS
        elif neighborhood == "rotation":
            offsets = _ROTATION_OFFSETS
        else:
            raise ValueError("neighborhood must be 'local' or 'rotation'")

        self.hidden_width = hidden_width
        self.rounds = rounds
        self.target_bit = target_bit
        self.neighborhood = neighborhood
        self.offsets = offsets
        self.feature_mode = "full"
        self.prediction_mode = "direct"
        self.checkpoint_config_sha256 = ""
        self.register_buffer("indices", torch.arange(64), persistent=False)

        self.stem = nn.Conv1d(2, hidden_width, kernel_size=1, bias=True)
        self.position = nn.Parameter(torch.zeros(1, hidden_width, _WORD_BITS))
        self.blocks = nn.ModuleList(ConvResidualBlock(hidden_width, offsets) for _ in range(6))
        self.output_norm = nn.LayerNorm(hidden_width)
        self.readout = nn.Linear(_WORD_BITS * hidden_width, 1)

    def validate_wiring(self) -> None:
        """Validate every persistent convolution tap buffer in this model."""
        for module in self.modules():
            if isinstance(module, CircularTapConv):
                module.validate_wiring()

    def forward(self, inputs: Tensor) -> Tensor:
        _validate_plaintexts(inputs)
        if inputs.device != self.stem.weight.device:
            raise ValueError("inputs and model must be on the same device")

        state = inputs.reshape(inputs.shape[0], 2, _WORD_BITS).mul(2.0).sub(1.0)
        state = self.stem(state) + self.position
        for block in self.blocks:
            state = block(state)
        features = self.output_norm(state.transpose(1, 2)).transpose(1, 2)
        features = F.gelu(features)
        logits = self.readout(features.flatten(start_dim=1))
        if logits.shape != (inputs.shape[0], 1) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("model produced invalid logits")
        return cast(Tensor, logits)

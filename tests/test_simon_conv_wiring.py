"""Independent block arithmetic and every-layer canonical checkpoint wiring."""
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from cipher_experiments.artifacts import file_sha256
from cipher_experiments.checkpoints import load_checkpoint, save_checkpoint
from cipher_experiments.config import ExperimentConfig
from cipher_experiments.models.prediction import build_predictor
from cipher_experiments.models.simon_conv import ConvResidualBlock


@pytest.mark.parametrize("kind", ["simon-cnn", "simon-rotation-cnn"])
@pytest.mark.parametrize("index", range(12))
def test_loader_checks_canonical_wiring_in_every_layer(
    kind: str, index: int, tmp_path: Path,
) -> None:
    cfg = ExperimentConfig(name="wiring", dataset="simon", model_kind=kind,
                           rounds=6, hidden_width=8, label_mode="null")
    model = build_predictor(8, model_kind=kind, rounds=6)
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, cfg, 1, 1, 0, .5)
    payload = torch.load(path, weights_only=True, map_location="cpu")
    key = f"blocks.{index // 2}.conv{index % 2 + 1}.gather_indices"
    payload["state_dict"][key] = payload["state_dict"][key].roll(1)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="wiring"):
        load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))


@pytest.mark.parametrize("kind", ["simon-cnn", "simon-rotation-cnn"])
def test_other_adjacency_buffers_are_rejected_even_with_all_matching_shapes(
    kind: str, tmp_path: Path,
) -> None:
    cfg = ExperimentConfig(name="wiring", dataset="simon", model_kind=kind,
                           rounds=6, hidden_width=8, label_mode="null")
    model = build_predictor(8, model_kind=kind, rounds=6)
    other = "simon-cnn" if kind == "simon-rotation-cnn" else "simon-rotation-cnn"
    other_state = build_predictor(8, model_kind=other, rounds=6).state_dict()
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, cfg, 1, 1, 0, .5)
    payload = torch.load(path, weights_only=True, map_location="cpu")
    for key in payload["state_dict"]:
        if key.endswith("gather_indices"):
            payload["state_dict"][key] = other_state[key]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="wiring"):
        load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))


@pytest.mark.parametrize("offsets", [(0, 1, 2, 3), (0, 1, 2, 8)])
def test_residual_block_independent_numeric_oracle(offsets: tuple[int, ...]) -> None:
    torch.manual_seed(121)
    block = ConvResidualBlock(8, offsets).double()
    x = torch.randn(2, 8, 32, dtype=torch.float64)
    expected = x
    # Use functional LN and per-tap kernel1 convolutions, never block.forward/gather.
    for norm, convolution in ((block.norm1, block.conv1), (block.norm2, block.conv2)):
        z = F.layer_norm(expected.transpose(1, 2), (8,), norm.weight, norm.bias, norm.eps)
        z = F.gelu(z.transpose(1, 2))
        terms = [F.conv1d(torch.roll(z, shifts=-o, dims=-1),
                          convolution.conv.weight[:, :, i:i+1])
                 for i, o in enumerate(offsets)]
        bias = convolution.conv.bias
        assert bias is not None
        expected = torch.stack(terms).sum(dim=0) + bias.view(1, -1, 1)
    expected = x + expected
    torch.testing.assert_close(block(x), expected, rtol=1e-10, atol=1e-10)

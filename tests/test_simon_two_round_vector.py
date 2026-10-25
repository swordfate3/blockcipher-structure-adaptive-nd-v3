"""Independent vector-channel candidate checks for two-round SIMON prediction."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cipher_experiments import training
from cipher_experiments.artifacts import file_sha256
from cipher_experiments.config import ExperimentConfig, load_config
from cipher_experiments.evaluation import load_checkpoint
from cipher_experiments.models.prediction import build_predictor
from cipher_experiments.models.simon_two_round import (
    TwoRoundLocalPredictor,
    TwoRoundVectorPredictor,
    _TwoRoundPredictor,
)
from cipher_experiments.two_round_diagnostics import two_round_local_diagnostics


def _rows(values: list[int]) -> Tensor:
    return torch.tensor(
        [[float((value >> (63 - bit)) & 1) for bit in range(64)] for value in values],
        dtype=torch.float32,
    )


def _manual_vector_forward(model: TwoRoundVectorPredictor, inputs: Tensor) -> Tensor:
    pieces: list[Tensor] = []
    for offset, module in zip((1, 8, 2), model.local_modules, strict=True):
        q = (model.target_bit + offset) % 32
        indices = (32 + q, (q + 1) % 32, (q + 8) % 32, (q + 2) % 32)
        local_input = inputs[:, indices].mul(2.0).sub(1.0)
        sequential = cast(nn.Sequential, module)
        first, second = cast(nn.Linear, sequential[0]), cast(nn.Linear, sequential[2])
        hidden = F.linear(local_input, first.weight, first.bias).tanh()
        local_logits = F.linear(hidden, second.weight, second.bias)
        pieces.append(torch.sigmoid(local_logits).mul(2.0).sub(1.0))
    combined_input = torch.cat(pieces, dim=1)
    first, last = cast(nn.Linear, model.combiner[0]), cast(nn.Linear, model.combiner[2])
    combined_hidden = F.linear(combined_input, first.weight, first.bias).tanh()
    combined = F.linear(combined_hidden, last.weight, last.bias)
    sign = 1.0 - 2.0 * inputs[:, model.target_bit : model.target_bit + 1]
    return combined * sign


def test_vector_parameter_count_initialization_independence_and_scalar_state_compatibility(
) -> None:
    torch.manual_seed(90210)
    vector = TwoRoundVectorPredictor(rounds=2, target_bit=17)
    state = vector.state_dict()
    assert sum(parameter.numel() for parameter in vector.parameters()) == 557
    local = cast(nn.Sequential, vector.local_modules[0])
    combiner = vector.combiner
    assert tuple(cast(nn.Linear, local[0]).weight.shape) == (16, 4)
    assert tuple(cast(nn.Linear, local[2]).weight.shape) == (4, 16)
    assert tuple(cast(nn.Linear, combiner[0]).weight.shape) == (8, 12)
    assert tuple(cast(nn.Linear, combiner[2]).weight.shape) == (1, 8)
    assert "indices" not in state
    assert tuple(state) == (
        "local_modules.0.0.weight", "local_modules.0.0.bias",
        "local_modules.0.2.weight", "local_modules.0.2.bias",
        "local_modules.1.0.weight", "local_modules.1.0.bias",
        "local_modules.1.2.weight", "local_modules.1.2.bias",
        "local_modules.2.0.weight", "local_modules.2.0.bias",
        "local_modules.2.2.weight", "local_modules.2.2.bias",
        "combiner.0.weight", "combiner.0.bias", "combiner.2.weight", "combiner.2.bias",
    )
    local_parameters = [list(module.parameters()) for module in vector.local_modules]
    assert all(
        left is not right
        for index, left_group in enumerate(local_parameters)
        for right_group in local_parameters[index + 1 :]
        for left in left_group
        for right in right_group
    )
    assert len({id(parameter) for parameter in vector.parameters()}) == len(
        list(vector.parameters())
    )

    torch.manual_seed(90210)
    same_initialization = TwoRoundVectorPredictor(rounds=2, target_bit=17)
    assert all(torch.equal(value, same_initialization.state_dict()[name])
               for name, value in state.items())

    scalar = TwoRoundLocalPredictor(rounds=2, target_bit=17)
    assert sum(parameter.numel() for parameter in scalar.parameters()) == 332
    assert tuple(scalar.state_dict()) == (
        "local_modules.0.0.weight", "local_modules.0.0.bias",
        "local_modules.0.2.weight", "local_modules.0.2.bias",
        "local_modules.1.0.weight", "local_modules.1.0.bias",
        "local_modules.1.2.weight", "local_modules.1.2.bias",
        "local_modules.2.0.weight", "local_modules.2.0.bias",
        "local_modules.2.2.weight", "local_modules.2.2.bias",
        "combiner.0.weight", "combiner.0.bias", "combiner.2.weight", "combiner.2.bias",
    )


@pytest.mark.parametrize("target_bit", range(32))
def test_vector_manual_forward_public_hooks_and_output_sign(target_bit: int) -> None:
    torch.manual_seed(101 + target_bit)
    model = TwoRoundVectorPredictor(rounds=2, target_bit=target_bit)
    scalar = TwoRoundLocalPredictor(rounds=2, target_bit=target_bit)
    assert torch.equal(model.indices, scalar.indices)
    values = [0, (1 << 64) - 1, 0xA1234567FEDCBA98, 0xFEDCBA9876543210]
    values.extend(1 << position for position in range(64))
    inputs = _rows(values)

    captured_inputs: list[Tensor] = []
    captured_local_outputs: list[Tensor] = []
    combined_inputs: list[Tensor] = []
    combined_outputs: list[Tensor] = []
    handles = []
    for module in model.local_modules:
        handles.append(module.register_forward_pre_hook(
            lambda _module, args: captured_inputs.append(args[0].detach().clone())
        ))
        handles.append(module.register_forward_hook(
            lambda _module, _args, output: captured_local_outputs.append(output.detach().clone())
        ))
    handles.append(model.combiner.register_forward_pre_hook(
        lambda _module, args: combined_inputs.append(args[0].detach().clone())
    ))
    handles.append(model.combiner.register_forward_hook(
        lambda _module, _args, output: combined_outputs.append(output.detach().clone())
    ))
    actual = model(inputs)
    for handle in handles:
        handle.remove()

    assert len(captured_inputs) == len(captured_local_outputs) == 3
    assert all(value.shape == (len(inputs), 4) for value in captured_inputs)
    assert all(value.shape == (len(inputs), 4) for value in captured_local_outputs)
    for index, offset in enumerate((1, 8, 2)):
        q = (target_bit + offset) % 32
        positions = (32 + q, (q + 1) % 32, (q + 8) % 32, (q + 2) % 32)
        assert torch.equal(captured_inputs[index], inputs[:, positions].mul(2.0).sub(1.0))
    assert combined_inputs[0].shape == (len(inputs), 12)
    assert torch.equal(combined_inputs[0], torch.cat([
        torch.sigmoid(value).mul(2.0).sub(1.0) for value in captured_local_outputs
    ], dim=1))
    assert torch.equal(combined_outputs[0], _manual_vector_forward(model, inputs) * (
        1.0 - 2.0 * inputs[:, target_bit : target_bit + 1]
    ))
    torch.testing.assert_close(actual, _manual_vector_forward(model, inputs), rtol=0, atol=0)
    assert actual.shape == (len(inputs), 1) and bool(torch.isfinite(actual).all())


def test_vector_public_x_target_sign_for_every_high_half_bit() -> None:
    inputs = _rows([0, (1 << 64) - 1, 0xA1234567FEDCBA98])
    for target_bit in range(32):
        model = TwoRoundVectorPredictor(target_bit=target_bit)
        final = cast(nn.Linear, model.combiner[2])
        with torch.no_grad():
            final.weight.zero_()
            final.bias.fill_(2.0)
        expected = (1.0 - 2.0 * inputs[:, target_bit : target_bit + 1]) * 2.0
        assert torch.equal(model(inputs), expected)


def test_vector_only_accepts_public_inputs_and_fixed_contract() -> None:
    assert not issubclass(TwoRoundVectorPredictor, TwoRoundLocalPredictor)
    assert issubclass(TwoRoundVectorPredictor, _TwoRoundPredictor)
    assert isinstance(build_predictor(
        16, model_kind="simon-r2-vector", rounds=2, target_bit=31,
    ), TwoRoundVectorPredictor)
    assert tuple(inspect.signature(TwoRoundVectorPredictor).parameters) == (
        "rounds", "target_bit"
    )
    for kwargs in ({"rounds": 1}, {"rounds": 3}, {"target_bit": 32},
                   {"rounds": True}, {"target_bit": True}):
        with pytest.raises((TypeError, ValueError)):
            TwoRoundVectorPredictor(**kwargs)
    with pytest.raises(ValueError, match="hidden_width"):
        build_predictor(32, model_kind="simon-r2-vector", rounds=2)
    invalid_factory_kwargs: tuple[dict[str, Any], ...] = (
        {"feature_mode": "support"}, {"prediction_mode": "public-residual"},
        {"rounds": 3}, {"target_bit": 32},
    )
    for kwargs in invalid_factory_kwargs:
        with pytest.raises(ValueError):
            build_predictor(16, model_kind="simon-r2-vector", **kwargs)

    inputs = _rows([0, (1 << 64) - 1, 0xA1234567FEDCBA98])
    with pytest.raises(ValueError, match="shape"):
        TwoRoundVectorPredictor()(torch.zeros((1, 63), dtype=torch.float32))
    with pytest.raises(ValueError, match="binary"):
        TwoRoundVectorPredictor()(torch.full((1, 64), 0.5, dtype=torch.float32))
    assert TwoRoundVectorPredictor()(inputs).shape == (3, 1)

    path = (Path(__file__).resolve().parents[1]
            / "src/cipher_experiments/models/simon_two_round.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    roots.update(alias.name.split(".")[0]
                 for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for alias in node.names)
    assert roots <= {"torch", "prediction", "typing", "__future__"}
    assert tuple(inspect.signature(TwoRoundVectorPredictor.forward).parameters) == (
        "self", "inputs"
    )
    for constructor in (TwoRoundVectorPredictor.__init__, TwoRoundVectorPredictor.forward):
        assert "key" not in constructor.__annotations__
        assert "labels" not in constructor.__annotations__
        assert "trace" not in constructor.__annotations__


@pytest.mark.parametrize("patch", [
    {"dataset": "synthetic"},
    {"experiment": "innovation1", "dataset": "differential"}, {"rounds": 3},
    {"target_bit": 32}, {"hidden_width": 32}, {"model_features": "support"},
    {"prediction_mode": "public-residual"},
])
def test_vector_config_enforces_specialized_boundaries(patch: dict[str, Any]) -> None:
    config = ExperimentConfig(
        name="vector-contract", dataset="simon", model_kind="simon-r2-vector",
        hidden_width=16, rounds=2, target_bit=0, label_mode="null",
    )
    with pytest.raises(ValueError, match="simon-r2"):
        ExperimentConfig(**{**config.to_dict(), **patch})


def test_vector_config_loads_strictly(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="vector-load", dataset="simon", model_kind="simon-r2-vector",
        hidden_width=16, rounds=2, target_bit=31, label_mode="null",
    )
    path = tmp_path / "vector.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert load_config(path) == config
    path.write_text(json.dumps({**config.to_dict(), "unknown": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="known fields"):
        load_config(path)


def test_vector_diagnostics_report_each_local_channel_and_reject_test() -> None:
    model = TwoRoundVectorPredictor(rounds=2, target_bit=0)
    generator = torch.Generator().manual_seed(1664)
    train_x = torch.randint(0, 2, (12, 64), generator=generator).float()
    dev_x = torch.randint(0, 2, (7, 64), generator=generator).float()
    train_y = torch.randint(0, 2, (12, 1), generator=generator).float()
    dev_y = torch.randint(0, 2, (7, 1), generator=generator).float()
    report = two_round_local_diagnostics(
        model, {"train": (train_x, train_y), "dev": (dev_x, dev_y)},
        batch_size=5, device=torch.device("cpu"),
    )
    assert report["model"] == "simon-r2-vector"
    assert report["scope"] == ["train", "dev"]
    modules = report["local_hidden_sigmoid_corners"]["modules"]
    assert len(modules) == 3
    for module in modules:
        assert module["channel_count"] == 4
        assert len(module["channels"]) == 4
        for channel in module["channels"]:
            assert channel["corner_count"] == 16
            assert 0 <= channel["min"] <= channel["mean"] <= channel["max"] <= 1
    with pytest.raises(ValueError, match="TEST is forbidden"):
        two_round_local_diagnostics(
            model,
            {"train": (train_x, train_y), "dev": (dev_x, dev_y),
             "test": (dev_x, dev_y)},
            batch_size=4, device=torch.device("cpu"),
        )


def test_vector_null_training_selected_diagnostics_precede_test_labels_and_strict_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig(
        name="vector-null-fixture", dataset="simon", model_kind="simon-r2-vector",
        hidden_width=16, rounds=2, target_bit=19, label_mode="null",
        train_rows=12, dev_rows=8, test_rows=6, batch_size=4, epochs=1,
    )
    generator = torch.Generator().manual_seed(7881)
    inputs = {
        split: torch.randint(0, 2, (size, 64), generator=generator).float()
        for split, size in (("train", 12), ("dev", 8), ("test", 6))
    }
    monkeypatch.setattr(training, "input_panels", lambda _config: inputs)
    from cipher_experiments.data import prediction as prediction_data

    def forbidden_encrypt(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("null-labeled vector fixture must not encrypt")

    monkeypatch.setattr(prediction_data, "_encrypt", forbidden_encrypt)
    output = tmp_path / "vector-null-run"
    calls: list[str] = []
    original_labels = training.labels_for
    original_diagnostics = two_round_local_diagnostics
    original_load = load_checkpoint

    def checked_load(*args: Any, **kwargs: Any) -> Any:
        result = original_load(*args, **kwargs)
        calls.append("loaded")
        return result

    def labels(cfg: ExperimentConfig, x: Tensor, split: str) -> Tensor:
        if split == "test":
            assert calls[-1] == "diagnostics"
            selected = json.loads((output / "selection.json").read_text(encoding="utf-8"))
            assert selected["test_labels_generated"] is False
        calls.append(split)
        return original_labels(cfg, x, split)

    def diagnostics(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert (output / "selection.json").is_file()
        assert calls[-1] == "loaded"
        assert set(args[1]) == {"train", "dev"}
        calls.append("diagnostics")
        return original_diagnostics(*args, **kwargs)

    monkeypatch.setattr(training, "labels_for", labels)
    monkeypatch.setattr(training, "load_checkpoint", checked_load)
    monkeypatch.setattr(training, "two_round_local_diagnostics", diagnostics)
    result = training.train(config, output)
    assert calls == ["train", "dev", "loaded", "diagnostics", "test"]
    assert result["updates"] == 3
    metadata = json.loads((output / "model_metadata.json").read_text(encoding="utf-8"))
    assert metadata["kind"] == "simon-r2-vector"
    assert metadata["parameters"] == 557
    assert metadata["role"] == "neural-candidate" and metadata["key_free"] is True
    assert "independent" in metadata["parameter_sharing"]
    report = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
    assert report["model"] == "simon-r2-vector"
    assert set(report["splits"]) == {"train", "dev"}
    assert "test" not in json.dumps(report).lower()

    checkpoint = output / "best.pt"
    model, saved_config, _, _ = load_checkpoint(
        checkpoint, file_sha256(checkpoint), device=torch.device("cpu")
    )
    assert isinstance(model, TwoRoundVectorPredictor)
    assert saved_config == config
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    name = next(iter(payload["state_dict"]))
    for suffix, malformed in (
        ("dtype", payload["state_dict"][name].double()),
        ("shape", payload["state_dict"][name].reshape(-1)[:1]),
    ):
        corrupted = dict(payload)
        corrupted["state_dict"] = dict(payload["state_dict"])
        corrupted["state_dict"][name] = malformed
        corrupt_path = tmp_path / f"corrupt-{suffix}.pt"
        torch.save(corrupted, corrupt_path)
        with pytest.raises(ValueError, match="state mismatch"):
            load_checkpoint(corrupt_path, file_sha256(corrupt_path), device=torch.device("cpu"))

"""Public-feature, architecture, and strict integration checks for two-round models."""

from __future__ import annotations

import ast
import inspect
import json
import random
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from cipher_experiments import training
from cipher_experiments.artifacts import file_sha256
from cipher_experiments.cli import main
from cipher_experiments.config import ExperimentConfig
from cipher_experiments.evaluation import load_checkpoint
from cipher_experiments.models.prediction import build_predictor
from cipher_experiments.models.simon_two_round import (
    TwoRoundLocalPredictor,
    TwoRoundTablePredictor,
)


def _rows(values: list[int]) -> Tensor:
    return torch.tensor(
        [[float((value >> (63 - bit)) & 1) for bit in range(64)] for value in values],
        dtype=torch.float32,
    )


def _bit(word: int, position: int) -> int:
    return (word >> (31 - (position % 32))) & 1


def _public_table_oracle(plaintext: int, target_bit: int) -> tuple[int, int]:
    x, y = plaintext >> 32, plaintext & 0xFFFFFFFF

    def f(word: int, position: int) -> int:
        return (_bit(word, position + 1) & _bit(word, position + 8)) ^ _bit(
            word, position + 2
        )

    u = y ^ sum(f(x, position) << (31 - position) for position in range(32))
    a = _bit(u, target_bit + 1)
    b = _bit(u, target_bit + 8)
    residual = _bit(x, target_bit) ^ _bit(u, target_bit + 2)
    return (2 * a + b, residual)


def _table_target(plaintext: int, target_bit: int, a: int, b: int, c: int) -> int:
    cell, t = _public_table_oracle(plaintext, target_bit)
    return t ^ (((cell // 2) ^ a) & ((cell % 2) ^ b)) ^ c


def test_public_four_cell_formula_for_every_high_target_bit() -> None:
    rng = random.Random(8841)
    values = [0, (1 << 64) - 1]
    values.extend(1 << position for position in range(64))
    values.extend(rng.getrandbits(64) for _ in range(64))
    inputs = _rows(values)
    # These arbitrary constants and targets exist only in this public synthetic test;
    # no production model receives them and no cipher labels or states are generated.
    a, b, c = 1, 0, 1
    for target_bit in range(32):
        model = TwoRoundTablePredictor(rounds=2, target_bit=target_bit)
        with torch.no_grad():
            for cell in range(4):
                residual = (((cell // 2) ^ a) & ((cell % 2) ^ b)) ^ c
                model.table_logits[cell] = 2.5 if residual else -2.5
        cell_indices, t = model.public_table_features(inputs)
        expected_pairs = [_public_table_oracle(value, target_bit) for value in values]
        expected_cells = torch.tensor([cell for cell, _ in expected_pairs])
        expected_t = torch.tensor([[float(bit)] for _, bit in expected_pairs])
        assert torch.equal(cell_indices, expected_cells)
        assert torch.equal(t, expected_t)

        synthetic_labels = torch.tensor(
            [[float(_table_target(value, target_bit, a, b, c))] for value in values]
        )
        logits = model(inputs)
        assert torch.equal((logits > 0), synthetic_labels.bool())
        table_targets = torch.tensor(
            [[float((((cell // 2) ^ a) & ((cell % 2) ^ b)) ^ c)]
             for cell, _ in expected_pairs]
        )
        residual_logits = model.table_logits.index_select(0, cell_indices).unsqueeze(1)
        absolute_bce = F.binary_cross_entropy_with_logits(logits, synthetic_labels)
        residual_bce = F.binary_cross_entropy_with_logits(residual_logits, table_targets)
        torch.testing.assert_close(absolute_bce, residual_bce, rtol=1e-6, atol=1e-7)


def test_parameter_counts_zero_init_nonpersistent_indices_and_disjoint_local_weights() -> None:
    table = TwoRoundTablePredictor()
    local = TwoRoundLocalPredictor()
    assert table.table_logits.shape == (4,)
    assert torch.equal(table.table_logits, torch.zeros(4))
    assert sum(parameter.numel() for parameter in table.parameters()) == 4
    assert sum(parameter.numel() for parameter in local.parameters()) == 332
    assert "indices" not in table.state_dict() and "indices" not in local.state_dict()
    assert table.indices.tolist() == local.indices.tolist()

    local_parameters = [list(module.parameters()) for module in local.local_modules]
    assert all(
        left_parameter is not right_parameter
        for index, left in enumerate(local_parameters)
        for right in local_parameters[index + 1 :]
        for left_parameter in left
        for right_parameter in right
    )
    assert len({id(parameter) for parameter in local.parameters()}) == len(
        list(local.parameters())
    )


@pytest.mark.parametrize("target_bit", [0, 31])
def test_local_modules_receive_exact_centered_public_inputs_and_backpropagate(
    target_bit: int,
) -> None:
    model = TwoRoundLocalPredictor(rounds=2, target_bit=target_bit)
    inputs = _rows([0, (1 << 64) - 1, 0xA1234567FEDCBA98, 0xFEDCBA9876543210])
    captured: list[Tensor] = []
    handles = [
        cast(torch.nn.Sequential, module)[0].register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach().clone())
        )
        for module in model.local_modules
    ]
    combiner_input: list[Tensor] = []
    combiner_output: list[Tensor] = []
    handles.append(model.combiner.register_forward_pre_hook(
        lambda _module, args: combiner_input.append(args[0].detach().clone())
    ))
    handles.append(model.combiner.register_forward_hook(
        lambda _module, _args, output: combiner_output.append(output.detach().clone())
    ))
    logits = model(inputs)
    for handle in handles:
        handle.remove()

    for module_index, offset in enumerate((1, 8, 2)):
        q = (target_bit + offset) % 32
        positions = (32 + q, (q + 1) % 32, (q + 8) % 32, (q + 2) % 32)
        expected = inputs[:, positions].mul(2.0).sub(1.0)
        assert torch.equal(captured[module_index], expected)
    sign = 1.0 - 2.0 * inputs[:, target_bit : target_bit + 1]
    assert logits.shape == (len(inputs), 1)
    assert torch.isfinite(logits).all()
    assert torch.equal(combiner_input[0], torch.cat([
        torch.sigmoid(module(value)).mul(2.0).sub(1.0)
        for module, value in zip(model.local_modules, captured, strict=True)
    ], dim=1))
    assert torch.equal(logits, combiner_output[0] * sign)

    labels = inputs[:, 0:1].clone()
    F.binary_cross_entropy_with_logits(logits, labels).backward()  # type: ignore[no-untyped-call]
    for module in (*model.local_modules, model.combiner):
        for parameter in module.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) > 0


def test_local_public_output_sign_for_every_high_half_bit() -> None:
    inputs = _rows([0, (1 << 64) - 1, 0xA1234567FEDCBA98])
    for target_bit in range(32):
        model = TwoRoundLocalPredictor(target_bit=target_bit)
        with torch.no_grad():
            final_layer = model.combiner[-1]
            assert isinstance(final_layer, torch.nn.Linear)
            final_layer.weight.zero_()
            final_layer.bias.fill_(2.0)
        expected = (1.0 - 2.0 * inputs[:, target_bit : target_bit + 1]) * 2.0
        assert torch.equal(model(inputs), expected)


def test_constructor_factory_and_forward_enforce_fixed_two_round_contract() -> None:
    assert isinstance(build_predictor(16, model_kind="simon-r2-table", rounds=2),
                      TwoRoundTablePredictor)
    assert isinstance(build_predictor(16, model_kind="simon-r2-local", rounds=2),
                      TwoRoundLocalPredictor)
    for constructor in (TwoRoundTablePredictor, TwoRoundLocalPredictor):
        assert tuple(inspect.signature(constructor).parameters) == ("rounds", "target_bit")
        assert tuple(inspect.signature(constructor.forward).parameters) == ("self", "inputs")
        for kwargs in ({"rounds": 1}, {"rounds": 3}, {"target_bit": 32}):
            with pytest.raises(ValueError):
                constructor(**kwargs)
        with pytest.raises(TypeError, match="built-in int"):
            constructor(rounds=True)
        with pytest.raises(TypeError, match="built-in int"):
            constructor(target_bit=True)
        model = constructor()
        with pytest.raises(ValueError, match="shape"):
            model(torch.zeros((0, 64)))
        with pytest.raises(ValueError, match="shape"):
            model(torch.zeros((1, 63)))
        with pytest.raises(TypeError, match="float32"):
            model(torch.zeros((1, 64), dtype=torch.float64))
        with pytest.raises(ValueError, match="finite"):
            model(torch.full((1, 64), float("nan")))
        with pytest.raises(ValueError, match="binary"):
            model(torch.full((1, 64), 0.5))
        with pytest.raises(ValueError, match="same device"):
            model.to(device="meta")(torch.zeros((1, 64)))
    with pytest.raises(ValueError, match="hidden_width"):
        build_predictor(64, model_kind="simon-r2-local")
    with pytest.raises(ValueError, match="simon-r2"):
        build_predictor(16, model_kind="simon-r2-local", rounds=2, feature_mode="support")
    with pytest.raises(ValueError, match="simon-r2"):
        build_predictor(16, model_kind="simon-r2-local", rounds=2,
                        prediction_mode="public-residual")
    with pytest.raises(ValueError, match="rounds"):
        build_predictor(16, model_kind="simon-r2-local", rounds=3)
    with pytest.raises(ValueError, match="target_bit"):
        build_predictor(16, model_kind="simon-r2-local", rounds=2, target_bit=32)


@pytest.mark.parametrize("model_kind", ["simon-r2-table", "simon-r2-local"])
def test_strict_config_training_metadata_and_explicit_evaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_kind: str
) -> None:
    config = ExperimentConfig(
        name="two-round-public-fixture", dataset="simon", model_kind=model_kind,
        hidden_width=16, rounds=2, target_bit=7, label_mode="null",
        train_rows=32, dev_rows=16, test_rows=16, batch_size=16, epochs=1,
    )
    for patch in (
        {"dataset": "synthetic"}, {"experiment": "innovation1", "dataset": "differential"},
        {"rounds": 3}, {"target_bit": 32}, {"hidden_width": 64},
        {"model_features": "support"}, {"prediction_mode": "public-residual"},
    ):
        with pytest.raises(ValueError, match="simon-r2"):
            ExperimentConfig(**{**config.to_dict(), **patch})

    generator = torch.Generator().manual_seed(1781)
    panels = {
        split: torch.randint(0, 2, (rows, 64), generator=generator).float()
        for split, rows in (("train", 32), ("dev", 16), ("test", 16))
    }
    monkeypatch.setattr(training, "input_panels", lambda _config: panels)
    from cipher_experiments.data import prediction as prediction_data

    def forbidden_encrypt(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("null-label two-round fixture must not encrypt")

    monkeypatch.setattr(prediction_data, "_encrypt", forbidden_encrypt)
    calls: list[str] = []
    labels_by_split: dict[str, Tensor] = {}
    original_labels = training.labels_for
    original_load = load_checkpoint

    def synthetic_labels(_config: ExperimentConfig, x: Tensor, split: str) -> Tensor:
        if split == "test":
            assert calls[-1] == "loaded"
            selection = json.loads((output / "selection.json").read_text())
            assert selection["test_labels_generated"] is False
        calls.append(split)
        labels = original_labels(_config, x, split)
        labels_by_split[split] = labels
        return labels

    def checked_load(*args: Any, **kwargs: Any) -> Any:
        result = original_load(*args, **kwargs)
        calls.append("loaded")
        return result

    output = tmp_path / model_kind
    monkeypatch.setattr(training, "labels_for", synthetic_labels)
    monkeypatch.setattr(training, "load_checkpoint", checked_load)
    result = training.train(config, output)
    assert calls == ["train", "dev", "loaded", "test"]
    assert result["updates"] == 2

    metadata = json.loads((output / "model_metadata.json").read_text())
    assert metadata["supervision"] == "absolute final target-bit BCE only"
    assert metadata["key_free"] is True
    assert metadata["input"] == (
        "public plaintext bits only; no key or intermediate states; "
        "final absolute bit labels only"
    )
    if model_kind == "simon-r2-table":
        assert metadata["role"] == "positive-control" and metadata["parameters"] == 4
        expected_counts = [0, 0, 0, 0]
        expected_ones = [0, 0, 0, 0]
        for row, label in zip(panels["train"], labels_by_split["train"], strict=True):
            integer = sum(int(bit) << (63 - index) for index, bit in enumerate(row.tolist()))
            cell, residual = _public_table_oracle(integer, config.target_bit)
            expected_counts[cell] += 1
            expected_ones[cell] += int(bool(label) ^ bool(residual))
        table_stats = metadata["train_table"]
        assert table_stats["counts"] == expected_counts
        assert table_stats["residual_ones"] == expected_ones
        assert table_stats["covered_all_cells"] == all(
            count > 0 for count in expected_counts
        )
        assert table_stats["conflicting_cells"] == [
            cell for cell in range(4) if expected_ones[cell] not in (0, expected_counts[cell])
        ]
    else:
        assert metadata["role"] == "neural-candidate" and metadata["parameters"] == 332
        assert "independent" in metadata["parameter_sharing"]

    checkpoint = output / "best.pt"
    model, saved, _, _ = load_checkpoint(checkpoint, file_sha256(checkpoint),
                                         device=torch.device("cpu"))
    assert saved == config
    assert isinstance(model, TwoRoundTablePredictor if model_kind == "simon-r2-table"
                      else TwoRoundLocalPredictor)
    evaluation_output = tmp_path / f"{model_kind}-evaluation"
    monkeypatch.setattr(
        training, "datasets",
        lambda _config: {split: (panels[split], labels_by_split[split])
                         for split in ("train", "dev", "test")},
    )
    assert main(["evaluate", "--checkpoint", str(checkpoint), "--sha256",
                 file_sha256(checkpoint), "--output", str(evaluation_output)]) == 0
    evaluation = json.loads((evaluation_output / "evaluation.json").read_text())
    assert evaluation["metrics"] == result["metrics"]["test"]

    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    name = next(iter(payload["state_dict"]))
    for suffix, malformed_value in (
        ("dtype", payload["state_dict"][name].double()),
        ("shape", payload["state_dict"][name].reshape(-1)[:1]),
    ):
        corrupted = dict(payload)
        corrupted["state_dict"] = dict(payload["state_dict"])
        corrupted["state_dict"][name] = malformed_value
        malformed = tmp_path / f"{model_kind}-{suffix}.pt"
        torch.save(corrupted, malformed)
        with pytest.raises(ValueError, match="state mismatch"):
            load_checkpoint(malformed, file_sha256(malformed), device=torch.device("cpu"))


def test_production_model_module_imports_no_cipher_or_data() -> None:
    path = Path(__file__).resolve().parents[1] / "src/cipher_experiments/models/simon_two_round.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    roots.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert roots <= {"torch", "prediction", "typing", "__future__"}
    for constructor in (TwoRoundTablePredictor, TwoRoundLocalPredictor):
        assert "key" not in constructor.__init__.__annotations__
        assert "labels" not in constructor.__init__.__annotations__

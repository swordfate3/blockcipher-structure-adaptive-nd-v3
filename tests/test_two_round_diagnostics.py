"""Aggregate-only diagnostics for the selected two-round local model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from cipher_experiments import training
from cipher_experiments.config import ExperimentConfig
from cipher_experiments.models.simon_two_round import TwoRoundLocalPredictor
from cipher_experiments.two_round_diagnostics import two_round_local_diagnostics


def _public_rows() -> list[list[float]]:
    rows: list[list[float]] = []
    for cell in range(8):
        row = [0.0] * 64
        for position, bit in zip((33, 40, 34), (cell >> 2, (cell >> 1) & 1, cell & 1),
                                 strict=True):
            row[position] = float(bit)
        rows.append(row)
    return rows


def _fixture() -> tuple[dict[str, tuple[Tensor, Tensor]], TwoRoundLocalPredictor]:
    base = _public_rows()
    train_x = torch.tensor([*base, base[0]], dtype=torch.float32)
    train_y = torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])
    dev_rows = [base[0].copy(), base[1].copy()]
    model = TwoRoundLocalPredictor(rounds=2, target_bit=0)
    included = int(model.indices[0])
    dev_rows[1][included] = 1.0 - dev_rows[1][included]
    dev_x = torch.tensor(dev_rows, dtype=torch.float32)
    dev_y = torch.tensor([[1.0], [0.0]])
    return {"train": (train_x, train_y), "dev": (dev_x, dev_y)}, model


def test_diagnostics_aggregate_public_groups_and_restore_state() -> None:
    panels, model = _fixture()
    for module in model.local_modules:
        final = module.get_submodule("2")
        assert isinstance(final, torch.nn.Linear)
        with torch.no_grad():
            final.weight.zero_()
            final.bias.zero_()
    output_layer = model.combiner[-1]
    assert isinstance(output_layer, torch.nn.Linear)
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.zero_()
    model.train()
    model.local_modules[1].eval()
    modes = [module.training for module in model.modules()]
    parameters = [parameter.detach().clone() for parameter in model.parameters()]
    panel_copies = {split: (x.clone(), y.clone()) for split, (x, y) in panels.items()}
    rng_state = torch.get_rng_state().clone()
    hook_counts = [len(module._forward_hooks) for module in model.modules()]
    observed_grad_modes: list[bool] = []

    for module in model.local_modules:
        module.register_forward_hook(
            lambda _module, _inputs, output: observed_grad_modes.append(
                torch.is_grad_enabled() or output.requires_grad
            )
        )
    original_hook_counts = [len(module._forward_hooks) for module in model.modules()]
    report = two_round_local_diagnostics(
        model, panels, batch_size=3, device=torch.device("cpu")
    )

    assert report["scope"] == ["train", "dev"]
    assert "verified as a cipher state" in report["state_interpretation"]
    corner_stats = report["local_hidden_sigmoid_corners"]["modules"]
    assert [item["corner_count"] for item in corner_stats] == [16] * 3
    assert all(item["min"] == item["max"] == item["mean"] == 0.5
               for item in report["local_hidden_sigmoid_corners"]["modules"])
    train = report["splits"]["train"]
    assert train["overall"] == {
        "rows": 9, "correct": 8, "errors": 1, "bce": pytest.approx(0.693147)
    }
    u_cells = train["public_u_groups"]["cells"]
    assert len(u_cells) == 8
    assert [cell["rows"] for cell in u_cells] == [2, 1, 1, 1, 1, 1, 1, 1]
    assert [cell["errors"] for cell in u_cells] == [1, 0, 0, 0, 0, 0, 0, 0]
    assert u_cells[0]["residual_labels"] == {
        "definition": "absolute_label XOR public X[target_bit]",
        "zero": 1, "one": 1, "consistent": False,
    }
    assert train["model_input_signatures"]["seen"] == {
        "groups": 8, "rows": 9, "errors": 1,
        "label_zero_rows": 8, "label_one_rows": 1,
        "unique_label_groups": 7, "conflicting_label_groups": 1,
    }
    assert train["model_input_signatures"]["unseen"] == {
        "groups": 0, "rows": 0, "errors": 0,
        "label_zero_rows": 0, "label_one_rows": 0,
        "unique_label_groups": 0, "conflicting_label_groups": 0,
    }
    dev = report["splits"]["dev"]["model_input_signatures"]
    assert dev["target_bit_in_signature"] is True
    assert dev["label_definition"] == "absolute output label Y, not public-Xj residual"
    assert dev["seen"]["rows"] == 1 and dev["seen"]["errors"] == 1
    assert dev["unseen"]["rows"] == 1 and dev["unseen"]["errors"] == 0
    assert dev["seen"]["groups"] == dev["unseen"]["groups"] == 1
    assert all(not mode for mode in observed_grad_modes)
    assert [module.training for module in model.modules()] == modes
    assert all(torch.equal(before, after) for before, after in zip(
        parameters, model.parameters(), strict=True
    ))
    assert all(torch.equal(original, current) for split in panels for original, current in zip(
        panel_copies[split], panels[split], strict=True
    ))
    assert torch.equal(torch.get_rng_state(), rng_state)
    # The caller's observer hooks remain, but no diagnostic hook is left behind.
    assert [len(module._forward_hooks) for module in model.modules()] == original_hook_counts
    assert all(current >= baseline for current, baseline in zip(
        original_hook_counts, hook_counts, strict=True
    ))
    json.dumps(report, allow_nan=False)


def test_diagnostics_reject_test_panel_and_remove_hooks_after_failure() -> None:
    panels, model = _fixture()
    with pytest.raises(ValueError, match="TEST is forbidden"):
        two_round_local_diagnostics(
            model, {**panels, "test": panels["dev"]}, batch_size=2,
            device=torch.device("cpu")
        )

    before = [len(module._forward_hooks) for module in model.modules()]
    handle = model.local_modules[1].register_forward_hook(
        lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic hook failure"))
    )
    with pytest.raises(RuntimeError, match="synthetic hook failure"):
        two_round_local_diagnostics(model, panels, batch_size=2, device=torch.device("cpu"))
    handle.remove()
    assert [len(module._forward_hooks) for module in model.modules()] == before


def test_training_writes_diagnostics_after_checkpoint_selection_before_test_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig(
        name="two-round-diagnostics-null", dataset="simon", model_kind="simon-r2-local",
        hidden_width=16, rounds=2, target_bit=7, label_mode="null", train_rows=8,
        dev_rows=4, test_rows=4, batch_size=4, epochs=1,
    )
    generator = torch.Generator().manual_seed(7123)
    inputs = {
        "train": torch.randint(0, 2, (8, 64), generator=generator).float(),
        "dev": torch.randint(0, 2, (4, 64), generator=generator).float(),
        "test": torch.randint(0, 2, (4, 64), generator=generator).float(),
    }
    monkeypatch.setattr(training, "input_panels", lambda _config: inputs)
    output = tmp_path / "null-run"
    calls: list[str] = []
    original_labels = training.labels_for

    def labels(cfg: ExperimentConfig, x: Tensor, split: str) -> Tensor:
        if split == "test":
            assert calls[-1] == "diagnostics"
            selection = json.loads((output / "selection.json").read_text())
            assert selection["test_labels_generated"] is False
        calls.append(split)
        return original_labels(cfg, x, split)

    from cipher_experiments.two_round_diagnostics import two_round_local_diagnostics as diagnose

    def checked_diagnostics(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert (output / "selection.json").is_file()
        assert not (output / "diagnostics.json").exists()
        assert set(args[1]) == {"train", "dev"}
        calls.append("diagnostics")
        return diagnose(*args, **kwargs)

    monkeypatch.setattr(training, "labels_for", labels)
    monkeypatch.setattr(training, "two_round_local_diagnostics", checked_diagnostics)
    result = training.train(config, output)
    assert calls == ["train", "dev", "diagnostics", "test"]
    assert result["updates"] == 2
    report = json.loads((output / "diagnostics.json").read_text())
    assert report["model"] == "simon-r2-local"
    assert set(report["splits"]) == {"train", "dev"}
    assert not any("test" in key.lower() for key in report)

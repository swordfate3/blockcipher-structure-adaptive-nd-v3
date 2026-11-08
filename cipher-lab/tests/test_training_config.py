"""Task1 public settings/ledger tests; no tensors, labels or cipher calls."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from cipher_lab import config
from cipher_lab.cli import main
from cipher_lab.execution import Clocks, Meter, Stopped


def prediction() -> dict[str, Any]:
    return {
        "schema": 1, "method": "bit-prediction-v1", "model": {"hidden_width": 32},
        "training": {"train": 12, "dev": 4, "test": 6, "batch_size": 3,
                     "epochs": 2, "learning_rate": 0.001, "init_seed": 0,
                     "sample_seed": 1, "shuffle_seed": 4294967295},
        "rounds": 13, "target_bit": 63, "label_mode": "cipher",
    }


def shared() -> dict[str, Any]:
    value = prediction()
    return {
        "schema": 1, "method": "shared-m-four-task-v1",
        "model": {"stem_width": 32, "channels": 64, "norm_groups": 8,
                  "suffix_kernels": [3, 3, 3, 3, 3], "head_hidden": 128,
                  "router_hidden": 32},
        "training": value["training"],
        "tasks": [{"cipher": cipher, "rounds": rounds, "difference": delta}
                  for cipher, rounds, delta in [
                      ("SIMECK", 16, 0x40), ("PRESENT", 7, 0xD00000),
                      ("SIMON", 13, 0x40), ("SKINNY", 8, 0x2000)]],
    }


def policy() -> dict[str, Any]:
    return {"schema": 2, "wall_seconds": 10, "cpu_seconds": 10,
            "deadline_utc": "2000-01-01T00:00:00Z", "real_blocks": 1000,
            "null_words": 1000, "optimizer_updates": 1000, "forward_rows": 1000,
            "rss_stop_bytes": 1000, "output_bytes": 1000}


def save(tmp_path: Path, value: object, name: str = "recipe.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def load(path: Path) -> Any:
    loader = getattr(config, "load_experiment", None)
    assert callable(loader), "Task1 requires explicit load_experiment dispatch"
    return loader(path)


@pytest.mark.parametrize("factory", [prediction, shared])
def test_load_once_roundtrip_and_frozen_specs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             factory: Any) -> None:
    value = factory()
    path = save(tmp_path, value)
    original = config._load
    reads: list[Path] = []

    def once(source: Path) -> object:
        reads.append(source)
        return original(source)

    monkeypatch.setattr(config, "_load", once)
    recipe = load(path)
    assert reads == [path]
    assert recipe.to_dict() == value
    for node in (recipe, recipe.model, recipe.training):
        with pytest.raises(FrozenInstanceError):
            setattr(node, next(iter(node.__dataclass_fields__)), 99)
    detached = recipe.to_dict()
    detached["training"]["train"] = 99
    assert recipe.training.train == 12
    if "tasks" in value:
        assert type(recipe.tasks) is tuple
        assert type(recipe.model.suffix_kernels) is tuple
        with pytest.raises(FrozenInstanceError):
            setattr(recipe.tasks[0], "rounds", 1)


@pytest.mark.parametrize("factory", [prediction, shared])
@pytest.mark.parametrize("node", ["root", "model", "training"])
@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_strict_fields(tmp_path: Path, factory: Any, node: str, mutation: str) -> None:
    value = factory()
    target = value if node == "root" else value[node]
    if mutation == "extra":
        target["private_key_seed"] = 123
    else:
        del target[next(iter(target))]
    with pytest.raises(ValueError):
        load(save(tmp_path, value))


@pytest.mark.parametrize("factory", [prediction, shared])
@pytest.mark.parametrize("field,bad", [
    ("train", 0), ("train", 2000001), ("train", True), ("train", 1.0),
    ("dev", 0), ("dev", 2000001), ("test", 0), ("test", 2000001),
    ("batch_size", 0), ("batch_size", True), ("batch_size", 5),
    ("epochs", 0), ("epochs", 100001), ("epochs", True),
    ("learning_rate", 0), ("learning_rate", -0.1), ("learning_rate", 1.01),
    ("learning_rate", True), ("learning_rate", "0.1"),
    ("learning_rate", float("nan")), ("learning_rate", float("inf")),
    ("learning_rate", 10**400),
    ("init_seed", -1), ("sample_seed", 4294967296), ("shuffle_seed", True),
])
def test_training_bounds(tmp_path: Path, factory: Any, field: str, bad: object) -> None:
    value = factory()
    value["training"][field] = bad
    with pytest.raises(ValueError):
        load(save(tmp_path, value))


@pytest.mark.parametrize("field", ["init_seed", "sample_seed", "shuffle_seed"])
@pytest.mark.parametrize("bad", [-1, 4294967296, True, 1.0, "1", None])
def test_each_seed_is_strict(tmp_path: Path, field: str, bad: object) -> None:
    value = prediction()
    value["training"][field] = bad
    with pytest.raises(ValueError):
        load(save(tmp_path, value))


@pytest.mark.parametrize("rows,epochs,lr,seed", [(1, 1, 1, 0), (2000000, 100000, 1e-12, 2**32-1)])
def test_training_positive_boundaries(tmp_path: Path, rows: int, epochs: int,
                                      lr: float, seed: int) -> None:
    value = prediction()
    value["training"].update(train=rows, dev=rows, test=rows, batch_size=rows,
                              epochs=epochs, learning_rate=lr,
                              init_seed=seed, sample_seed=seed, shuffle_seed=seed)
    assert load(save(tmp_path, value)).to_dict() == value


@pytest.mark.parametrize("split", ["train", "dev", "test"])
@pytest.mark.parametrize("rows", [1, 3, 65537, 65538])
def test_shared_split_even_and_bounded(tmp_path: Path, split: str, rows: int) -> None:
    value = shared()
    value["training"].update(batch_size=1)
    value["training"][split] = rows
    with pytest.raises(ValueError):
        load(save(tmp_path, value))


@pytest.mark.parametrize("rows", [2, 65536])
def test_shared_split_boundaries(tmp_path: Path, rows: int) -> None:
    value = shared()
    value["training"].update(train=rows, dev=rows, test=rows, batch_size=rows)
    assert load(save(tmp_path, value)).to_dict() == value


@pytest.mark.parametrize("field,bad", [
    ("schema", 2), ("schema", True), ("method", "shared-m-four-task-v1"),
    ("rounds", 0), ("rounds", 45), ("rounds", True),
    ("target_bit", -1), ("target_bit", 64), ("target_bit", True),
    ("label_mode", "random"), ("label_mode", True),
    ("model", {"stem_width": 32}), ("training", []),
])
def test_prediction_fields(tmp_path: Path, field: str, bad: object) -> None:
    value = prediction()
    value[field] = bad
    with pytest.raises(ValueError):
        load(save(tmp_path, value))


@pytest.mark.parametrize("rounds,bit", [(1, 0), (44, 63)])
@pytest.mark.parametrize("label", ["cipher", "null"])
def test_prediction_round_bit_boundaries(tmp_path: Path, rounds: int, bit: int, label: str) -> None:
    value = prediction()
    value.update(rounds=rounds, target_bit=bit, label_mode=label)
    assert load(save(tmp_path, value)).to_dict() == value


@pytest.mark.parametrize("width", [0, 4097, True, 1.0, "32", None])
def test_mlp_width_rejected(tmp_path: Path, width: object) -> None:
    value = prediction()
    value["model"]["hidden_width"] = width
    with pytest.raises(ValueError):
        load(save(tmp_path, value))


@pytest.mark.parametrize("width", [1, 4096])
def test_mlp_width_boundaries(tmp_path: Path, width: int) -> None:
    value = prediction()
    value["model"]["hidden_width"] = width
    assert load(save(tmp_path, value)).model.hidden_width == width


@pytest.mark.parametrize("field", ["stem_width", "channels", "norm_groups", "head_hidden",
                                    "router_hidden"])
@pytest.mark.parametrize("bad", [0, 1025, True, 1.0, "32", None])
def test_shared_model_widths_rejected(tmp_path: Path, field: str, bad: object) -> None:
    value = shared()
    value["model"][field] = bad
    with pytest.raises(ValueError):
        load(save(tmp_path, value))


@pytest.mark.parametrize("field", ["stem_width", "channels"])
def test_groupnorm_must_divide_both_widths(tmp_path: Path, field: str) -> None:
    value = shared()
    value["model"][field] = 33
    with pytest.raises(ValueError):
        load(save(tmp_path, value))


@pytest.mark.parametrize("kernels", [[], [2], [0], [33], [True], [1.0], [3]*17, "3"])
def test_shared_kernels_rejected(tmp_path: Path, kernels: object) -> None:
    value = shared()
    value["model"]["suffix_kernels"] = kernels
    with pytest.raises(ValueError):
        load(save(tmp_path, value))


@pytest.mark.parametrize("width,kernels", [(1, [1]), (1024, [31]*16)])
def test_shared_model_boundaries(tmp_path: Path, width: int, kernels: list[int]) -> None:
    value = shared()
    value["model"].update(stem_width=width, channels=width, norm_groups=width,
                          head_hidden=width, router_hidden=width, suffix_kernels=kernels)
    assert load(save(tmp_path, value)).to_dict() == value


@pytest.mark.parametrize("index,max_rounds", [(0, 44), (1, 31), (2, 44), (3, 32)])
@pytest.mark.parametrize("boundary", ["low", "high", "over", "zero"])
def test_task_round_limits(tmp_path: Path, index: int, max_rounds: int, boundary: str) -> None:
    value = shared()
    rounds = {"low": 1, "high": max_rounds, "over": max_rounds+1, "zero": 0}[boundary]
    value["tasks"][index]["rounds"] = rounds
    if boundary in ("over", "zero"):
        with pytest.raises(ValueError):
            load(save(tmp_path, value))
    else:
        assert load(save(tmp_path, value)).tasks[index].rounds == rounds


@pytest.mark.parametrize("delta", [0, -1, 2**64, True, 1.0, "0x40", None])
def test_task_delta_nonzero_uint64(tmp_path: Path, delta: object) -> None:
    value = shared()
    value["tasks"][0]["difference"] = delta
    with pytest.raises(ValueError):
        load(save(tmp_path, value))


def test_task_delta_max_and_public_key_sizes(tmp_path: Path) -> None:
    value = shared()
    value["tasks"][0]["difference"] = 2**64-1
    recipe = load(save(tmp_path, value))
    assert recipe.tasks[0].difference == 2**64-1
    specs = importlib.import_module("cipher_lab.training_config")
    families = [(item.cipher, item.key_bits, item.block_bits) for item in specs.TASK_FAMILIES]
    assert families == [
        ("SIMECK", 128, 64), ("PRESENT", 80, 64), ("SIMON", 128, 64), ("SKINNY", 64, 64)]


@pytest.mark.parametrize("mutation", ["missing", "extra", "reorder", "duplicate", "unknown",
                                      "extra_field", "missing_field", "bool_rounds", "not_array"])
def test_four_ordered_task_declarations(tmp_path: Path, mutation: str) -> None:
    value = shared()
    tasks = value["tasks"]
    if mutation == "missing":
        tasks.pop()
    elif mutation == "extra":
        tasks.append(tasks[0])
    elif mutation == "reorder":
        tasks.reverse()
    elif mutation == "duplicate":
        tasks[1] = tasks[0]
    elif mutation == "unknown":
        tasks[0]["cipher"] = "AES"
    elif mutation == "extra_field":
        tasks[0]["key"] = 1
    elif mutation == "missing_field":
        del tasks[0]["difference"]
    elif mutation == "bool_rounds":
        tasks[0]["rounds"] = True
    else:
        value["tasks"] = {}
    with pytest.raises(ValueError):
        load(save(tmp_path, value))


@pytest.mark.parametrize("factory", [prediction, shared])
@pytest.mark.parametrize("raw_mutation", ["dup_root", "dup_nested", "nan", "infinity", "overflow"])
def test_strict_json_dispatch(tmp_path: Path, factory: Any, raw_mutation: str) -> None:
    raw = json.dumps(factory())
    old, new = {
        "dup_root": ('"schema": 1', '"schema": 1, "schema": 1'),
        "dup_nested": ('"epochs": 2', '"epochs": 2, "epochs": 2'),
        "nan": ('0.001', 'NaN'), "infinity": ('0.001', 'Infinity'),
        "overflow": ('0.001', '1e999'),
    }[raw_mutation]
    path = tmp_path / "recipe.json"
    path.write_text(raw.replace(old, new))
    with pytest.raises(ValueError):
        load(path)


@pytest.mark.parametrize("factory", [prediction, shared])
def test_method_cross_fields_and_unsupported_method(tmp_path: Path, factory: Any) -> None:
    value = factory()
    value["method"] = "integral-empirical-v1"
    with pytest.raises(ValueError):
        load(save(tmp_path, value))
    value["method"] = "unknown"
    with pytest.raises(ValueError):
        load(save(tmp_path, value))


def test_integral_dispatch_unchanged(tmp_path: Path) -> None:
    value = {"schema": 1, "method": "integral-empirical-v1",
             "cells": [{"rounds": 1, "active": [0, 1]}],
             "groups": {"discovery": 1, "screen": 1, "confirm": 1}, "chunk_points": 1}
    assert isinstance(load(save(tmp_path, value)), config.Recipe)
    assert config.load_recipe(save(tmp_path, value)).to_dict() == value


@pytest.mark.parametrize("field", list(policy()))
def test_policy_every_field_required(tmp_path: Path, field: str) -> None:
    value = policy()
    del value[field]
    with pytest.raises(ValueError):
        config.load_policy(save(tmp_path, value))


@pytest.mark.parametrize("field,bad", [("schema", 1), ("schema", True), ("schema", 2.0),
                                       ("optimizer_updates", -1), ("optimizer_updates", True),
                                       ("forward_rows", -1), ("forward_rows", 1.0)])
def test_policy_schema2_bounds(tmp_path: Path, field: str, bad: object) -> None:
    value = policy()
    value[field] = bad
    with pytest.raises(ValueError):
        config.load_policy(save(tmp_path, value))


def test_policy_schema2_roundtrip(tmp_path: Path) -> None:
    value = policy()
    assert config.load_policy(save(tmp_path, value)).to_dict() == value


@pytest.mark.parametrize("unit,reason", [("optimizer_updates", "OPTIMIZER_UPDATES_LIMIT"),
                                          ("forward_rows", "FORWARD_ROWS_LIMIT")])
def test_training_ledger_reservations(tmp_path: Path, unit: Any, reason: str) -> None:
    settings = config.load_policy(save(tmp_path, policy()))
    meter = Meter(replace(settings, deadline_utc="2099-01-01T00:00:00Z"),
                  Clocks(lambda: 0.0, lambda: 0.0, lambda: datetime(2030, 1, 1, tzinfo=UTC)),
                  lambda: 0)
    meter.reserve(unit, 999)
    meter.complete(unit, 998)
    with pytest.raises(ValueError):
        meter.complete(unit, 2)
    with pytest.raises(Stopped, match=reason):
        meter.reserve(unit, 2)
    meter.reserve(unit, 1)
    assert meter.snapshot()[f"attempted_{unit}"] == 1000
    assert meter.snapshot()[f"completed_{unit}"] == 998
    for bad in (-1, True, 1.0):
        with pytest.raises(ValueError):
            meter.reserve(unit, bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        meter.reserve("unknown", 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("mode,real,null,updates,forward", [
    ("cipher", 22, 0, 8, 54), ("null", 0, 22, 8, 54), ("shared", 704, 0, 32, 216)])
def test_check_four_units_no_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str], mode: str,
                                     real: int, null: int, updates: int, forward: int) -> None:
    value = shared() if mode == "shared" else prediction()
    if mode != "shared":
        value["label_mode"] = mode
    recipe = save(tmp_path, value)
    limits = policy()
    limits.update(real_blocks=real, null_words=null, optimizer_updates=updates-1,
                  forward_rows=forward)
    rules = save(tmp_path, limits, "policy.json")
    before = sorted(tmp_path.rglob("*"))
    assert main(["check", "--config", str(recipe), "--policy", str(rules)]) == 0
    result = json.loads(capsys.readouterr().out)
    estimate = result["estimate"]
    assert estimate["planned_real_blocks"] == real
    assert estimate["planned_null_words"] == null
    assert estimate["planned_optimizer_updates"] == updates
    assert estimate["planned_forward_rows"] == forward
    assert result["planned_calls_fit_policy"] == {
        "real_blocks": True, "null_words": True, "optimizer_updates": False, "forward_rows": True}
    assert sorted(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("factory", [prediction, shared])
def test_poisoned_static_check(tmp_path: Path, factory: Any) -> None:
    recipe = save(tmp_path, factory())
    rules = save(tmp_path, policy(), "policy.json")
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(part in fullname.split('.') for part in
               ('torch', 'numpy', 'ciphers', 'data', 'models',
                'training', 'integral', 'artifacts')):
            raise AssertionError('forbidden science or output import')
sys.meta_path.insert(0, Block())
from cipher_lab.cli import main
raise SystemExit(main(sys.argv[1:]))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               PYTHONDONTWRITEBYTECODE="1")
    before = sorted(tmp_path.rglob("*"))
    completed = subprocess.run([sys.executable, "-c", script, "check",
                                "--config", str(recipe), "--policy", str(rules)],
                               cwd=tmp_path, env=env, capture_output=True,
                               text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "Traceback" not in completed.stderr
    assert sorted(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("name,method", [("prediction-example", "bit-prediction-v1"),
                                         ("shared-m-example", "shared-m-four-task-v1")])
def test_shipped_examples(name: str, method: str) -> None:
    recipe = load(Path(__file__).resolve().parents[1] / "configs" / f"{name}.json")
    assert recipe.method == method
    assert recipe.schema == 1


def test_direct_typed_specs_enforce_contract(tmp_path: Path) -> None:
    recipe = load(save(tmp_path, shared()))
    with pytest.raises(ValueError):
        replace(recipe, tasks=tuple(reversed(recipe.tasks)))
    with pytest.raises(ValueError):
        replace(recipe.tasks[0], difference=0)
    with pytest.raises(ValueError):
        replace(recipe.model, suffix_kernels=[3])
    with pytest.raises(ValueError):
        replace(recipe.training, init_seed=True)
    with pytest.raises(ValueError):
        replace(recipe, model=object())
    with pytest.raises(ValueError):
        replace(recipe, training=object())

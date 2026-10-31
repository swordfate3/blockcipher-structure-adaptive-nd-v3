"""Expression contracts with fake ciphertexts/null only; never real encryption."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from cipher_experiments import checkpoints, training
from cipher_experiments.artifacts import file_sha256
from cipher_experiments.ciphers import simon
from cipher_experiments.config import ExperimentConfig, load_config
from cipher_experiments.data import prediction
from cipher_experiments.models.prediction import build_predictor

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"bit": (), "xor2": (0, 8), "xor3": (0, 8, 16), "xor-and": (0, 8, 16)}


@pytest.fixture(autouse=True)
def forbid_real_encrypt(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> int:
        raise AssertionError("real encryption forbidden in expression tests")
    monkeypatch.setattr(prediction, "_encrypt", forbidden)
    monkeypatch.setattr(simon, "encrypt", forbidden)


def label(x: Tensor, **changes: Any) -> Tensor:
    options: dict[str, Any] = dict(dataset="simon", rounds=6, target_bit=0,
                                   key_seed=190502, split="train")
    return prediction.make_labels(x, **(options | changes))


@pytest.mark.parametrize("expression,bits", [
    ("bit", ()), ("xor2", (0, 63)), ("xor3", (0, 8, 63)),
    ("xor-and", (0, 8, 63)), ("xor-and", (63, 0, 8)),
])
def test_complete_fake_truth_table_and_msb_endpoints(
        monkeypatch: pytest.MonkeyPatch, expression: str, bits: tuple[int, ...]) -> None:
    truths = list(itertools.product((0, 1), repeat=3))
    ciphertexts = [(a << 63) | (b << 55) | c for a, b, c in truths]
    x = prediction.make_inputs(8, 1, 1, seed=195601)["train"]
    calls: list[tuple[int, int, int]] = []
    stream = iter(ciphertexts)

    def fake(plaintext: int, key: int, rounds: int) -> int:
        calls.append((plaintext, key, rounds))
        return next(stream)

    monkeypatch.setattr(prediction, "_encrypt", fake)
    target = bits[0] if bits else 0
    actual = label(x, label_expression=expression, label_bits=bits, target_bit=target)
    expected = []
    for ciphertext in ciphertexts:
        selected = [(ciphertext >> (63 - bit)) & 1 for bit in (bits or (target,))]
        if expression == "bit":
            value = selected[0]
        elif expression == "xor2":
            value = selected[0] ^ selected[1]
        elif expression == "xor3":
            value = selected[0] ^ selected[1] ^ selected[2]
        else:
            value = selected[0] ^ (selected[1] & selected[2])
        expected.append([float(value)])
    assert actual.tolist() == expected
    assert actual.shape == (8, 1) and actual.dtype == torch.float32
    assert actual.is_contiguous() and not actual.requires_grad
    assert [p for p, _, _ in calls] == prediction._plaintexts(x)
    assert {k for _, k, _ in calls} == {random.Random(190502).getrandbits(128)}
    assert {r for _, _, r in calls} == {6}
    assert len(calls) == len(x)
    # This finite fake complete ciphertext subdomain is balanced, not a cipher claim.
    assert actual.sum().item() == 4


@pytest.mark.parametrize("target", [0, 63])
def test_default_bit_matches_frozen_legacy_path_across_chunks(
        monkeypatch: pytest.MonkeyPatch, target: int) -> None:
    x = prediction.make_inputs(8193, 1, 1, seed=17)["train"]
    calls: list[int] = []

    def fake(plaintext: int, key: int, rounds: int) -> int:
        calls.append(plaintext)
        return plaintext ^ 0xA55AA55A5AA55AA5

    monkeypatch.setattr(prediction, "_encrypt", fake)
    actual = label(x, target_bit=target)
    mask = 1 << (63 - target)
    # Exact previous implementation with fake ciphertext values, no real encrypt.
    expected = torch.tensor([[float(bool((p ^ 0xA55AA55A5AA55AA5) & mask))]
                             for p in prediction._plaintexts(x)], dtype=torch.float32)
    assert actual.numpy().tobytes() == expected.numpy().tobytes()
    assert len(calls) == len(x)
    explicit = label(x, target_bit=target, label_expression="bit", label_bits=[])
    assert torch.equal(explicit, actual)


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_null_stream_is_expression_independent_and_never_encrypts(split: str) -> None:
    x = torch.zeros(256, 64)
    material = (b"cipher-experiments/prediction-null/v1\0"
                + (195602).to_bytes(4, "big") + b"\0" + split.encode("ascii"))
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))
    expected = torch.tensor([[float(rng.getrandbits(1))] for _ in x])
    before = random.getstate()
    for expression, bits in ARMS.items():
        assert torch.equal(label(x, label_mode="null", null_seed=195602, split=split,
                                 label_expression=expression, label_bits=bits), expected)
    assert random.getstate() == before
    assert set(expected.flatten().tolist()) == {0., 1.}


def test_synthetic_default_is_unchanged() -> None:
    x = prediction.make_inputs(4, 2, 2, seed=22)["train"]
    y = label(x, dataset="synthetic", target_bit=63)
    assert torch.equal(y, x[:, :1]) and y.data_ptr() != x.data_ptr()


BAD_DECLARATIONS: list[dict[str, Any]] = [
    {"label_expression": "xor"}, {"label_expression": True}, {"label_expression": None},
    {"label_bits": [0]}, {"label_bits": None}, {"label_bits": ""},
    {"label_bits": (False,)}, {"label_bits": (64,)},
    {"label_expression": "xor2", "label_bits": ()},
    {"label_expression": "xor2", "label_bits": (0, 0)},
    {"label_expression": "xor2", "label_bits": (0, 64)},
    {"label_expression": "xor2", "label_bits": (0, -1)},
    {"label_expression": "xor2", "label_bits": (0, True)},
    {"label_expression": "xor2", "label_bits": (0, 8.)},
    {"label_expression": "xor2", "label_bits": (1, 8)},
    {"label_expression": "xor2", "label_bits": (8, 0), "target_bit": 8},
    {"label_expression": "xor2", "label_bits": (0, 8), "target_bit": False},
    {"label_expression": "xor3", "label_bits": (0, 8)},
    {"label_expression": "xor3", "label_bits": (0, 16, 8)},
    {"label_expression": "xor3", "label_bits": (0, 8, 8)},
    {"label_expression": "xor-and", "label_bits": (0, 8)},
    {"label_expression": "xor-and", "label_bits": (0, 16, 8)},
    {"label_expression": "xor-and", "label_bits": (0, 0, 8)},
    {"label_expression": "xor-and", "label_bits": (0, 8, 8)},
]


@pytest.mark.parametrize("changes", BAD_DECLARATIONS)
@pytest.mark.parametrize("mode", ["cipher", "null"])
def test_strict_declarations_fail_before_null_or_cipher(
        changes: dict[str, Any], mode: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        ExperimentConfig(**(dict(name="invalid", dataset="simon", label_mode=mode) | changes))
    with pytest.raises((ValueError, TypeError)):
        label(torch.zeros(1, 64), label_mode=mode, **changes)


@pytest.mark.parametrize("changes", [
    {"dataset": "synthetic"}, {"dataset": "circuit-synthetic"},
    {"experiment": "innovation1", "dataset": "differential"},
    {"model_features": "support"}, {"prediction_mode": "public-residual"},
    {"prediction_mode": "public-coordinate"}, {"prediction_mode": "plaintext-xor"},
    {"prediction_mode": "plaintext-xor-no-final-low"},
    *[{"model_kind": kind} for kind in (
        "logic16", "circuit-aligned", "circuit-random", "simon-structure", "simon-unrolled",
        "simon-r2-table", "simon-r2-local", "simon-r2-vector", "simon-cnn",
        "simon-rotation-cnn", "simon-vector")],
])
def test_nonbit_only_prediction_simon_mlp_full_direct(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="non-bit expressions require"):
        ExperimentConfig(**(dict(name="bad-combo", dataset="simon", label_expression="xor2",
                                 label_bits=(0, 8)) | changes))


def test_data_nonbit_rejects_synthetic_even_before_label_shortcut() -> None:
    with pytest.raises(ValueError, match="non-bit expressions"):
        label(torch.zeros(1, 64), dataset="synthetic", label_expression="xor2", label_bits=(0, 8))


# Frozen before source edits: these include every old field, not current dataclass defaults.
LEGACY_DEFAULT: dict[str, Any] = {
    'name': 'legacy-default', 'dataset': 'synthetic', 'experiment': 'prediction',
    'train_rows': 64, 'dev_rows': 32, 'test_rows': 32, 'data_seed': 1, 'key_seed': 2,
    'init_seed': 3, 'rounds': 3, 'target_bit': 0, 'hidden_width': 32, 'model_kind': 'mlp',
    'model_features': 'full', 'prediction_mode': 'direct', 'label_mode': 'cipher',
    'null_seed': 191302, 'batch_size': 16, 'epochs': 2, 'learning_rate': 0.001,
    'device': 'cpu', 'max_seconds': 60, 'train_pool_rows': None, 'dev_every_epochs': 1,
}
LEGACY_CASES = [
    (LEGACY_DEFAULT, '07be17e9124079c8e8d554395a9fcab3b59c5cbacc988fccbbbabb723855844a'),
    (LEGACY_DEFAULT | dict(name='legacy-simon', dataset='simon', rounds=6, hidden_width=128,
                          train_rows=65536, dev_rows=8192, test_rows=65536,
                          train_pool_rows=65536, batch_size=256, epochs=32, device='cuda:0'),
     '7b482fee8796e8e8dc9e99815b06622830764a15863c5933d6476a8668344910'),
    (LEGACY_DEFAULT | dict(name='legacy-i1', experiment='innovation1', dataset='differential',
                          train_rows=2, dev_rows=2, test_rows=2, batch_size=2, epochs=1),
     'ce44c890a2d6f85d56228bb203eeabdcc43b18ecd3c4c8670e7c501eea52e7f2'),
]


@pytest.mark.parametrize("raw,sha", LEGACY_CASES)
def test_frozen_old_config_dict_and_sha(raw: dict[str, Any], sha: str) -> None:
    config = ExperimentConfig(**raw)
    assert config.to_dict() == raw and config.sha256() == sha
    explicit_json: dict[str, Any] = raw | {"label_expression": "bit", "label_bits": []}
    assert ExperimentConfig(**explicit_json).to_dict() == raw


@pytest.mark.parametrize("expression,bits", ARMS.items())
def test_json_normalization_roundtrip_and_checkpoint_identity(
        tmp_path: Path, expression: str, bits: tuple[int, ...]) -> None:
    json_options: dict[str, Any] = dict(name="identity", dataset="simon", rounds=6,
                                       label_expression=expression, label_bits=list(bits))
    config = ExperimentConfig(**json_options)
    assert type(config.label_bits) is tuple and config.label_bits == bits
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    assert load_config(config_path) == config
    model = build_predictor(config.hidden_width, rounds=config.rounds)
    path = tmp_path / "checkpoint.pt"
    checkpoints.save_checkpoint(path, model, config, 1, 2, 0, .5)
    payload = torch.load(path, weights_only=True)
    assert payload["schema"] == checkpoints.CHECKPOINT_SCHEMA
    loaded, resolved, _, _ = checkpoints.load_checkpoint(
        path, file_sha256(path), device=torch.device("cpu"))
    assert resolved == config and resolved.sha256() == config.sha256()
    assert loaded.checkpoint_config_sha256 == config.sha256()
    assert all(torch.equal(value, loaded.state_dict()[key])
               for key, value in model.state_dict().items())
    other = replace(config, label_expression="xor3" if expression == "xor2" else "xor2",
                    label_bits=(0, 8, 16) if expression == "xor2" else (0, 8))
    assert other.sha256() != config.sha256()
    if expression != "bit":
        different_bits = replace(config, label_bits=(0, 9) if expression == "xor2" else (0, 9, 16))
        assert different_bits.sha256() != config.sha256()
    payload["config"] = other.to_dict()  # raw config changed without its identity
    torch.save(payload, path)
    with pytest.raises(ValueError, match="configuration SHA"):
        checkpoints.load_checkpoint(path, file_sha256(path), device=torch.device("cpu"))


def test_legacy_sparse_raw_checkpoint_sha_not_reinterpreted(tmp_path: Path) -> None:
    raw = {"name": "older-than-defaults", "hidden_width": 32}
    sha = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    path = tmp_path / "old.pt"
    torch.save(dict(schema=checkpoints.CHECKPOINT_SCHEMA, config=raw, config_sha256=sha,
                    state_dict=dict(build_predictor(32).state_dict()), epoch=1, updates=2,
                    majority=0, probability=.5), path)
    model, config, _, _ = checkpoints.load_checkpoint(
        path, file_sha256(path), device=torch.device("cpu"))
    assert config.label_expression == "bit" and config.label_bits == ()
    assert model.checkpoint_config_sha256 == sha != config.sha256()
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        checkpoints.load_checkpoint(path, config.sha256(), device=torch.device("cpu"))


def test_default_labels_for_keeps_old_fake_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    def old_fake(x: Tensor, *, dataset: str, rounds: int, target_bit: int,
                 key_seed: int, label_mode: str, null_seed: int, split: str) -> Tensor:
        assert split == "train"
        return x[:, :1].clone()
    monkeypatch.setattr(prediction, "make_labels", old_fake)
    x = torch.zeros(2, 64)
    assert torch.equal(training.labels_for(ExperimentConfig(name="default"), x, "train"),
                       x[:, :1])


@pytest.mark.parametrize("expression,bits", ARMS.items())
def test_reference_panels_bit_only(monkeypatch: pytest.MonkeyPatch,
                                   expression: str, bits: tuple[int, ...]) -> None:
    calls: list[int] = []

    def reference(x: Tensor, rounds: int, target: int) -> Tensor:
        calls.append(target)
        return x[:, :1]

    monkeypatch.setattr(training, "public_reference", reference)
    config = ExperimentConfig(name="reference", dataset="simon", rounds=3,
                              label_expression=expression, label_bits=bits)
    panels = {"train": torch.zeros(2, 64)}
    refs = training.reference_panels(config, panels)
    assert bool(refs) == (expression == "bit")
    assert len(calls) == (1 if expression == "bit" else 0)


@pytest.mark.parametrize("expression,bits", ARMS.items())
@pytest.mark.parametrize("mode", ["cipher", "null"])
def test_fake_training_passes_semantics_and_loads_selected_before_test(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        expression: str, bits: tuple[int, ...], mode: str) -> None:
    events: list[str] = []
    original_load = checkpoints.load_checkpoint

    def load(*args: Any, **kwargs: Any) -> Any:
        result = original_load(*args, **kwargs)
        events.append("strict-selected-load")
        return result

    def fake_labels(x: Tensor, **kwargs: Any) -> Tensor:
        split = kwargs["split"]
        if expression != "bit":
            assert kwargs["label_expression"] == expression and kwargs["label_bits"] == bits
        else:
            assert "label_expression" not in kwargs and "label_bits" not in kwargs
        if split == "test":
            assert events[-1] == "strict-selected-load"
            assert (tmp_path / "run/selection.json").is_file()
        events.append(split)
        return x[:, :1].clone()  # arbitrary fake labels; no encryption

    monkeypatch.setattr(training, "load_checkpoint", load)
    monkeypatch.setattr(prediction, "make_labels", fake_labels)
    config = ExperimentConfig(name="fake-training", dataset="simon", rounds=6,
                              train_rows=8, dev_rows=4, test_rows=4, batch_size=4, epochs=1,
                              hidden_width=128, label_expression=expression, label_bits=bits,
                              label_mode=mode)
    result = training.train(config, tmp_path / "run")
    assert events == ["train", "dev", "strict-selected-load", "test"]
    assert result["updates"] == 2 and result["presentations"] == 8
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    for artifact in (manifest, result):
        assert artifact["label_expression"] == expression
        assert artifact["label_bits"] == list(bits)
        assert artifact["effective_label_bits"] == list(bits or (0,))
        assert artifact["label_bit_order"] == "64-bit ciphertext MSB=0, LSB=63"
        assert artifact["label_mode"] == mode
        assert artifact["label_source"] == (
            "independent uniform random bit; expression not evaluated"
            if mode == "null" else "ciphertext expression")
        if expression != "bit":
            assert "expression" in artifact["experiment_kind"]
            assert "single-bit" not in artifact["experiment_kind"]
    assert "public_reference_constant" not in result["metrics"]["test"]
    assert result["selected_epoch"] == 1


@pytest.mark.parametrize("dataset,effective_bits,bit_order,source", [
    ("synthetic", [0], "64-bit plaintext input MSB=0, LSB=63",
     "first plaintext input bit: x[0]; target_bit ignored"),
    ("circuit-synthetic", list(range(8)),
     "input column indices 0..63; first eight bits are relevant",
     "XOR of four disjoint adjacent input ANDs: "
     "(x[0] AND x[1]) XOR (x[2] AND x[3]) XOR "
     "(x[4] AND x[5]) XOR (x[6] AND x[7]); target_bit ignored"),
])
def test_synthetic_training_metadata_describes_actual_labels(
        tmp_path: Path, dataset: str, effective_bits: list[int],
        bit_order: str, source: str) -> None:
    config = ExperimentConfig(name="synthetic-semantics", dataset=dataset, target_bit=63,
                              train_rows=8, dev_rows=4, test_rows=4, batch_size=4, epochs=1,
                              device="cpu")
    original_config = config.to_dict()
    original_sha = config.sha256()
    output = tmp_path / "run"
    result = training.train(config, output)
    manifest = json.loads((output / "manifest.json").read_text())
    saved_result = json.loads((output / "results.json").read_text())
    assert saved_result == result
    for artifact in (manifest, saved_result):
        assert artifact["label_expression"] == "bit"  # unchanged config declaration
        assert artifact["label_bits"] == []
        assert artifact["effective_label_bits"] == effective_bits
        assert 63 not in artifact["effective_label_bits"]
        assert artifact["label_bit_order"] == bit_order
        assert "ciphertext" not in artifact["label_bit_order"]
        assert artifact["label_source"] == source
        assert artifact["label_mode"] == "cipher"
        assert artifact["config_sha256"] == original_sha
    assert result["status"] == "completed"
    assert result["updates"] == 2 and result["presentations"] == 8
    assert result["selected_epoch"] == 1
    assert set(result["metrics"]) == {"train", "dev", "test"}
    assert json.loads((output / "config.json").read_text()) == original_config
    assert config.to_dict() == original_config and config.sha256() == original_sha
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    assert checkpoint["config"] == original_config
    assert checkpoint["config_sha256"] == original_sha


def test_all_ten_configs_bind_parameters_semantics_and_budgets() -> None:
    formal_updates = 0
    for smoke in (False, True):
        folder = ROOT / ("configs/innovation2-cipher-expression" + ("-smoke" if smoke else ""))
        assert {p.stem for p in folder.glob("*.json")} == {*ARMS, "null"}
        for arm in (*ARMS, "null"):
            config = load_config(folder / f"{arm}.json")
            expression = "bit" if arm == "null" else arm
            assert config.label_expression == expression and config.label_bits == ARMS[expression]
            assert (config.dataset, config.experiment, config.rounds, config.target_bit,
                    config.model_kind, config.hidden_width, config.model_features,
                    config.prediction_mode) == ("simon", "prediction", 6, 0, "mlp", 128,
                                               "full", "direct")
            assert (config.data_seed, config.key_seed, config.init_seed, config.null_seed) == (
                195601, 190502, 17, 195602)
            assert config.dev_every_epochs == 1 and config.learning_rate == .001
            model = build_predictor(config.hidden_width, rounds=config.rounds)
            assert sum(p.numel() for p in model.parameters()) == 24961
            updates = config.epochs * ((config.train_rows + config.batch_size - 1)
                                       // config.batch_size)
            if smoke:
                assert (config.train_rows, config.dev_rows, config.test_rows,
                        config.train_pool_rows, config.batch_size, config.epochs) == (
                            8, 4, 4, 8, 4, 1)
                assert config.label_mode == "null" and config.device == "cpu"
                assert updates == 2 and config.epochs * config.train_rows == 8
                # Exercise each declared expression's real null label API; fixture forbids encrypt.
                x = prediction.make_inputs(8, 4, 4, seed=config.data_seed)["train"]
                assert training.labels_for(config, x, "train").shape == (8, 1)
            else:
                assert (config.train_rows, config.dev_rows, config.test_rows,
                        config.train_pool_rows, config.batch_size, config.epochs) == (
                            65536, 8192, 65536, 65536, 256, 32)
                assert config.label_mode == ("null" if arm == "null" else "cipher")
                assert config.device == "cuda:0" and config.max_seconds == 1800
                assert updates == 8192 and config.epochs * config.train_rows == 2097152
                formal_updates += updates
    assert formal_updates == 40960

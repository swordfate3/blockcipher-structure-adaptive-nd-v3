"""Task3 CLI/config/saved-report wiring; fake tiny protocols only, never science.

The CLI check child blocks scientific imports and verifies no files are created.
Runs use only the public demo oracle (identity/rounds=0) on a tiny synthetic
protocol; the renderer is fed saved summaries and must not touch data/models.
"""
from __future__ import annotations

import builtins
import json
import os
import shutil
import subprocess
import sys
import types
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import cast

import cipher_lab_plugins.fourier.cli as cli
import pytest
from cipher_lab_plugins.fourier import report
from cipher_lab_plugins.fourier.codec import (
    FourierPredictor,
    FourierTerm,
    decode_predictor,
    encode_predictor,
    validate_identity,
)
from cipher_lab_plugins.fourier.config import (
    FourierRecipe,
    estimate_fourier,
    load_experiment,
    parse_fourier,
)
from cipher_lab_plugins.fourier.source import source_identity
from cipher_lab_plugins.fourier.tests.fourier_fakes import science_guard

from cipher_lab.artifacts import RunStore, read_saved

__all__ = ["science_guard"]

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
FORMAL = {
    "schema": 1, "method": "bit-fourier-search-v1", "synthetic": False, "rounds": 6,
    "target_bit": 0, "input_bits": 64, "pairs_per_layer": 65536, "beam_width": 32,
    "tau": 0.1, "coef_size": 65536, "dev_size": 65536, "test_size": 65536,
    "search_seed": 19691111, "coef_seed": 29691111, "dev_seed": 39691111,
    "test_seed": 49691111,
}
EXAMPLE = {
    "schema": 1, "method": "bit-fourier-search-v1", "synthetic": True, "rounds": 0,
    "target_bit": 0, "input_bits": 6, "pairs_per_layer": 256, "beam_width": 8,
    "tau": 0.5, "coef_size": 256, "dev_size": 256, "test_size": 256,
    "search_seed": 19691112, "coef_seed": 29691112, "dev_seed": 39691112,
    "test_seed": 49691112,
}
POLICY = {
    "schema": 2, "wall_seconds": 60, "cpu_seconds": 60,
    "deadline_utc": "2099-01-01T00:00:00Z", "real_blocks": 4000, "null_words": 4000,
    "rss_stop_bytes": 1000000, "output_bytes": 4000000,
    "optimizer_updates": 0, "forward_rows": 0,
}


def settings(tmp_path: Path, value: Mapping[str, object]) -> list[str]:
    config, policy = tmp_path / "config.json", tmp_path / "policy.json"
    config.write_text(json.dumps(dict(value)))
    policy.write_text(json.dumps(POLICY))
    return ["--config", str(config), "--policy", str(policy)]


# ---- 3.1 config/static check -------------------------------------------------

def test_formal_fourier_recipe_and_static_estimate() -> None:
    loaded = load_experiment(PROJECT / "configs/fourier-r6.json")
    assert isinstance(loaded, FourierRecipe)
    assert loaded.to_dict() == FORMAL
    estimate = cli._estimate(loaded)
    assert estimate["planned_real_blocks"] == estimate["planned_null_words"] == 8585216
    assert estimate["planned_optimizer_updates"] == estimate["planned_forward_rows"] == 0
    assert estimate["search_real_blocks"] == 8388608
    assert estimate["max_sample_bucket_products"] == 536870912
    assert estimate["tau"] == 0.1 and estimate["beam_width"] == 32
    static = estimate_fourier(loaded)
    assert static["max_real_blocks"] == static["max_null_words"] == 8585216
    assert static["updates"] == static["nn_forward_rows"] == 0
    assert "worst" in str(estimate["basis"]).lower()


def test_example_fourier_is_independent_synthetic_not_simon() -> None:
    loaded = load_experiment(PROJECT / "configs/fourier-example.json")
    assert isinstance(loaded, FourierRecipe)
    assert loaded.to_dict() == EXAMPLE
    assert loaded.synthetic is True
    assert loaded.rounds == 0
    assert cli._estimate(loaded)["planned_real_blocks"] == 3840
    # Demo cannot impersonate six-round/64-bit formal bounds.
    with pytest.raises(ValueError):
        parse_fourier({**EXAMPLE, "rounds": 6})
    with pytest.raises(ValueError):
        parse_fourier({**FORMAL, "synthetic": True})
    # Old integral parsing keeps rejecting Fourier-only fields instead of widening.
    from cipher_lab.config import _parse_recipe  # strict unknown-field rejection

    old = json.loads((PROJECT / "configs/integral-example.json").read_text())
    with pytest.raises(ValueError):
        parse_fourier(old)
    from cipher_lab.config import load_experiment as load_base
    assert load_base(PROJECT / "configs/integral-example.json") is not None
    with pytest.raises(ValueError):
        _parse_recipe({**old, "pairs_per_layer": 65536})


def test_fourier_check_zero_science_and_no_writes(tmp_path: Path) -> None:
    args = settings(tmp_path, FORMAL)
    script = """
import importlib.abc, sys
BLOCKED = ('numpy', 'torch', 'cipher_lab.data', 'cipher_lab_plugins.fourier.search',
           'cipher_lab_plugins.fourier.method', 'cipher_lab_plugins.fourier.data',
           'cipher_lab.ciphers', 'cipher_lab.execution', 'cipher_lab.artifacts')
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for root in BLOCKED:
            if fullname == root or fullname.startswith(root + '.'):
                raise AssertionError('scientific import during check: ' + fullname)
        return None
sys.meta_path.insert(0, Block())
from cipher_lab_plugins.fourier.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    done = subprocess.run([sys.executable, "-c", script, "check", *args],
                          cwd=tmp_path, capture_output=True, text=True, check=False,
                          env=dict(os.environ, PYTHONPATH=os.pathsep.join((
                              str(PROJECT / "src"), str(PROJECT / "plugins"))),
                                   PYTHONDONTWRITEBYTECODE="1"))
    assert done.returncode == 0, done.stderr
    output = json.loads(done.stdout)
    assert output["estimate"]["planned_real_blocks"] == 8585216
    assert output["planned_calls_fit_policy"]["real_blocks"] is False
    assert output["planned_calls_fit_policy"]["null_words"] is False
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_fourier_invalid_combinations_rejected_before_claim(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    assert cli.main(["check", *settings(tmp_path, FORMAL), "--public-plan", str(plan)]) == 2
    assert "invalid method/plan combination" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**EXAMPLE, "rounds": 6}))
    assert cli.main(["check", "--config", str(bad), "--policy",
                     str(tmp_path / "policy.json")]) == 2
    capsys.readouterr()
    for backend in ("simon", "ciphers"):
        root = tmp_path / f"root-{backend}"
        assert cli.main(["run", *settings(tmp_path, EXAMPLE), "--root", str(root),
                         "--id", "toy", "--backend", backend]) == 2
        assert "invalid method/backend combination" in capsys.readouterr().err
        assert not root.exists()
    root = tmp_path / "root-formal-demo"
    assert cli.main(["run", *settings(tmp_path, FORMAL), "--root", str(root),
                     "--id", "formal", "--backend", "demo"]) == 2
    capsys.readouterr()
    assert not root.exists()


# ---- 3.2 saved-only renderer -------------------------------------------------

def _layer(depth: int) -> dict[str, object]:
    return {"depth": depth, "tested": 64, "survivors": 3, "kept": 2, "truncated": 1,
            "sample_bucket_products": 8192, "score_min": -0.25, "score_max": 0.91,
            "pair_collisions": 0, "count": 65536, "masks": [1 << 63],
            "candidates": [{"mask": 1 << 63, "signed_sum": 59638, "count": 65536}]}


def _summary(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "method": "bit-fourier-search-v1", "backend_identity": "simon",
        "synthetic": False, "scientific_validation": False,
        "outcome": "POSITIVE_EXPLORATORY", "search_status": "COMPLETE",
        "coef_status": "COMPLETE", "dev_status": "COMPLETE", "test_status": "COMPLETE",
        "null_status": "NO_CANDIDATE", "source_sha256": "a" * 64,
        "tau": 0.1, "beam_width": 32, "sampling": "with_replacement",
        "split_exact_intersection": "UNKNOWN",
        "limitations": ["bounded_beam_not_complete_GL", "single_key_only"],
        "real": {
            "layers": [_layer(1), _layer(2)], "coef_status": "COMPLETE",
            "model_ref": "model/spectral.pt", "model_sha256": "b" * 64,
            "baseline": 0, "intercept": 0.0, "count": 65536,
            "terms": [{"mask": 1 << 63, "signed_sum": 65000, "coefficient": 0.991}],
            "dev": {"count": 65536, "correct": 40000, "accuracy": 0.6103, "baseline": 0,
                    "baseline_correct": 33000, "baseline_accuracy": 0.5035,
                    "delta": 0.1068},
            "test": {"count": 65536, "correct": 40100, "accuracy": 0.6119, "baseline": 0,
                     "baseline_correct": 33000, "baseline_accuracy": 0.5035,
                     "delta": 0.1084, "d_sum": 7100,
                     "d_histogram": {"-1": 6200, "0": 45936, "1": 13400},
                     "lower": 0.1084 - 0.0119, "alpha": 0.01,
                     "positive_exploratory": True},
        },
        "null": {"layers": [], "coef_status": "NO_CANDIDATE"},
    }
    base.update(overrides)
    return base


def _counts() -> dict[str, object]:
    return {
        "attempted_real_blocks": 8585216, "completed_real_blocks": 8585216,
        "attempted_null_words": 1310720, "completed_null_words": 1310720,
        "attempted_optimizer_updates": 0, "completed_optimizer_updates": 0,
        "attempted_forward_rows": 0, "completed_forward_rows": 0,
        "real_sample_bucket_products_attempted": 536870912,
        "real_sample_bucket_products_completed": 536870912,
        "null_sample_bucket_products_attempted": 1024,
        "null_sample_bucket_products_completed": 1024,
        "real_coefficient_rows_attempted": 65568,
        "real_coefficient_rows_completed": 65568,
        "null_coefficient_rows_attempted": 0, "null_coefficient_rows_completed": 0,
        "real_scoring_rows_attempted": 131072, "real_scoring_rows_completed": 131072,
        "null_scoring_rows_attempted": 0, "null_scoring_rows_completed": 0,
        "wall_elapsed_seconds": 12.5, "cpu_elapsed_seconds": 12.0, "rss_bytes": 1024,
    }


def saved_run(tmp_path: Path, status: str = "SUCCEEDED",
              summary: Mapping[str, object] | None = None,
              root: Path | None = None) -> dict[str, object]:
    root = root or tmp_path
    store = RunStore.claim(root, "saved", output_bytes=4_000_000, needs_models=True,
                           source_supplier=source_identity)
    store.write_snapshot({"recipe": dict(FORMAL), "summary": dict(summary or _summary()),
                          "counts": _counts()})
    store.finalize({"status": status, "summary": dict(summary or _summary()),
                    "counts": _counts()})
    return read_saved(root, "saved")


def test_renderer_shows_saved_layers_terms_gates_and_cost(tmp_path: Path) -> None:
    saved = saved_run(tmp_path)
    files = report.render_spectral_report(saved)
    for name in ("index.html", "report.md", "source.json"):
        assert set(files) == {"index.html", "report.md", "source.json"}
    text = files["report.md"].decode()
    for expected in ("tau=0.1", "beam_width=32", "严格大于 tau²/2", "第 1 层", "第 2 层",
                     "truncated=1", "beam 截断", "mask=0x8000000000000000",
                     "coefficient=0.991", "signed_sum=65000", "baseline=0",
                     "DEV：", "TEST：", "0.6103", "0.6119", "lower=", "alpha=0.01",
                     "positive_exploratory=True", "单 key", "real_blocks",
                     "sample_bucket_products", "coefficient_rows", "scoring_rows",
                     "0.1068", "8585216"):
        assert expected in text, expected
    html = files["index.html"].decode()
    assert html.count("<svg") == 2
    assert "<script>" not in text and "<script>" not in html
    assert json.loads(files["source.json"])["run_sha256"] == saved["run_sha256"]


@pytest.mark.parametrize("case", ["empty_beam", "coef_empty", "dev_gate"])
def test_renderer_never_fills_unvisited_splits(tmp_path: Path, case: str) -> None:
    if case == "empty_beam":
        summary = _summary(outcome="NO_CANDIDATE", search_status="NO_CANDIDATE",
                           coef_status="NOT_STARTED_GATE", dev_status="NOT_STARTED_GATE",
                           test_status="NOT_STARTED_GATE", null_status="NOT_COMPLETED_REAL_GATE",
                           real={"layers": []}, null={"layers": []})
    elif case == "coef_empty":
        summary = _summary(outcome="NO_CANDIDATE", coef_status="NO_CANDIDATE",
                           dev_status="NOT_STARTED_GATE", test_status="NOT_STARTED_GATE",
                           real={"layers": [_layer(1)], "coef_status": "NO_CANDIDATE"},
                           null={"layers": []})
    else:
        base = _summary(outcome="NO_DEV_GATE", dev_status="COMPLETE",
                        test_status="NOT_STARTED_GATE")
        real = cast(dict[str, object], deepcopy(base["real"]))
        real.pop("test")
        real["dev"] = {"count": 65536, "correct": 32768, "accuracy": 0.5, "baseline": 0,
                       "baseline_correct": 32768, "baseline_accuracy": 0.5, "delta": 0.0}
        base["real"] = real
        summary = base
    saved = saved_run(tmp_path, summary=summary)
    files = report.render_spectral_report(saved)
    for name in ("index.html", "report.md"):
        text = files[name].decode()
        assert "未访问" in text and "不补 0" in text
        if case != "dev_gate":
            assert "0.500000" not in text
            assert "50%" not in text
    text = files["report.md"].decode()
    if case == "dev_gate":
        assert "0.5" in text and "NOT_STARTED_GATE" in text
    else:
        assert "无已保存模型" in text or "NO_CANDIDATE" in text


def test_renderer_stopped_unknown_and_damaged_source_are_not_success(tmp_path: Path) -> None:
    stopped = _summary(outcome="STOPPED", reason="CPU_LIMIT", interrupted_phase="search",
                       search_status="PARTIAL", coef_status="NOT_STARTED_GATE",
                       dev_status="NOT_STARTED_GATE", test_status="NOT_STARTED_GATE",
                       scientific_validation=False)
    text = report.render_spectral_report(
        saved_run(tmp_path, status="STOPPED", summary=stopped))["report.md"].decode()
    assert "STOPPED" in text and "CPU_LIMIT" in text and "PARTIAL" in text
    assert "partial" in text.lower() or "未知" in text
    running = tmp_path / "running"
    store = RunStore.claim(running, "open", output_bytes=4_000_000, needs_models=True,
                           source_supplier=source_identity)
    store.write_snapshot({"recipe": dict(FORMAL), "summary": _summary(), "counts": _counts()})
    unknown = report.render_spectral_report(read_saved(running, "open"))["report.md"].decode()
    assert "UNKNOWN" in unknown and "RUNNING" in unknown
    damaged = _summary(source_sha256="NOT-A-HASH")
    text = report.render_spectral_report(
        saved_run(tmp_path, summary=damaged, root=tmp_path / "damaged"))["report.md"].decode()
    assert "不推断、不补造来源身份" in text
    assert "NOT-A-HASH" not in text


def test_renderer_demo_is_not_simon(tmp_path: Path) -> None:
    demo = _summary(backend_identity="demo", synthetic=True, scientific_validation=False)
    files = report.render_spectral_report(saved_run(tmp_path, summary=demo))
    for name in ("index.html", "report.md"):
        text = files[name].decode()
        assert "合成工程演示" in text and "不是 SIMON" in text
        assert "科学验证" in text


def test_signed_negative_saved_fields_are_preserved(tmp_path: Path) -> None:
    summary = _summary()
    real = cast(dict[str, object], deepcopy(summary["real"]))
    real["terms"] = [{"mask": 1 << 63, "signed_sum": -65000, "coefficient": -0.991}]
    test = cast(dict[str, object], deepcopy(real["test"]))
    test.update({"d_sum": -7100, "delta": -0.1084, "lower": -0.1203,
                 "positive_exploratory": False})
    real["test"] = test
    summary["real"] = real
    files = report.render_spectral_report(saved_run(tmp_path, summary=summary))
    for name in ("index.html", "report.md"):
        text = files[name].decode()
        assert "signed_sum=-65000" in text
        assert "d_sum=-7100" in text
        assert "coefficient=-0.991" in text
        assert "signed_sum=未保存" not in text and "d_sum=未保存" not in text


def test_dynamic_saved_fields_are_escaped_in_both_renderers(tmp_path: Path) -> None:
    evil = '<img src=x onerror=alert(1)>'
    summary = _summary(backend_identity=evil, reason=evil, method=evil,
                       limitations=[evil])
    files = report.render_spectral_report(saved_run(tmp_path, summary=summary))
    for name in ("index.html", "report.md"):
        text = files[name].decode()
        assert "<img" not in text
        assert "&lt;img" in text


@pytest.mark.parametrize("saved_value,expected", [(True, "True"), (False, "False"),
                                                  (None, "未保存（不补造）")])
def test_scientific_validation_is_reported_not_re_adjudicated(
    tmp_path: Path, saved_value: bool | None, expected: str,
) -> None:
    summary = _summary()
    if saved_value is None:
        summary.pop("scientific_validation", None)
    else:
        summary["scientific_validation"] = saved_value
    text = report.render_spectral_report(saved_run(tmp_path, summary=summary))["report.md"].decode()
    assert f"scientific_validation={expected}" in text
    assert "不重新裁定" in text
    if saved_value is True:
        assert "scientific_validation=false" not in text


def test_renderer_does_not_import_science_read_files_or_mutate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = saved_run(tmp_path)
    before = deepcopy(saved)
    original = builtins.__import__

    def no_science(name: str, globals: Mapping[str, object] | None = None,
                   locals: Mapping[str, object] | None = None,
                   fromlist: tuple[str, ...] = (), level: int = 0) -> object:
        if name.startswith("cipher_lab"):
            pytest.fail("renderer imported science")
        return original(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", no_science)
        patch.setattr(Path, "read_bytes", lambda p: pytest.fail("report I/O"))
        patch.setattr(Path, "open", lambda *a, **k: pytest.fail("report I/O"))
        files = report.render_spectral_report(saved)
    assert saved == before
    assert files["index.html"].startswith(b"<!doctype html>")


def test_saved_dispatch_uses_method_not_recipe_shape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    summary = _summary()
    sav = saved_run(tmp_path, summary=summary)
    store = RunStore.claim(tmp_path, "other", output_bytes=4_000_000, needs_models=True,
                           source_supplier=source_identity)
    store.finalize({"status": "FAILED", "summary": {"method": "bit-fourier-search-v1",
                                                    "backend_identity": "demo",
                                                    "synthetic": True},
                    "recipe": {"method": "unknown"}})
    assert cli._saved_report(store, tmp_path, "other", "simon") == 4
    output = json.loads(capsys.readouterr().out)
    assert output["backend_identity"] == "demo"
    html = (tmp_path / "result/other/index.html").read_text()
    assert "Fourier" in html or "频谱" in html
    assert sav["status"] == "SUCCEEDED"
    empty = RunStore.claim(tmp_path, "unknown", output_bytes=4_000_000,
                           source_supplier=source_identity)
    empty.finalize({"status": "FAILED", "summary": {"backend_identity": "UNVERIFIED"}})
    assert cli._saved_report(empty, tmp_path, "unknown", "demo") == 6
    assert json.loads(capsys.readouterr().out)["reason"] == "REPORT_METHOD_UNKNOWN"


# ---- 3.3/3.4 integration and clean-source fake loop --------------------------

def test_demo_run_known_function_and_saved_report(tmp_path: Path,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "out"
    code = cli.main(["run", *settings(tmp_path, EXAMPLE), "--root", str(root),
                     "--id", "toy", "--backend", "demo"])
    output = json.loads(capsys.readouterr().out)
    assert code == 0 and output["report_status"] == "SAVED"
    saved = read_saved(root, "toy")
    summary = cast(dict[str, object], saved["summary"])
    assert summary["method"] == "bit-fourier-search-v1"
    assert summary["outcome"] == "POSITIVE_EXPLORATORY"
    assert summary["scientific_validation"] is False
    real = cast(dict[str, object], summary["real"])
    assert cast(dict[str, object], real["dev"])["accuracy"] == 1.0
    assert cast(dict[str, object], real["test"])["accuracy"] == 1.0
    terms = cast(list[dict[str, object]], real["terms"])
    assert [term["mask"] for term in terms] == [1 << 63]
    counts = cast(dict[str, object], saved["counts"])
    assert counts["attempted_optimizer_updates"] == counts["completed_forward_rows"] == 0
    assert counts["completed_real_blocks"] == 3840
    model = root / "models/toy/spectral.pt"
    recipe = parse_fourier(dict(EXAMPLE))
    expected = validate_identity({
        "source_sha256": summary["source_sha256"], "backend": "demo", "synthetic": True,
        "task": {"cipher": "synthetic", "rounds": 0, "target_bit": 0, "input_bits": 6},
        "label_mode": "target_bit"}, recipe)
    decoded = decode_predictor(model.read_bytes(), expected_identity=expected)
    assert decoded.recipe == recipe
    html = (root / "result/toy/index.html").read_text()
    assert "合成工程演示" in html and "1" in html
    assert (root / "result/toy/report.md").exists()
    assert (root / "result/toy/source.json").exists()


def test_formal_simon_assembles_spectral_not_integral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from cipher_lab_plugins.fourier.data import SpectralData

    fake_cipher = types.ModuleType("cipher_lab.ciphers.simon")

    def fake_encrypt(points: object, key: tuple[int, ...], rounds: int) -> object:
        raise AssertionError("fake cipher must not be called by this assembly test")

    fake_cipher.encrypt_batch = fake_encrypt  # type: ignore[attr-defined]
    package = types.ModuleType("cipher_lab.ciphers")
    package.simon = fake_cipher  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cipher_lab.ciphers", package)
    monkeypatch.setitem(sys.modules, "cipher_lab.ciphers.simon", fake_cipher)
    seen: list[tuple[SpectralData, str]] = []

    def terminal(recipe: object, data: object, meter: object, store: RunStore, *,
                 backend_identity: str) -> None:
        assert isinstance(data, SpectralData)
        seen.append((data, backend_identity))
        store.finalize({"status": "SUCCEEDED", "summary": {
            "method": "bit-fourier-search-v1", "backend_identity": backend_identity,
            "synthetic": False, "outcome": "NOT_STARTED_GATE",
            "search_status": "NOT_STARTED", "coef_status": "NOT_STARTED_GATE",
            "dev_status": "NOT_STARTED_GATE", "test_status": "NOT_STARTED_GATE",
            "null_status": "NOT_STARTED", "real": {"layers": []},
            "null": {"layers": []}}})
    import cipher_lab_plugins.fourier.method as method

    monkeypatch.setattr(method, "run_fourier", terminal)
    root = tmp_path / "formal"
    code = cli.main(["run", *settings(tmp_path, FORMAL), "--root", str(root),
                     "--id", "formal", "--backend", "simon"])
    assert code == 0, capsys.readouterr()
    assert seen and seen[0][1] == "simon"
    assert (root / "models").exists()
    assert not (root / "data/formal/dataset").exists()


def test_evaluate_loader_still_rejects_fourier_json_model(tmp_path: Path) -> None:
    recipe = parse_fourier(dict(EXAMPLE))
    identity = validate_identity({
        "source_sha256": "c" * 64, "backend": "demo", "synthetic": True,
        "task": {"cipher": "synthetic", "rounds": 0, "target_bit": 0, "input_bits": 6},
        "label_mode": "target_bit"}, recipe)
    model = FourierPredictor(recipe=recipe, identity=identity, count=256,
                             intercept_sum=0, intercept=0.0, baseline=0,
                             terms=(FourierTerm(1 << 63, 256, 1.0),), coefficient_rows=512)
    content = encode_predictor(model)
    path = tmp_path / "spectral.pt"
    path.write_bytes(content)
    script = r"""
import hashlib, importlib.abc, os, random, secrets, sys, types, uuid
BLOCKED = ('torch', 'cipher_lab.ciphers', 'cipher_lab.training')
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for name in BLOCKED:
            if fullname == name or fullname.startswith(name + '.'):
                raise AssertionError('scientific import: ' + fullname)
        return None
sys.meta_path.insert(0, Block())
calls = []
def forbidden(*a, **k):
    calls.append(1)
    raise AssertionError('forbidden scientific call')
counter = iter(range(1, 100000))
uuid.uuid4 = lambda: uuid.UUID(int=next(counter))
os.urandom = lambda size: b'\0' * size
secrets.randbits = lambda bits: 0
import numpy.random  # deterministic fake entropy for lazy library initialization
# Inert dependencies only: actual core preflight/load_checkpoint executes, but
# torch and every scientific dependency are module substitutes, never imported.
class Inert:
    __init__ = forbidden
    __call__ = forbidden
    backward = forbidden
    step = forbidden
for module_name, names in {
    'torch': ('Tensor',),
    'cipher_lab.data.bundles': ('PredictionTask', 'TaskDeclaration', 'FeatureProbeTask',
        'FrontierTask', 'ResidualPredictionTask'),
    'cipher_lab.evaluation.metrics': ('Model', 'PublicModelSpec', 'TrainBaseline',
        'validate_baseline', 'validate_model', 'EvaluationPanels', 'evaluate_panel',
        'evaluate_models', 'binary_metrics', 'evaluate_batches'),
    'cipher_lab.models': ('BitMLP', 'SharedM', 'FeatureBitMLP', 'FrontierBitMLP'),
}.items():
    module = types.ModuleType(module_name)
    for name in names:
        setattr(module, name, Inert)
    sys.modules[module_name] = module
torch = sys.modules['torch']  # no torch import, only the inert substitute above
assert torch is sys.modules['torch']
torch.nn = types.SimpleNamespace(Module=Inert, Linear=Inert)
torch.optim = types.SimpleNamespace(Adam=Inert, AdamW=Inert, SGD=Inert)
torch.load = forbidden
from cipher_lab import artifacts
artifacts.uuid4 = lambda: uuid.UUID(int=next(counter))
for name in ('token_bytes', 'randbits', 'randbelow', 'choice'):
    setattr(secrets, name, forbidden)
random.SystemRandom = forbidden
random._urandom = forbidden
os.urandom = forbidden
import cipher_lab.execution
cipher_lab.execution.production_rss_bytes = forbidden
torch.load = forbidden
torch.nn.Module.__call__ = forbidden
torch.nn.Linear.__init__ = forbidden
torch.Tensor.backward = forbidden
for optimizer in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
    optimizer.__init__ = forbidden
    optimizer.step = forbidden
from cipher_lab.evaluation.checkpoints import load_checkpoint
content = open(sys.argv[1], 'rb').read()
try:
    load_checkpoint(content, hashlib.sha256(content).hexdigest())
except ValueError:
    pass
else:
    raise SystemExit('old loader accepted a spectral JSON model')
assert not calls, 'checkpoint rejection path performed a forbidden call'
print('REJECTED_WITH_GUARDS')
"""
    done = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True,
                          text=True, check=False,
                          env=dict(os.environ, PYTHONPATH=os.pathsep.join((
                              str(PROJECT / "src"), str(PROJECT / "plugins"))),
                                   PYTHONDONTWRITEBYTECODE="1"))
    assert done.returncode == 0, done.stderr + done.stdout
    assert done.stdout.strip() == "REJECTED_WITH_GUARDS"


def test_clean_source_fake_closed_loop(tmp_path: Path) -> None:
    copy = tmp_path / "clean source"
    copy.mkdir()
    shutil.copytree(PROJECT / "src", copy / "src")
    shutil.copytree(PROJECT / "configs", copy / "configs")
    plugin_package = copy / "plugins/cipher_lab_plugins"
    plugin_package.mkdir(parents=True)
    shutil.copy2(PROJECT / "plugins/cipher_lab_plugins/__init__.py", plugin_package / "__init__.py")
    shutil.copytree(PROJECT / "plugins/cipher_lab_plugins/fourier", plugin_package / "fourier",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(PROJECT / "pyproject.toml", copy / "pyproject.toml")
    shutil.copy2(PROJECT / ".gitignore", copy / ".gitignore")
    assert not (copy / "data").exists() and not (copy / "models").exists()
    (copy / "policy.json").write_text(json.dumps(POLICY))
    root = tmp_path / "out"
    script = r"""
import hashlib, importlib.abc, json, os, random, secrets, sys, uuid
from pathlib import Path

BLOCKED = ('torch', 'psutil', 'cipher_lab.ciphers', 'cipher_lab.models',
           'cipher_lab.training')
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for name in BLOCKED:
            if fullname == name or fullname.startswith(name + '.'):
                raise AssertionError('scientific import: ' + fullname)
        return None
sys.meta_path.insert(0, Block())

def forbidden(*a, **k):
    raise AssertionError('real scientific call forbidden')

counter = iter(range(1, 100000))
uuid.uuid4 = lambda: uuid.UUID(int=next(counter))
os.urandom = lambda size: b'\0' * size
secrets.randbits = lambda bits: 0
import numpy.random  # deterministic fake entropy for lazy library initialization
from cipher_lab import artifacts
artifacts.uuid4 = lambda: uuid.UUID(int=next(counter))
for name in ('token_bytes', 'randbits', 'randbelow', 'choice'):
    setattr(secrets, name, forbidden)
random.SystemRandom = forbidden
random._urandom = forbidden
os.urandom = forbidden
import cipher_lab.execution
cipher_lab.execution.production_rss_bytes = forbidden
from cipher_lab_plugins.fourier.cli import main

copy = Path.cwd()
check = main(['check', '--config', 'configs/fourier-example.json',
              '--policy', 'policy.json'])
assert check == 0, check
run = main(['run', '--config', 'configs/fourier-example.json', '--policy', 'policy.json',
            '--root', sys.argv[1], '--id', 'toy', '--backend', 'demo'])
assert run == 0, run
from cipher_lab.artifacts import read_saved
saved = read_saved(Path(sys.argv[1]), 'toy')
summary = saved['summary']
assert summary['outcome'] == 'POSITIVE_EXPLORATORY'
assert summary['scientific_validation'] is False
real = summary['real']
assert real['dev']['accuracy'] == 1.0 and real['test']['accuracy'] == 1.0
expected_source = {}
package = copy / 'src/cipher_lab'
for path in sorted(package.rglob('*.py')):
    expected_source['cipher_lab/' + path.relative_to(package).as_posix()] = \
        hashlib.sha256(path.read_bytes()).hexdigest()
plugin_package = copy / 'plugins/cipher_lab_plugins'
expected_source['cipher_lab_plugins/__init__.py'] = hashlib.sha256(
    (plugin_package / '__init__.py').read_bytes()).hexdigest()
fourier_package = plugin_package / 'fourier'
for path in sorted(fourier_package.rglob('*.py')):
    if path.relative_to(fourier_package).parts[0] == 'tests':
        continue
    expected_source['cipher_lab_plugins/' + path.relative_to(plugin_package).as_posix()] = \
        hashlib.sha256(path.read_bytes()).hexdigest()
expected_source_sha = hashlib.sha256(json.dumps(
    expected_source, sort_keys=True, separators=(',', ':'), ensure_ascii=True
).encode()).hexdigest()
assert summary['source_sha256'] == expected_source_sha
assert dict(sorted(saved['source'].items())) == dict(sorted(expected_source.items()))
from cipher_lab_plugins.fourier.codec import decode_predictor
from cipher_lab_plugins.fourier.config import parse_fourier
recipe = parse_fourier(json.loads((copy / 'configs/fourier-example.json').read_text()))
content = (Path(sys.argv[1]) / 'models/toy/spectral.pt').read_bytes()
encoded = json.loads(content)
expected_identity = {
    'source_sha256': expected_source_sha, 'backend': 'demo', 'synthetic': True,
    'task': {'cipher': 'synthetic', 'rounds': 0, 'target_bit': 0, 'input_bits': 6},
    'label_mode': 'target_bit',
}
assert encoded['identity'] == expected_identity
model = decode_predictor(content, expected_identity=expected_identity)
assert model.recipe == recipe and model.terms[0].mask == 1 << 63
for changed in ({**expected_identity, 'source_sha256': '0' * 64},
                {**expected_identity, 'backend': 'simon'},
                {**expected_identity, 'synthetic': False},
                {**expected_identity, 'label_mode': 'pseudorandom_function'}):
    try:
        decode_predictor(content, expected_identity=changed)
    except ValueError:
        pass
    else:
        raise AssertionError('tampered expected identity accepted')
counts = saved['counts']
assert counts['attempted_optimizer_updates'] == counts['completed_forward_rows'] == 0
print(json.dumps({'check': check, 'run': run, 'status': saved['status'],
                  'outcome': summary['outcome'],
                  'real_blocks': counts['completed_real_blocks'],
                  'null_words': counts['completed_null_words'],
                  'source_files': len(expected_source),
                  'source_sha256': summary['source_sha256'],
                  'expected_source_sha': expected_source_sha,
                  'run_sha256': saved['run_sha256'],
                  'model_identity_bound': True}, sort_keys=True))
"""
    done = subprocess.run([sys.executable, "-c", script, str(root)], cwd=copy,
                          capture_output=True, text=True, check=False,
                          env=dict(os.environ, PYTHONPATH=os.pathsep.join((
                              str(copy / "src"), str(copy / "plugins"))),
                                   PYTHONDONTWRITEBYTECODE="1"))
    assert done.returncode == 0, done.stderr + done.stdout
    evidence = json.loads(done.stdout.strip().splitlines()[-1])
    assert evidence["check"] == evidence["run"] == 0
    assert evidence["status"] == "SUCCEEDED"
    assert evidence["outcome"] == "POSITIVE_EXPLORATORY"
    assert evidence["real_blocks"] == 3840
    assert evidence["null_words"] == 3072
    assert evidence["source_files"] > 40 and len(evidence["source_sha256"]) == 64
    assert evidence["source_sha256"] == evidence["expected_source_sha"]
    assert evidence["model_identity_bound"] is True
    assert not (copy / "data").exists() and not (copy / "models").exists()

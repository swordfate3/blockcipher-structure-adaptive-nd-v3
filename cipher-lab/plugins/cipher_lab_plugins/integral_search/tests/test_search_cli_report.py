"""Search CLI and saved-only rendering; no scientific execution permission."""
from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

import cipher_lab.execution as execution
import cipher_lab_plugins.integral_search.cli as cli
import cipher_lab_plugins.integral_search.method as method
import cipher_lab_plugins.integral_search.report as report
from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.config import StageSizes
from cipher_lab_plugins.integral_search.config import SearchRecipe, load_experiment

PROJECT = Path(__file__).resolve().parents[4]


def recipe() -> SearchRecipe:
    return SearchRecipe(1, "integral-search-v1", (11, 12, 13, 14),
                        (tuple(range(0, 32, 2)), tuple(range(32, 64, 2))),
                        10, 16, StageSizes(2112, 320, 256), 65536)


def settings(tmp_path: Path) -> list[str]:
    config, policy = tmp_path / "config.json", tmp_path / "policy.json"
    small = SearchRecipe(1, "integral-search-v1", (11, 12),
                         ((0, 2, 4), (32, 34, 36)), 2, 3, StageSizes(1, 1, 1), 4)
    config.write_text(json.dumps(small.to_dict()))
    policy.write_text(json.dumps({
        "schema": 2, "wall_seconds": 120, "cpu_seconds": 120,
        "deadline_utc": "2099-01-01T00:00:00Z", "real_blocks": 100,
        "null_words": 100, "rss_stop_bytes": 1000000, "output_bytes": 20000000,
        "optimizer_updates": 0, "forward_rows": 0,
    }))
    return ["--config", str(config), "--policy", str(policy)]


@pytest.fixture(autouse=True)
def no_production(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*a: object, **k: object) -> None:
        pytest.fail("production cipher/RNG/RSS called")
    monkeypatch.setattr("cipher_lab.ciphers.simon.encrypt_many", forbidden)
    monkeypatch.setattr("cipher_lab.ciphers.simon.encrypt_batch", forbidden)
    monkeypatch.setattr("secrets.token_bytes", forbidden)
    monkeypatch.setattr(execution, "production_rss_bytes", forbidden)


def test_search_static_worst_case_full_plan_no_early_stop_guarantee() -> None:
    estimate = cli._estimate(recipe())
    assert estimate["planned_real_blocks"] == estimate["planned_null_words"] == 726663168
    assert estimate["planned_real_groups"] == estimate["planned_null_groups"] == 32256
    assert estimate["planned_optimizer_updates"] == estimate["planned_forward_rows"] == 0
    cells = cast(list[dict[str, object]], estimate["cells"])
    assert len(cells) == 12
    assert [cell["tier"] for cell in cells] == ["discovery"] * 8 + ["expanded"] * 4
    assert [cell["points_per_group"] for cell in cells] == [1024] * 8 + [65536] * 4
    assert all(cell["conditional"] is True for cell in cells[8:])
    assert "worst" in str(estimate["basis"]).lower()
    assert "guarantee" in str(estimate["basis"]).lower()


def test_fixed_search_recipe_is_complete_and_exact() -> None:
    assert load_experiment(PROJECT / "configs/integral-search.json") == recipe()


def test_search_check_is_zero_scientific_imports_and_writes(tmp_path: Path) -> None:
    arguments = settings(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(recipe().to_dict()))
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('numpy', 'torch', 'cipher_lab.integral', 'cipher_lab.data',
                                'cipher_lab.ciphers', 'cipher_lab.execution',
                                'cipher_lab.artifacts')):
            raise AssertionError('scientific import during check')
sys.meta_path.insert(0, Block())
from cipher_lab_plugins.integral_search.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    done = subprocess.run([sys.executable, "-c", script, "check", *arguments],
                          cwd=tmp_path, capture_output=True, text=True, check=False,
                          env=dict(os.environ, PYTHONPATH=os.pathsep.join(
                              (str(PROJECT / "src"), str(PROJECT / "plugins"))),
                                   PYTHONDONTWRITEBYTECODE="1"))
    assert done.returncode == 0, done.stderr
    output = json.loads(done.stdout)
    assert output["estimate"]["planned_real_blocks"] == 726663168
    assert output["planned_calls_fit_policy"]["real_blocks"] is False
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("backend", ["demo", "simon"])
def test_search_uses_same_integral_assembly_without_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], backend: str,
) -> None:
    from cipher_lab.demo import DemoBytes, demo_rss_bytes, fake_encrypt_batch

    if backend == "simon":
        source = DemoBytes()
        monkeypatch.setattr("secrets.token_bytes", lambda n: source(n))
        monkeypatch.setattr("cipher_lab.ciphers.simon.encrypt_batch", fake_encrypt_batch)
        monkeypatch.setattr(execution, "production_rss_bytes", demo_rss_bytes)
    seen: list[object] = []
    def terminal(recipe: object, collector: object, meter: object, store: RunStore,
                 *, backend_identity: str) -> dict[str, object]:
        from cipher_lab.data.cubes import CubeCollector

        assert isinstance(recipe, SearchRecipe) and isinstance(collector, CubeCollector)
        seen.append(recipe)
        store.finalize({"status": "SUCCEEDED", "summary": {
            "method": "integral-search-v1", "backend_identity": backend_identity,
            "tiers": {"discovery": {"status": "COMPLETED"}}}})
        return {"status": "WRONG"}
    monkeypatch.setattr(method, "run_integral", terminal)
    root = tmp_path / "out"
    assert cli.main(["run", *settings(tmp_path), "--root", str(root), "--id", "search",
                     "--backend", backend]) == 0
    output = json.loads(capsys.readouterr().out)
    assert seen and output["report_status"] == "SAVED"
    assert not (root / "models").exists()
    assert read_saved(root, "search")["status"] == "SUCCEEDED"


def saved_search(tmp_path: Path) -> tuple[RunStore, dict[str, object]]:
    store = RunStore.claim(tmp_path, "saved", output_bytes=1000000)
    ref = store.write_payload("basis", {"frozen_basis": list(range(1, 21))})
    store.write_snapshot({"recipe": recipe().to_dict(), "counts": {
        "completed_real_groups": 2112, "completed_null_groups": 17}})
    store.finalize({"status": "STOPPED", "summary": {
        "method": "integral-search-v1", "backend_identity": "demo", "reason": "CPU_LIMIT",
        "tiers": {"discovery": {"status": "COMPLETED"},
                  "expanded": {"status": "GATED", "gate_cell_index": 8}},
        "selections": [{"rounds": 11, "tier": "discovery", "pattern_index": 1,
                        "dimensions": [0, 0], "ranks": [2080, 2079],
                        "rule": "SAVED_RULE<script>"}],
        "cells": [{"cell": {"rounds": 11, "active": [0, 2]}, "tier": "discovery",
                   "pattern_index": 0, "outcome": "NOT_APPLICABLE",
                   "real": {"discovery_dimension": 0, "discovery_rank": 2080,
                            "frozen_dimension": 0, "confirmation": "NOT_APPLICABLE",
                            "basis_ref": ref, "stages": {
                                "discovery": {"requested": 2112, "collected": 2112,
                                              "status": "COLLECTED"},
                                "screen": {"requested": 320, "collected": 0,
                                           "status": "NOT_APPLICABLE"}}}},
                  {"tier": "expanded", "rounds": 12, "pattern_index": 1,
                   "outcome": "NOT_STARTED_GATE", "not_started_reason": "REAL_NOT_CONFIRMED"},
                  {"tier": "expanded", "outcome": "INCOMPLETE",
                   "real": {"stages": {"discovery": {"requested": 2112,
                                                       "status": "INCOMPLETE"}}}}]}})
    return store, read_saved(tmp_path, "saved")


def test_search_renderer_saved_evidence_no_reselection_or_math(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, saved = saved_search(tmp_path)
    before = deepcopy(saved)
    terminal = (tmp_path / "data/saved/run.json").read_bytes()
    original = builtins.__import__
    def no_science(name: str, globals: Mapping[str, object] | None = None,
                   locals: Mapping[str, object] | None = None,
                   fromlist: tuple[str, ...] = (), level: int = 0) -> object:
        if name.startswith("cipher_lab"):
            pytest.fail("renderer imported science/selection")
        return original(name, globals, locals, fromlist, level)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", no_science)
        patch.setattr(Path, "read_bytes", lambda p: pytest.fail("report I/O"))
        files = report.render_report(saved)
    for filename in ("index.html", "report.md"):
        text = files[filename].decode()
        for expected in ("tier", "discovery", "expanded", "pattern_index", "2080",
                         "2112", "320", "256", "17", "NOT_STARTED_GATE", "未访问不等于阴性",
                         "SAVED_RULE", "2079", "探索内", "不是证明", "INCOMPLETE"):
            assert expected in text
        assert "<script>" not in text
        assert "选择维数" in text and "D ranks" in text
        assert "实际采组" in text and "跳过" in text
    html = files["index.html"].decode()
    assert html.count('class="representative"') == 4
    assert 'data-dimension="20"' not in html
    assert html.count("<svg") == 2
    assert saved == before
    store.write_report_files(files)
    assert (tmp_path / "data/saved/run.json").read_bytes() == terminal
    assert json.loads(files["source.json"])["run_sha256"] == saved["run_sha256"]


def test_report_branch_uses_saved_method_not_recipe_shape(tmp_path: Path) -> None:
    _, saved = saved_search(tmp_path)
    summary = cast(dict[str, object], saved["summary"])
    summary["method"] = "integral-empirical-v1"
    assert "探索内" not in report.render_report(saved)["index.html"].decode()
    summary.pop("method")
    assert "探索内" in report.render_report(saved)["index.html"].decode()
    saved["recipe"] = {"method": "unknown"}
    assert "探索内" not in report.render_report(saved)["index.html"].decode()


def test_search_partial_unknown_stage_counts_are_not_zero_filled(tmp_path: Path) -> None:
    _, saved = saved_search(tmp_path)
    html = report.render_report(saved)["index.html"].decode()
    assert "实际采组 未保存 / 请求 2112" in html
    assert "null：发现维数 未保存" in html
    assert "NOT_STARTED_GATE" in html and "非阴性" in html


@pytest.mark.parametrize("saved_method", [None, "unknown", "integral-search-v1"])
def test_saved_search_dispatch_identity_and_terminal_immutable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], saved_method: str | None,
) -> None:
    store = RunStore.claim(tmp_path, "dispatch", output_bytes=1000000)
    store.finalize({"status": "FAILED", "summary": {
        **({"method": saved_method} if saved_method is not None else {}),
        "backend_identity": "UNVERIFIED"}})
    path = tmp_path / "data/dispatch/run.json"
    before = path.read_bytes()
    code = cli._saved_report(store, tmp_path, "dispatch", "demo")
    output = json.loads(capsys.readouterr().out)
    assert code == (4 if saved_method == "integral-search-v1" else 6)
    assert output["backend_identity"] == "UNVERIFIED"
    assert output["report_status"] == ("SAVED" if code == 4 else "FAILED")
    if code == 6:
        assert output["reason"] == "REPORT_METHOD_UNKNOWN"
    assert path.read_bytes() == before


@pytest.mark.parametrize("point", ["render", "write", "read"])
def test_search_report_failure_preserves_terminal_and_same_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], point: str,
) -> None:
    import cipher_lab.artifacts as artifacts

    store, _ = saved_search(tmp_path)
    path = tmp_path / "data/saved/run.json"
    before = path.read_bytes()
    def broken(*a: object, **k: object) -> dict[str, bytes]:
        raise RuntimeError("PRIVATE")
    if point == "render":
        monkeypatch.setattr(report, "render_report", broken)
    elif point == "write":
        monkeypatch.setattr(RunStore, "write_report_files", broken)
    else:
        monkeypatch.setattr(artifacts, "read_saved", broken)
    monkeypatch.setattr(RunStore, "finalize", lambda *a: pytest.fail("no refinalize"))
    assert cli._saved_report(store, tmp_path, "saved", "simon") == (7 if point == "read" else 6)
    output = capsys.readouterr().out
    assert "PRIVATE" not in output
    assert json.loads(output)["scientific_status"] == ("UNKNOWN" if point == "read" else "STOPPED")
    assert path.read_bytes() == before


def test_search_demo_real_method_full_observer_and_frozen_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "out"
    assert cli.main(["run", *settings(tmp_path), "--root", str(root), "--id", "loop",
                     "--backend", "demo"]) == 5
    output = json.loads(capsys.readouterr().out)
    assert output["report_status"] == "SAVED"
    saved = read_saved(root, "loop")
    summary = cast(dict[str, object], saved["summary"])
    assert summary["method"] == "integral-search-v1"
    cells = cast(list[dict[str, object]], summary["cells"])
    real = cast(dict[str, object], cells[0]["real"])
    assert real["frozen_dimension"] == 2079
    payloads = cast(dict[str, dict[str, object]], saved["payloads"])
    assert len(cast(list[int], payloads[str(real["basis_ref"])]["frozen_basis"])) == 2079
    counts = cast(dict[str, int], saved["counts"])
    assert counts["completed_real_blocks"] == counts["completed_null_words"] == 12
    assert counts["completed_real_groups"] == counts["completed_null_groups"] == 3
    html = Path(str(output["report_path"])).read_text()
    assert "NOT_STARTED" in html and "未访问不等于阴性" in html
    assert "2080" in html and not (root / "models").exists()


def test_search_report_distinguishes_completed_negative_from_skips(tmp_path: Path) -> None:
    _, saved = saved_search(tmp_path)
    for filename in ("index.html", "report.md"):
        text = report.render_report(saved)[filename].decode()
        assert "完整采集后 K_D / K_DS 为空才是该 cell 的经验阴性" in text
        assert "C FAILED 拒绝整个冻结空间，不挑幸存者" in text


def test_search_report_displays_even_inconsistent_saved_choice_without_reselection(
    tmp_path: Path,
) -> None:
    _, saved = saved_search(tmp_path)
    summary = cast(dict[str, object], saved["summary"])
    summary["selections"] = [{"rounds": 11, "dimensions": [17, 19],
                              "ranks": [2063, 2061], "pattern_index": 0,
                              "rule": "SAVED_ONLY"}]
    html = report.render_report(saved)["index.html"].decode()
    assert '&quot;pattern_index&quot;: 0' in html
    assert '17' in html and '19' in html and '2063' in html and '2061' in html
    assert "SAVED_ONLY" in html


def test_high_static_worst_case_three_cells_and_discovery_only_counts() -> None:
    high = load_experiment(PROJECT / "configs/integral-search-k18k20.json")
    assert isinstance(high, SearchRecipe)
    estimate = cli._estimate(high)
    assert estimate["planned_real_blocks"] == estimate["planned_null_words"] == 4227858432
    assert estimate["planned_real_groups"] == estimate["planned_null_groups"] == 8064
    assert estimate["planned_optimizer_updates"] == estimate["planned_forward_rows"] == 0
    cells = cast(list[dict[str, object]], estimate["cells"])
    assert len(cells) == 3
    assert [cell["tier"] for cell in cells] == ["discovery", "discovery", "expanded"]
    assert [cell["points_per_group"] for cell in cells] == [262144, 262144, 1048576]
    assert cells[-1]["conditional"] is True
    assert sum(cast(int, cell["points_per_group"]) * high.groups.discovery
               for cell in cells) == 3321888768
    assert len(cells) * high.groups.discovery == 6336


def test_high_search_check_is_static_even_with_insufficient_policy(tmp_path: Path) -> None:
    args = settings(tmp_path)
    (tmp_path / "config.json").write_bytes(
        (PROJECT / "configs/integral-search-k18k20.json").read_bytes())
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('numpy', 'torch', 'cipher_lab.integral', 'cipher_lab.data',
                                'cipher_lab.ciphers', 'cipher_lab.execution',
                                'cipher_lab.artifacts')):
            raise AssertionError('scientific import during high check')
sys.meta_path.insert(0, Block())
from cipher_lab_plugins.integral_search.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    done = subprocess.run([sys.executable, "-c", script, "check", *args], cwd=tmp_path,
                          capture_output=True, text=True, check=False,
                          env=dict(os.environ, PYTHONPATH=os.pathsep.join(
                              (str(PROJECT / "src"), str(PROJECT / "plugins"))),
                                   PYTHONDONTWRITEBYTECODE="1"))
    assert done.returncode == 0, done.stderr
    output = json.loads(done.stdout)
    assert output["estimate"]["planned_real_blocks"] == 4227858432
    assert output["planned_calls_fit_policy"]["real_blocks"] is False
    assert output["planned_calls_fit_policy"]["null_words"] is False
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_search_claim_failure_precedes_meter_and_preserves_existing_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = ["run", *settings(tmp_path), "--root", str(tmp_path / "out"),
                 "--id", "sample", "--backend", "demo"]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "schema": 1, "method": "integral-search-v1", "rounds": [11],
        "patterns": [[0, 2, 4], [32, 34, 36]], "discovery_active": 2,
        "expanded_active": 3, "groups": {"discovery": 1, "screen": 1, "confirm": 1},
        "chunk_points": 4,
    }))
    RunStore.claim(tmp_path / "out", "sample", output_bytes=100000)
    before = {p: p.read_bytes() for p in (tmp_path / "out").rglob("*") if p.is_file()}
    monkeypatch.setattr(execution, "Meter", lambda *a: pytest.fail("Meter before claim"))
    assert cli.main(arguments) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "CLAIM_FAILED"
    assert {p: p.read_bytes() for p in before} == before
    assert not (tmp_path / "out/models").exists()



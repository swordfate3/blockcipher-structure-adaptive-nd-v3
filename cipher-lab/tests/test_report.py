"""Pure renderer tests using saved synthetic summaries, never scientific sampling."""
import builtins
import importlib
import json
import sys
from collections.abc import Mapping
from hashlib import sha256
from html import unescape
from pathlib import Path
from typing import cast

import pytest

from cipher_lab.artifacts import ArtifactError, RunStore, read_saved
from cipher_lab.integral.report import render_report


def fixture_saved(tmp_path: Path) -> tuple[RunStore, dict[str, object]]:
    store = RunStore.claim(tmp_path, "demo", output_bytes=1000000)
    ref = store.write_payload("basis", {"discovery_basis": list(range(1, 11)),
                                        "frozen_basis": list(range(1, 11))})
    store.write_snapshot({"recipe": {"method": "<script>alert(1)</script>"},
                          "policy": {"wall_seconds": 12}, "counts": {"completed_real_groups": 3}})
    store.finalize({"status": "SUCCEEDED", "summary": {
        "backend_identity": "demo", "scientific_validation": False,
        "reason": '<img src=x onerror="x">',
        "meter": {"wall_elapsed_seconds": 2.5, "cpu_elapsed_seconds": 1.2},
        "cells": [{"cell": {"rounds": 1, "active": [0, 1]}, "outcome": "CONFIRMED",
                   "real": {"discovery_dimension": 11, "frozen_dimension": 10,
                            "confirmation": "PASSED", "basis_ref": ref},
                   "null": {"discovery_dimension": 9, "frozen_dimension": 0,
                            "confirmation": "NOT_APPLICABLE"}}]}})
    return store, read_saved(tmp_path, "demo")


def test_three_pure_files_use_frozen_sha_saved_dimensions_and_only_four_representatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, saved = fixture_saved(tmp_path)
    before = (tmp_path / "data/demo/run.json").read_bytes()
    original_import = builtins.__import__

    def no_science(name: str, globals: Mapping[str, object] | None = None,
                   locals: Mapping[str, object] | None = None,
                   fromlist: tuple[str, ...] = (), level: int = 0) -> object:
        if any(part in name for part in ("cipher_lab.ciphers", "cipher_lab.data", "linear",
                                         "observables", "model")):
            raise AssertionError("renderer must not import science")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", no_science)
        importlib.reload(sys.modules[render_report.__module__])
        patch.setattr(Path, "read_bytes", lambda p: pytest.fail("renderer performed I/O"))
        patch.setattr(Path, "write_bytes", lambda p, b: pytest.fail("renderer performed I/O"))
        files = render_report(saved)
    assert set(files) == {"report.md", "index.html", "source.json"}
    source = json.loads(files["source.json"])
    assert source == {"schema": 1, "run_id": "demo", "run_ref": "data/demo/run.json",
                      "run_sha256": sha256(before).hexdigest()}
    html = files["index.html"].decode()
    md = files["report.md"].decode()
    assert html.count("<svg") == 2
    assert "11" in html and "10" in html
    assert 'data-dimension="10"' in html
    assert html.count('class="representative"') == 4
    assert "0x4" in html and "0x5" not in html
    assert "完整基" in md and "NOT_APPLICABLE" in md
    for text in (html, md):
        assert "合成工程" in text and "经验" in text and "不是证明" in text
        assert "2.5" in text and "数据量" in text and "限制" in text
        assert "../data" not in text and "download" not in text
        assert "<script>" not in text and "<img src=x" not in text
    assert "&lt;script&gt;" in html and "&lt;img" in html
    store.write_report_files(files)
    assert (tmp_path / "data/demo/run.json").read_bytes() == before
    assert render_report(saved) == files


def test_report_failure_does_not_change_scientific_terminal(tmp_path: Path) -> None:
    store, saved = fixture_saved(tmp_path)
    before = (tmp_path / "data/demo/run.json").read_bytes()
    occupied = tmp_path / "result/demo/index.html"
    occupied.write_bytes(b"foreign")
    with pytest.raises(ArtifactError):
        store.write_report_files(render_report(saved))
    assert (tmp_path / "data/demo/run.json").read_bytes() == before
    assert read_saved(tmp_path, "demo")["status"] == "SUCCEEDED"


@pytest.mark.parametrize("status", ["RUNNING", "STOPPED", "FAILED", "ALARM"])
def test_incomplete_or_negative_result_never_becomes_scientific_success(
    tmp_path: Path, status: str,
) -> None:
    _, saved = fixture_saved(tmp_path)
    saved["status"] = status
    saved["summary"] = {"backend_identity": "UNVERIFIED", "cells": [],
                        "reason": "PUBLIC_REASON"}
    html = render_report(saved)["index.html"].decode()
    assert status in html
    assert "未核验" in html and "科学验证" in html
    if status == "RUNNING":
        assert "UNKNOWN" in html


def test_renderer_uses_saved_summary_not_recomputed_payload_metrics(tmp_path: Path) -> None:
    _, saved = fixture_saved(tmp_path)
    payloads = cast(dict[str, object], saved["payloads"])
    payloads["payload/basis.json"] = {"frozen_basis": [0] * 200}
    files = render_report(saved)
    assert b'data-dimension="10"' in files["index.html"]
    assert 'data-dimension="200"' not in files["report.md"].decode()
    assert "冻结维数 10" in files["report.md"].decode()


@pytest.mark.parametrize("wall,cpu", [
    ("[墙钟](https://invalid/wall)", "![CPU](https://invalid/cpu)"),
    ("![墙钟](https://invalid/wall)", "[CPU](https://invalid/cpu)"),
])
def test_saved_elapsed_values_are_literal_markdown_not_links_or_images(
    tmp_path: Path, wall: str, cpu: str,
) -> None:
    store = RunStore.claim(tmp_path, "timing", output_bytes=1000000)
    store.finalize({"status": "SUCCEEDED", "summary": {
        "meter": {"wall_elapsed_seconds": wall, "cpu_elapsed_seconds": cpu}}})
    saved = read_saved(tmp_path, "timing")
    files = render_report(saved)
    md = files["report.md"].decode()
    html = files["index.html"].decode()
    assert wall not in md and cpu not in md
    assert "](" not in md and "![" not in md
    assert "&#91;墙钟&#93;&#40;https://invalid/wall&#41;" in md
    assert "&#91;CPU&#93;&#40;https://invalid/cpu&#41;" in md
    assert "&#33;&#91;" in md
    timing = next(line for line in md.splitlines() if line.startswith("耗时："))
    assert unescape(timing).startswith(f"耗时：墙钟 {wall} 秒；CPU {cpu} 秒。")
    # HTML treats Markdown punctuation as literal text, not resource elements.
    assert wall in html and cpu in html
    assert "<a " not in html and "<img " not in html


def test_markdown_user_text_cannot_create_data_or_remote_links(tmp_path: Path) -> None:
    _, saved = fixture_saved(tmp_path)
    saved["summary"] = {"reason": "[打开](../data/private.json) ![图片](https://invalid/x)",
                        "cells": []}
    files = render_report(saved)
    md = files["report.md"].decode()
    html = files["index.html"].decode()
    assert "[打开](../data" not in md
    assert "![图片](https" not in md
    assert '<a ' not in html and '<img ' not in html


@pytest.mark.parametrize("backend,marker", [("demo", "合成工程"), ("simon", "不认证"),
                                           ("UNVERIFIED", "未核验")])
def test_backend_identity_distinguishes_demo_adapter_and_unknown(
    tmp_path: Path, backend: str, marker: str,
) -> None:
    _, saved = fixture_saved(tmp_path)
    saved["summary"] = {"backend_identity": backend, "scientific_validation": False}
    for filename in ("index.html", "report.md"):
        text = render_report(saved)[filename].decode()
        assert marker in text
        if backend == "simon":
            assert "密码实现或科学有效性" in text
        if backend == "demo":
            assert "不是真实密码科学结果" in text

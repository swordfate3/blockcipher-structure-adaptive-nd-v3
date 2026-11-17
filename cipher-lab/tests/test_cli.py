"""CLI lifecycle using fake engineering inputs only; never invoke production cipher/RSS."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

import cipher_lab.artifacts as artifacts
import cipher_lab.cli as cli
import cipher_lab.execution as execution
import cipher_lab.integral.method as method
import cipher_lab.integral.report as report
from cipher_lab.artifacts import RunStore, read_saved

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(autouse=True)
def forbid_production_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("production cipher/RNG/RSS called")
    monkeypatch.setattr("cipher_lab.ciphers.simon.encrypt_many", forbidden)
    monkeypatch.setattr("cipher_lab.ciphers.simon.encrypt_batch", forbidden)
    monkeypatch.setattr("secrets.token_bytes", forbidden)
    monkeypatch.setattr(execution, "production_rss_bytes", forbidden)


def inputs(tmp_path: Path, **limits: object) -> list[str]:
    config = tmp_path / "config.json"
    policy = tmp_path / "policy.json"
    config.write_text(
        json.dumps(
            {
                "schema": 1,
                "method": "integral-empirical-v1",
                "cells": [{"rounds": 3, "active": [32, 33]}],
                "groups": {"discovery": 1, "screen": 1, "confirm": 1},
                "chunk_points": 3,
            }
        )
    )
    policy.write_text(
        json.dumps(
            {
                "schema": 2,
                "optimizer_updates": 0,
                "forward_rows": 0,
                "wall_seconds": 120,
                "cpu_seconds": 120,
                "deadline_utc": "2099-01-01T00:00:00Z",
                "real_blocks": 100,
                "null_words": 100,
                "rss_stop_bytes": 1000000,
                "output_bytes": 20000000,
                **limits,
            }
        )
    )
    return ["--config", str(config), "--policy", str(policy)]


def args(tmp_path: Path, **limits: object) -> list[str]:
    return [
        "run",
        *inputs(tmp_path, **limits),
        "--root",
        str(tmp_path / "out"),
        "--id",
        "sample",
        "--backend",
        "demo",
    ]


def result(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    captured = capsys.readouterr()
    assert "PRIVATE" not in captured.out + captured.err
    assert "Traceback" not in captured.err
    return cast(dict[str, object], json.loads(captured.out))


def test_subprocess_check_and_demo_poison_real_cipher_and_private_random(tmp_path: Path) -> None:
    settings = inputs(tmp_path)
    script = """
import importlib.abc, secrets, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('cipher_lab.ciphers', 'cipher_experiments', 'torch')):
            raise AssertionError('PRIVATE production import')
sys.meta_path.insert(0, Block())
def poison(*a, **k):
    raise AssertionError('PRIVATE production call')
secrets.token_bytes = poison
from cipher_lab import execution
execution.production_rss_bytes = poison
from cipher_lab.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1")
    before = sorted(tmp_path.rglob("*"))
    check = subprocess.run(
        [sys.executable, "-c", script, "check", *settings],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr
    assert sorted(tmp_path.rglob("*")) == before
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "run",
            *settings,
            "--root",
            str(tmp_path / "out"),
            "--id",
            "sample",
            "--backend",
            "demo",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode in (0, 5), run.stderr
    output = json.loads(run.stdout)
    assert output["scientific_status"] in ("SUCCEEDED", "ALARM")
    assert output["report_failed"] is False
    saved = read_saved(tmp_path / "out", "sample")
    summary = cast(dict[str, object], saved["summary"])
    assert summary["backend_identity"] == "demo"
    assert summary["scientific_validation"] is False
    counts = cast(dict[str, int], saved["counts"])
    assert counts["completed_real_blocks"] == counts["attempted_real_blocks"] == 12
    assert counts["completed_null_words"] == counts["attempted_null_words"] == 12
    assert counts["completed_real_groups"] == counts["completed_null_groups"] == 3
    assert summary["meter"] is not None
    assert saved["policy"] == json.loads((tmp_path / "policy.json").read_text())
    cells = cast(list[dict[str, object]], summary["cells"])
    real = cast(dict[str, object], cells[0]["real"])
    assert (
        real["frozen_dimension"] == 2079
    )  # one nonzero quadratic constraint in the full 2080 observer
    payloads = cast(dict[str, dict[str, object]], saved["payloads"])
    assert len(cast(list[int], payloads[str(real["basis_ref"])]["frozen_basis"])) == 2079
    html = Path(output["report_path"]).read_text()
    assert "合成工程" in html and "不是真实" in html
    assert "模拟" in output["count_semantics"]
    source = json.loads((tmp_path / "out/result/sample/source.json").read_text())
    assert source["run_sha256"] == saved["run_sha256"]
    assert not (tmp_path / "out/models").exists()


@pytest.mark.parametrize("tail", [[], ["--backend", "auto"], ["--backend", "simon", "--dry-run"]])
def test_run_requires_explicit_backend_and_no_fake_dry_run(tmp_path: Path, tail: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            ["run", *inputs(tmp_path), "--root", str(tmp_path / "out"), "--id", "sample", *tail]
        )
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


def test_run_requires_root_and_no_legacy_commands(tmp_path: Path) -> None:
    for command in (
        ["run", *inputs(tmp_path), "--id", "x", "--backend", "demo"],
        ["evaluate"],
        ["train"],
        ["runtrain"],
    ):
        with pytest.raises(SystemExit) as error:
            cli.main(command)
        assert error.value.code == 2
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", ".", "a/b"])
def test_invalid_id_does_not_claim_or_import_backend(
    tmp_path: Path,
    run_id: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = args(tmp_path)
    arguments[arguments.index("--id") + 1] = run_id
    monkeypatch.setattr(execution, "Meter", lambda *a: pytest.fail("Meter before claim"))
    assert cli.main(arguments) == 2
    assert result(capsys)["scientific_status"] == "UNKNOWN"
    assert not (tmp_path / "out").exists()


def test_occupied_run_is_not_overwritten(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = args(tmp_path, real_blocks=0)
    assert cli.main(arguments) == 3
    result(capsys)
    before = {p: p.read_bytes() for p in (tmp_path / "out").rglob("*") if p.is_file()}
    assert cli.main(arguments) == 2
    assert result(capsys)["reason"] == "CLAIM_FAILED"
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize(
    "limits,reason",
    [
        ({"real_blocks": 0}, "REAL_BLOCKS_LIMIT"),
        ({"deadline_utc": "2000-01-01T00:00:00Z"}, "DEADLINE"),
    ],
)
def test_cooperative_stop_is_saved_and_reported(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    limits: dict[str, object],
    reason: str,
) -> None:
    assert cli.main(args(tmp_path, **limits)) == 3
    output = result(capsys)
    assert output["scientific_status"] == "STOPPED" and output["report_failed"] is False
    saved = read_saved(tmp_path / "out", "sample")
    assert cast(dict[str, object], saved["summary"])["reason"] == reason


def test_method_private_failure_is_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken() -> None:
        raise RuntimeError("PRIVATE sample key and conversation")

    monkeypatch.setattr(method, "_engineering_control", broken)
    assert cli.main(args(tmp_path)) == 4
    output = result(capsys)
    assert output["scientific_status"] == "FAILED" and output["report_failed"] is False
    assert all(
        b"PRIVATE" not in p.read_bytes() for p in (tmp_path / "out").rglob("*") if p.is_file()
    )


@pytest.mark.parametrize(
    "status,code", [("SUCCEEDED", 0), ("STOPPED", 3), ("FAILED", 4), ("ALARM", 5)]
)
def test_saved_status_not_return_value_controls_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    code: int,
) -> None:
    def terminal(
        recipe: object, collector: object, meter: object, store: RunStore, *, backend_identity: str
    ) -> dict[str, object]:
        assert backend_identity == "demo"
        store.finalize({"status": status, "summary": {"backend_identity": backend_identity}})
        return {"status": "WRONG"}

    monkeypatch.setattr(method, "run_integral", terminal)
    assert cli.main(args(tmp_path)) == code
    assert result(capsys)["scientific_status"] == status


@pytest.mark.parametrize("point", ["render", "write", "read"])
def test_report_failure_does_not_mutate_scientific_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    point: str,
) -> None:
    def terminal(
        recipe: object, collector: object, meter: object, store: RunStore, *, backend_identity: str
    ) -> dict[str, object]:
        store.finalize({"status": "SUCCEEDED"})
        return {}

    def broken(*a: object, **k: object) -> dict[str, bytes]:
        raise RuntimeError("PRIVATE filesystem details")

    monkeypatch.setattr(method, "run_integral", terminal)
    if point == "render":
        monkeypatch.setattr(report, "render_report", broken)
    elif point == "write":
        monkeypatch.setattr(RunStore, "write_report_files", broken)
    else:
        # Reading fails: no independently known terminal may be claimed.
        monkeypatch.setattr(artifacts, "read_saved", broken)
    assert cli.main(args(tmp_path)) == (7 if point == "read" else 6)
    output = result(capsys)
    assert output["scientific_status"] == ("UNKNOWN" if point == "read" else "SUCCEEDED")
    assert output["report_failed"] is True
    assert read_saved(tmp_path / "out", "sample")["status"] == "SUCCEEDED"


@pytest.mark.parametrize("raises", [False, True])
def test_missing_terminal_is_unknown_not_repaired_after_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    raises: bool,
) -> None:
    def incomplete(*a: object, **k: object) -> dict[str, object]:
        if raises:
            raise RuntimeError("PRIVATE")
        return {"status": "SUCCEEDED"}

    monkeypatch.setattr(method, "run_integral", incomplete)
    assert cli.main(args(tmp_path)) == 7
    output = result(capsys)
    assert output["scientific_status"] == "UNKNOWN"
    assert output["report_status"] == "NOT_ATTEMPTED"
    assert read_saved(tmp_path / "out", "sample")["status"] == "RUNNING"


def test_meter_initialization_failure_preserves_public_setup_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken(*a: object) -> None:
        raise execution.Stopped("CLOCK_UNAVAILABLE")

    monkeypatch.setattr(execution, "Meter", broken)
    assert cli.main(args(tmp_path)) == 3
    assert result(capsys)["scientific_status"] == "STOPPED"
    saved = read_saved(tmp_path / "out", "sample")
    assert cast(dict[str, object], saved["summary"])["reason"] == "CLOCK_UNAVAILABLE"
    assert cast(dict[str, object], saved["summary"])["backend_identity"] == "demo"
    assert saved["policy"] == json.loads((tmp_path / "policy.json").read_text())


def test_setup_and_output_failure_never_echo_exceptions_or_refinalize_sealed_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken(*a: object) -> None:
        raise RuntimeError("PRIVATE")

    monkeypatch.setattr(execution, "Meter", broken)
    assert cli.main(args(tmp_path)) == 4
    assert result(capsys)["scientific_status"] == "FAILED"
    arguments = args(tmp_path, output_bytes=1)
    arguments[arguments.index("--id") + 1] = "small"
    assert cli.main(arguments) == 2
    assert result(capsys)["scientific_status"] == "UNKNOWN"


def test_report_writes_follow_read_saved_and_do_not_change_terminal_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = RunStore.write_report_files

    def write(store: RunStore, files: Mapping[str, bytes]) -> None:
        path = tmp_path / "out/data/sample/run.json"
        before = path.read_bytes()
        assert json.loads(before)["status"] in ("SUCCEEDED", "ALARM")
        assert json.loads(files["source.json"])["run_sha256"] == sha256(before).hexdigest()
        original(store, files)
        assert path.read_bytes() == before

    monkeypatch.setattr(RunStore, "write_report_files", write)
    assert cli.main(args(tmp_path)) in (0, 5)
    assert result(capsys)["report_failed"] is False


def test_demo_unique_engineering_bytes_are_not_production_randomness() -> None:
    from cipher_lab.demo import DemoBytes, fake_encrypt_many

    source = DemoBytes()
    assert len({source(16) for _ in range(1000)}) == 1000
    assert len(source(8)) == 8
    assert fake_encrypt_many((0, 1 << 63), (0, 0, 0, 0), 3) == (3, (1 << 63) ^ 3)
    with pytest.raises(ValueError):
        source(3)


def test_claim_failure_json_exit_matches_process_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = args(tmp_path, output_bytes=1)
    assert cli.main(arguments) == 2
    assert result(capsys)["exit_code"] == 2


def test_snapshot_failure_never_refinalizes_storage_owner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(store: RunStore, value: Mapping[str, object]) -> None:
        raise RuntimeError("PRIVATE")

    def invalid_finalize(*a: object) -> None:
        pytest.fail("snapshot failure may have sealed store; do not finalize")

    monkeypatch.setattr(RunStore, "write_snapshot", broken)
    monkeypatch.setattr(RunStore, "finalize", invalid_finalize)
    assert cli.main(args(tmp_path)) == 7
    assert result(capsys)["scientific_status"] == "UNKNOWN"


def test_settings_private_exception_text_not_echoed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(path: Path) -> None:
        raise ValueError("PRIVATE")

    monkeypatch.setattr(cli, "load_experiment", broken)
    assert cli.main(["check", *inputs(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert "PRIVATE" not in captured.out + captured.err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("backend", ["demo", "simon"])
@pytest.mark.parametrize("failure_at", ["settings", "method_snapshot", "payload"])
def test_method_storage_failure_uses_saved_identity_without_second_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    backend: str, failure_at: str,
) -> None:
    from cipher_lab.demo import DemoBytes, demo_rss_bytes, fake_encrypt_batch

    # Even the simon assembly path uses only fake oracle/randomness/RSS.
    fake_cipher = ModuleType("cipher_lab.ciphers.simon")
    fake_cipher.encrypt_batch = fake_encrypt_batch  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cipher_lab.ciphers.simon", fake_cipher)
    private = DemoBytes()
    monkeypatch.setattr("secrets.token_bytes", lambda size: (
        private(size) if size in (8, 16) else private.bulk_words(size // 8)))
    monkeypatch.setattr(execution, "production_rss_bytes", demo_rss_bytes)
    original_payload = RunStore.write_payload
    original_snapshot = RunStore.write_snapshot

    def overflow_payload(store: RunStore, name: str, value: object) -> str:
        return original_payload(store, name, "x" * 20000001)

    def overflow_snapshot(store: RunStore, value: Mapping[str, object]) -> None:
        selected = ((failure_at == "settings" and "policy" in value)
                    or (failure_at == "method_snapshot" and "counts" in value))
        if selected:
            # The failed proposed snapshot must not replace committed identity.
            original_snapshot(store, {**value, "summary": {
                "backend_identity": "simon" if backend == "demo" else "demo",
                "scientific_validation": True, "padding": "x" * 20000001,
            }})
        else:
            original_snapshot(store, value)

    def forbidden(*a: object) -> None:
        pytest.fail("sealed store must not be finalized a second time")

    if failure_at == "payload":
        monkeypatch.setattr(RunStore, "write_payload", overflow_payload)
    else:
        monkeypatch.setattr(RunStore, "write_snapshot", overflow_snapshot)
    monkeypatch.setattr(RunStore, "finalize", forbidden)
    arguments = args(tmp_path)
    arguments[-1] = backend
    # Setup-time overflow on the recipe/policy snapshot leaves saved["recipe"] empty,
    # so the saved record has no method identity. The CLI must not render or infer
    # an integral report from that absence; it returns REPORT_METHOD_UNKNOWN (6).
    # method_snapshot/payload failures occur after the recipe was committed, so
    # those paths still dispatch to the integral renderer and keep exit 4/SAVED.
    expected_code = 6 if failure_at == "settings" else 4
    assert cli.main(arguments) == expected_code
    output = result(capsys)
    assert output["scientific_status"] == "FAILED"
    expected_report_failed = failure_at == "settings"
    assert output["report_failed"] is expected_report_failed
    saved = read_saved(tmp_path / "out", "sample")
    assert saved["summary"] == {
        "reason": "OUTPUT_LIMIT", "backend_identity": backend, "scientific_validation": False,
    }
    assert output["backend_identity"] == backend
    assert output["scientific_validation"] is False
    if failure_at == "settings":
        assert output["reason"] == "REPORT_METHOD_UNKNOWN"
        assert "report_path" not in output
        assert not (tmp_path / "out" / "result" / "sample" / "index.html").exists()
        assert not (tmp_path / "out" / "result" / "sample" / "source.json").exists()
    else:
        html = Path(str(output["report_path"])).read_text()
        assert ("合成工程演示" in html) is (backend == "demo")
        assert ("真实后端身份由装配层记录" in html) is (backend == "simon")
        assert "OUTPUT_LIMIT" in html and "尚无已保存维数" in html
        source = json.loads((tmp_path / "out/result/sample/source.json").read_text())
        assert source["run_sha256"] == saved["run_sha256"]


@pytest.mark.parametrize("active,rounds", [([0, 1], 1), ([40, 41, 42], 44)])
def test_demo_honors_variable_cells_without_scientific_shortcuts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], active: list[int], rounds: int,
) -> None:
    arguments = args(tmp_path, real_blocks=24, null_words=24)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["cells"] = [{"rounds": rounds, "active": active}]
    config_path.write_text(json.dumps(config))
    assert cli.main(arguments) in (0, 5)
    result(capsys)
    saved = read_saved(tmp_path / "out", "sample")
    counts = cast(dict[str, int], saved["counts"])
    assert counts["completed_real_blocks"] == (12 if len(active) == 2 else 24)
    assert saved["recipe"] == config


@pytest.mark.parametrize("report_import_fails", [False, True])
def test_execution_import_failure_is_public_and_uses_saved_setup_terminal(
    tmp_path: Path, report_import_fails: bool,
) -> None:
    arguments = args(tmp_path)
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cipher_lab.execution':
            raise RuntimeError('PRIVATE execution import canary')
        if REPORT_IMPORT_FAILS and fullname == 'cipher_lab.integral.report':
            raise RuntimeError('PRIVATE report import canary')
sys.meta_path.insert(0, Block())
from cipher_lab.cli import main
raise SystemExit(main(sys.argv[1:]))
""".replace("REPORT_IMPORT_FAILS", repr(report_import_fails))
    completed = subprocess.run(
        [sys.executable, "-c", script, *arguments], cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, check=False,
    )
    assert "PRIVATE" not in completed.stdout + completed.stderr
    assert "Traceback" not in completed.stdout + completed.stderr
    output = json.loads(completed.stdout)
    assert completed.returncode == output["exit_code"] == (6 if report_import_fails else 4)
    assert output["scientific_status"] == "FAILED"
    assert output["report_failed"] is report_import_fails
    assert output["reason"] == ("REPORT_FAILED" if report_import_fails else "BACKEND_SETUP_FAILED")
    saved = read_saved(tmp_path / "out", "sample")
    assert saved["status"] == "FAILED"
    assert saved["summary"] == {
        "backend_identity": "demo", "scientific_validation": False,
        "reason": "BACKEND_SETUP_FAILED",
    }
    assert all(b"PRIVATE" not in p.read_bytes()
               for p in (tmp_path / "out").rglob("*") if p.is_file())
    if not report_import_fails:
        html = Path(output["report_path"]).read_text()
        assert "合成工程演示" in html and "BACKEND_SETUP_FAILED" in html
        source = json.loads((tmp_path / "out/result/sample/source.json").read_text())
        assert source["run_sha256"] == saved["run_sha256"]


@pytest.mark.parametrize("saved_backend", ["simon", "UNVERIFIED", None])
def test_saved_identity_not_current_cli_backend_controls_public_exit_and_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], saved_backend: str | None,
) -> None:
    root = tmp_path / "out"
    store = RunStore.claim(root, "sample", output_bytes=100_000)
    # Commit a known recipe identity first so this test continues to probe that
    # the saved backend (including its absence) — not the current CLI argument —
    # controls the public exit and rendered report. Unknown-method rejection is
    # covered elsewhere (this file's setup-storage cases and test_training_cli).
    store.write_snapshot({"recipe": {"method": "integral-empirical-v1"}})
    summary: dict[str, object] = {"scientific_validation": False}
    if saved_backend is not None:
        summary["backend_identity"] = saved_backend
    store.finalize({"status": "FAILED", "summary": summary})
    before = (root / "data/sample/run.json").read_bytes()
    assert cli._saved_report(store, root, "sample", "demo") == 4
    output = result(capsys)
    assert output["backend_identity"] == (saved_backend or "UNVERIFIED")
    if saved_backend != "simon":
        assert "计数来源未核验" in str(output["count_semantics"])
    assert output["scientific_validation"] is False
    html = Path(str(output["report_path"])).read_text()
    assert "合成工程演示" not in html
    assert ("后端身份未核验" in html) is (saved_backend != "simon")
    assert (root / "data/sample/run.json").read_bytes() == before


def test_first_identity_write_failure_does_not_infer_metadata_from_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RunStore.write_snapshot

    def overflow(store: RunStore, value: Mapping[str, object]) -> None:
        original(store, {**value, "summary": {"padding": "x" * 20000001}})

    def forbidden(*a: object) -> None:
        pytest.fail("failed owner must not be refinalized")

    monkeypatch.setattr(RunStore, "write_snapshot", overflow)
    monkeypatch.setattr(RunStore, "finalize", forbidden)
    # The very first identity snapshot failed: no backend_identity was committed
    # and no recipe/method was ever saved. The CLI must not infer metadata from
    # the current --backend argument nor fabricate an integral report; it seals
    # as REPORT_METHOD_UNKNOWN with exit 6 and produces no report files.
    assert cli.main(args(tmp_path)) == 6
    output = result(capsys)
    assert output["backend_identity"] == "UNVERIFIED"
    assert output["reason"] == "REPORT_METHOD_UNKNOWN"
    assert output["report_failed"] is True
    assert "report_path" not in output
    saved = read_saved(tmp_path / "out", "sample")
    assert saved["summary"] == {"reason": "OUTPUT_LIMIT"}
    assert not (tmp_path / "out" / "result" / "sample" / "index.html").exists()
    assert not (tmp_path / "out" / "result" / "sample" / "source.json").exists()


@pytest.mark.parametrize("amount", [1, 2, 3, 65536])
def test_demo_bulk_words_preserves_continuous_eight_byte_stream(amount: int) -> None:
    from cipher_lab.demo import DemoBytes

    scalar, bulk = DemoBytes(), DemoBytes()
    assert scalar(8) == bulk(8)
    for size in (1, 10, 16):
        assert scalar(size) == bulk(size)
    assert bulk.bulk_words(amount) == b"".join(scalar(8) for _ in range(amount))
    assert scalar(8) == bulk(8)
    for size in (1, 10, 16):
        assert scalar(size) == bulk(size)


@pytest.mark.parametrize("amount", [0, -1, True, 65537, 1.0])
def test_demo_bulk_rejects_invalid_word_counts(amount: int) -> None:
    from cipher_lab.demo import DemoBytes

    with pytest.raises((TypeError, ValueError)):
        DemoBytes().bulk_words(amount)


def test_check_does_not_import_numpy_or_cipher(tmp_path: Path) -> None:
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('numpy', 'cipher_lab.ciphers', 'torch')):
            raise AssertionError('PRIVATE forbidden scientific dependency')
sys.meta_path.insert(0, Block())
from cipher_lab.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, "check", *inputs(tmp_path)], cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["planned_calls_fit_policy"]["real_blocks"] is True


@pytest.mark.parametrize("backend", ["demo", "simon"])
def test_integral_cli_explicit_batch_only_preserves_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    import numpy as np
    from numpy.typing import NDArray

    from cipher_lab.demo import DemoBytes, demo_rss_bytes

    source = DemoBytes()
    calls: list[int] = []
    def forbidden(*a: object, **k: object) -> None:
        pytest.fail("scalar cipher or production RNG/RSS used")
    def fake(points: NDArray[np.uint64], key: tuple[int, ...], rounds: int
             ) -> NDArray[np.uint64]:
        calls.append(len(points))
        assert points.dtype == np.dtype(np.uint64) and points.ndim == 1
        return points ^ np.uint64(rounds)
    fake_cipher = ModuleType("cipher_lab.ciphers.simon")
    fake_cipher.encrypt_many = forbidden  # type: ignore[attr-defined]
    fake_cipher.encrypt_batch = fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cipher_lab.ciphers.simon", fake_cipher)
    monkeypatch.setattr("cipher_lab.demo.fake_encrypt_many", forbidden)
    monkeypatch.setattr("cipher_lab.demo.fake_encrypt_batch", fake, raising=False)
    # The production assembly still uses arbitrary exact byte sizes; this fake
    # models that contract without confusing a key request with a null request.
    def random(size: int) -> bytes:
        return source(size) if size in (8, 16) else b"x"*size
    monkeypatch.setattr("secrets.token_bytes", random)
    monkeypatch.setattr(execution, "production_rss_bytes", demo_rss_bytes)
    arguments = args(tmp_path)
    arguments[-1] = backend
    assert cli.main(arguments) in (0, 5)
    output = result(capsys)
    assert calls == [3, 1]*3
    saved = read_saved(tmp_path / "out", "sample")
    counts = cast(dict[str, int], saved["counts"])
    assert counts["completed_real_blocks"] == counts["attempted_real_blocks"] == 12
    assert counts["completed_null_words"] == counts["attempted_null_words"] == 12
    assert output["backend_identity"] == backend and output["scientific_validation"] is False


@pytest.mark.parametrize("chunk", [1, 2, 3, 4])
def test_demo_cli_bulk_null_word_stream_not_key_size_stream(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], chunk: int,
) -> None:
    arguments = args(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["chunk_points"] = chunk
    config_path.write_text(json.dumps(config))
    assert cli.main(arguments) in (0, 5)
    result(capsys)
    saved = read_saved(tmp_path / "out", "sample")
    counts = cast(dict[str, int], saved["counts"])
    assert counts["completed_real_groups"] == counts["completed_null_groups"] == 3
    assert counts["completed_real_blocks"] == counts["completed_null_words"] == 12
    # Every group has the same null sequence regardless of 8/16-byte bulk size.
    payloads = cast(dict[str, dict[str, object]], saved["payloads"])
    summary = cast(dict[str, object], saved["summary"])
    cells = cast(list[dict[str, object]], summary["cells"])
    null = cast(dict[str, object], cells[0]["null"])
    assert null["frozen_dimension"] == 2079
    assert len(cast(list[int], payloads[str(null["basis_ref"])]["frozen_basis"])) == 2079


def test_production_public_exit_identity_is_not_all_family_kat_or_proof(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli._public_exit("SUCCEEDED", report_status="SAVED", backend="ciphers") == 0
    output = result(capsys)
    assert output["scientific_validation"] is False
    assert "不认证实现" in str(output["count_semantics"])
    assert "不等于所有密码家族 KAT" in str(output["count_semantics"])

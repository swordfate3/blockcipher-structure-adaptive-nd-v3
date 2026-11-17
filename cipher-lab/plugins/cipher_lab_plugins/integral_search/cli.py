"""Explicit CLI assembly; check has no scientific imports or side effects."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from cipher_lab.config import Policy, load_policy
from cipher_lab_plugins.integral_search.config import SearchRecipe, load_experiment


def _estimate(recipe: SearchRecipe) -> dict[str, object]:
    from cipher_lab_plugins.integral_search.estimate import estimate
    return estimate(recipe)


def _public_exit(status: str, *, report_status: str, backend: str,
                 reason: str | None = None, report_path: Path | None = None,
                 exit_code: int | None = None) -> int:
    report_failed = report_status == "FAILED"
    code = {"SUCCEEDED": 0, "STOPPED": 3, "FAILED": 4, "ALARM": 5}.get(status, 7)
    if report_failed and status != "UNKNOWN":
        code = 6
    if exit_code is not None:
        code = exit_code
    value: dict[str, object] = {
        "scientific_status": status, "backend_identity": backend,
        "scientific_validation": False, "report_status": report_status,
        "report_failed": report_failed, "exit_code": code,
        "count_semantics": ("模拟 blocks/words/groups；fake engineering，非实际 SIMON；RSS 模拟为 0"
                            if backend == "demo" else
                            "生产适配器身份与计数，不认证实现；"
                            "既有工程门禁不等于所有密码家族 KAT 或科学证明"
                            if backend in ("simon", "ciphers") else
                            "计数来源未核验；不认定为模拟或实际密码计数，未经科学验证"),
    }
    if reason is not None:
        value["reason"] = reason
    if report_path is not None:
        value["report_path"] = str(report_path)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return code


def _run(recipe: SearchRecipe,
         policy: Policy, root: Path,
         run_id: str, backend: str) -> int:
    # Claim precedes production imports/randomness/resource probes. Never reopen
    # an occupied owner or infer a root from historical cwd artifacts.
    from cipher_lab.artifacts import ArtifactError, RunStore
    from cipher_lab_plugins.integral_search.source import source_identity

    try:
        store = RunStore.claim(root, run_id, output_bytes=policy.output_bytes,
                               needs_models=False, source_supplier=source_identity)
    except Exception:
        return _public_exit("UNKNOWN", report_status="NOT_ATTEMPTED", backend=backend,
                            reason="CLAIM_FAILED", exit_code=2)
    try:
        # Commit minimal identity first so later setup/method storage failures
        # can retain it. A failed first write must not infer identity retroactively.
        store.write_snapshot({"summary": {
            "backend_identity": backend, "scientific_validation": False}})
        store.write_snapshot({"recipe": recipe.to_dict(), "policy": policy.to_dict()})
    except Exception:
        # Storage owns failure sealing; no finalize retry on a potentially sealed owner.
        return _saved_report(store, root, run_id, backend, reason="SETUP_SAVE_FAILED")

    stopped_type: type[Stopped] | None = None
    meter: Meter | None = None
    invoke: Callable[[], object]
    try:
        from cipher_lab.execution import Clocks, Meter, Stopped, production_rss_bytes

        stopped_type = Stopped
        if backend == "demo":
            from cipher_lab.demo import demo_rss_bytes

            rss = demo_rss_bytes
        else:
            rss = production_rss_bytes
        meter = Meter(policy, Clocks(), rss)
        from cipher_lab.data.cubes import CubeCollector
        from cipher_lab.integral.method import BackendIdentity
        from cipher_lab_plugins.integral_search.method import run_integral

        if backend == "demo":
            from cipher_lab.demo import DemoBytes, fake_encrypt_batch

            source = DemoBytes()
            collector = CubeCollector(
                meter, None, source, recipe.chunk_points, batch_oracle=fake_encrypt_batch,
                null_random=lambda size: source.bulk_words(size // 8))
        else:
            from secrets import token_bytes

            from cipher_lab.ciphers.simon import encrypt_batch

            collector = CubeCollector(meter, None, token_bytes, recipe.chunk_points,
                                      batch_oracle=encrypt_batch)
        invoke = lambda: run_integral(  # noqa: E731
            recipe, collector, meter, store, backend_identity=cast(BackendIdentity, backend))
    except Exception as error:
        # Method has not been called: preserve a public setup terminal if possible.
        is_stop = stopped_type is not None and isinstance(error, stopped_type)
        status = "STOPPED" if is_stop else "FAILED"
        engineering = (isinstance(error, ValueError)
                       and str(error) == 'ENGINEERING_LIMIT_OUTPUT')
        reason = (error.reason if stopped_type is not None and isinstance(error, stopped_type)
                  else "ENGINEERING_LIMIT" if engineering
                  else "BACKEND_SETUP_FAILED")
        setup_summary: dict[str, object] = {
            "backend_identity": backend, "scientific_validation": False, "reason": reason}
        try:
            store.finalize({"status": status, "summary": setup_summary})
        except Exception:
            pass  # possibly sealed or media failure; read_saved alone decides known status
        return _saved_report(store, root, run_id, backend, reason=reason)
    try:
        invoke()
    except ArtifactError:
        pass  # only the storage owner attempts its own failure persistence
    except Exception:
        pass  # method started: never invent/finalize a missing scientific terminal
    return _saved_report(store, root, run_id, backend)


def _saved_report(store: object, root: Path, run_id: str, backend: str,
                  *, reason: str | None = None) -> int:
    from cipher_lab.artifacts import RunStore, read_saved

    assert isinstance(store, RunStore)
    try:
        saved = read_saved(root, run_id)
    except Exception:
        return _public_exit("UNKNOWN", report_status="FAILED", backend=backend,
                            reason="SAVED_READ_FAILED")
    # Once a record is readable, its identity (including absence) is authoritative;
    # current CLI arguments must not relabel historical saved results.
    summary = cast(dict[str, object], saved["summary"])
    saved_backend = summary.get("backend_identity")
    backend = (saved_backend if isinstance(saved_backend, str)
               and saved_backend in ("demo", "simon", "ciphers", "speck-study", "UNVERIFIED")
               else "UNVERIFIED")
    status = saved.get("status")
    if status not in ("SUCCEEDED", "STOPPED", "FAILED", "ALARM"):
        return _public_exit("UNKNOWN", report_status="NOT_ATTEMPTED", backend=backend,
                            reason="MISSING_TERMINAL")
    try:
        method = summary.get("method", cast(dict[str, object], saved["recipe"]).get("method"))
        if method == "integral-search-v1":
            from cipher_lab_plugins.integral_search.report import render_report
            files = render_report(saved)
        else:
            # Missing/unknown saved evidence is not an integral method identity.
            return _public_exit(str(status), report_status="FAILED", backend=backend,
                                reason="REPORT_METHOD_UNKNOWN")
        store.write_report_files(files)
    except Exception:
        return _public_exit(str(status), report_status="FAILED", backend=backend,
                            reason="REPORT_FAILED")
    return _public_exit(str(status), report_status="SAVED", backend=backend, reason=reason,
                        report_path=root.absolute() / "result" / run_id / "index.html")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cipher-lab --plugin integral_search", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "run"):
        sub = commands.add_parser(command, allow_abbrev=False)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--policy", type=Path, required=True)
        if command == "run":
            sub.add_argument("--root", type=Path, required=True)
            sub.add_argument("--id", required=True)
            sub.add_argument("--backend", choices=("demo", "simon"), required=True)
    args = parser.parse_args(argv)
    try:
        recipe = load_experiment(args.config)
        policy = load_policy(args.policy)
    except (OSError, ValueError):
        print("cipher-lab: invalid or unreadable settings", file=sys.stderr)
        return 2
    if args.command == "run":
        return _run(recipe, policy, args.root, args.id, args.backend)
    estimate = _estimate(recipe)
    print(json.dumps({
        "device": "cpu", "recipe": recipe.to_dict(), "policy": policy.to_dict(),
        "estimate": estimate,
        "planned_calls_fit_policy": {
            key: cast(int, estimate["planned_" + key]) <= getattr(policy, key)
            for key in ("real_blocks", "null_words", "optimizer_updates", "forward_rows")},
        "limits": "Static validation only, not authorization or runtime/OS resource enforcement.",
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0

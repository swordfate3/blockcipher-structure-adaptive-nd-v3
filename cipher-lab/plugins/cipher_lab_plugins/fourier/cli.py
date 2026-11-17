"""Explicit Fourier check/run entry: no core CLI runners or scientific check imports."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from cipher_lab.config import Policy, load_policy
from cipher_lab_plugins.fourier.config import FourierRecipe, estimate_fourier, load_experiment

if TYPE_CHECKING:
    from cipher_lab.execution import Meter, Stopped


def _estimate(recipe: FourierRecipe) -> dict[str, object]:
    counts = estimate_fourier(recipe)
    return {
        "basis": ("Bounded 64-layer paired bucket search plus one coefficient panel, "
                  "one DEV panel and a conditional TEST panel on both real/null sides. "
                  "Worst-case counts include all planned labels; early beam/coef/DEV "
                  "stops and beam truncation are not a budget guarantee. check never "
                  "samples, encrypts or fits." if not recipe.synthetic else
                  "Independent tiny synthetic protocol: same bounded count formulas, "
                  "but no SIMON rounds and an artificial-oracle demo only; never a "
                  "six-round scientific result."),
        "units": {
            "real_blocks": "one complete cipher encryption of a 64-bit plaintext, "
                           "not round operations",
            "null_words": "one evaluation of the fixed public pseudorandom function "
                          "per prediction label",
            "optimizer_updates": "one optimizer step; Fourier search uses none",
            "forward_rows": "one neural model sample row; Fourier search uses none",
        },
        "planned_real_blocks": counts["max_real_blocks"],
        "planned_null_words": counts["max_null_words"],
        "planned_optimizer_updates": counts["updates"],
        "planned_forward_rows": counts["nn_forward_rows"],
        "search_real_blocks": counts["search_real_blocks"],
        "max_sample_bucket_products": counts["max_sample_bucket_products"],
        "tau": recipe.tau,
        "beam_width": recipe.beam_width,
    }


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



def _run(recipe: FourierRecipe, policy: Policy, root: Path,
         run_id: str, backend: str) -> int:
    # Claim BEFORE production imports, entropy or resource probes.
    from cipher_lab.artifacts import ArtifactError, RunStore
    from cipher_lab_plugins.fourier.source import source_identity

    try:
        store = RunStore.claim(root, run_id, output_bytes=policy.output_bytes,
                               needs_models=True, source_supplier=source_identity)
    except Exception:
        return _public_exit("UNKNOWN", report_status="NOT_ATTEMPTED", backend=backend,
                            reason="CLAIM_FAILED", exit_code=2)
    try:
        store.write_snapshot({"summary": {
            "backend_identity": backend, "scientific_validation": False}})
        store.write_snapshot({"recipe": recipe.to_dict(), "policy": policy.to_dict()})
    except Exception:
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
        from cipher_lab_plugins.fourier.data import SpectralData
        from cipher_lab_plugins.fourier.method import run_fourier

        if backend == "demo":
            from cipher_lab.demo import DemoBytes, fake_encrypt_batch

            data = SpectralData(recipe, fake_encrypt_batch, DemoBytes(), meter)
        else:
            from secrets import token_bytes

            from cipher_lab.ciphers.simon import encrypt_batch

            data = SpectralData(recipe, encrypt_batch, token_bytes, meter)
        invoke = lambda: run_fourier(  # noqa: E731
            recipe, data, meter, store, backend_identity=backend)
    except Exception as error:
        is_stop = stopped_type is not None and isinstance(error, stopped_type)
        status = "STOPPED" if is_stop else "FAILED"
        reason = (error.reason if stopped_type is not None and isinstance(error, stopped_type)
                  else "BACKEND_SETUP_FAILED")
        summary: dict[str, object] = {
            "backend_identity": backend, "scientific_validation": False, "reason": reason}
        if meter is not None:
            summary.update(meter=meter.snapshot(), method=recipe.method,
                           synthetic=backend == "demo")
        try:
            store.finalize({"status": status, "summary": summary})
        except Exception:
            pass  # saved evidence alone decides terminal status
        return _saved_report(store, root, run_id, backend, reason=reason)
    try:
        invoke()
    except ArtifactError:
        pass  # storage owner seals its own failures
    except Exception:
        pass  # no invented terminal after the method started
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
        if method == "bit-fourier-search-v1":
            from cipher_lab_plugins.fourier.report import render_spectral_report

            files = render_spectral_report(saved)
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



def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="cipher-lab --plugin cipher_lab_plugins.fourier.cli",
                                     allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "run"):
        sub = commands.add_parser(command, allow_abbrev=False)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--policy", type=Path, required=True)
        sub.add_argument("--public-plan", type=Path)
        if command == "run":
            sub.add_argument("--root", type=Path, required=True)
            sub.add_argument("--id", required=True)
            sub.add_argument("--backend", choices=("demo", "simon", "ciphers", "speck-study"),
                             required=True)
    args = parser.parse_args(argv)
    try:
        recipe = load_experiment(args.config)
        policy = load_policy(args.policy)
    except (OSError, ValueError):
        print("cipher-lab: invalid or unreadable settings", file=sys.stderr)
        return 2
    if args.public_plan is not None:
        print("cipher-lab: invalid method/plan combination", file=sys.stderr)
        return 2
    if args.command == "run":
        allowed = ("demo",) if recipe.synthetic else ("simon",)
        if args.backend not in allowed:
            print("cipher-lab: invalid method/backend combination", file=sys.stderr)
            return 2
        return _run(recipe, policy, args.root, args.id, args.backend)
    estimate = _estimate(recipe)
    print(json.dumps({
        "device": "cpu", "recipe": recipe.to_dict(), "policy": policy.to_dict(),
        "estimate": estimate,
        "planned_calls_fit_policy": {
            "real_blocks": cast(int, estimate["planned_real_blocks"]) <= policy.real_blocks,
            "null_words": cast(int, estimate["planned_null_words"]) <= policy.null_words,
            "optimizer_updates": cast(int, estimate["planned_optimizer_updates"])
            <= policy.optimizer_updates,
            "forward_rows": cast(int, estimate["planned_forward_rows"]) <= policy.forward_rows,
        },
        "limits": "Static validation only, not authorization or runtime/OS resource enforcement.",
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Explicit CLI assembly; check has no scientific imports or side effects."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from cipher_lab.config import Policy, Recipe, load_experiment, load_policy
from cipher_lab.speck_study_config import SpeckStudyRecipe, estimate_study
from cipher_lab.speck_study_launch import LaunchSettings
from cipher_lab.training_config import PredictionRecipe, SharedMRecipe


def _estimate(
    recipe: Recipe | PredictionRecipe | SharedMRecipe | SpeckStudyRecipe,
) -> dict[str, object]:
    """Static counts, not scientific execution or resource/time prediction."""
    if isinstance(recipe, SpeckStudyRecipe):
        study_counts = estimate_study(recipe)
        return dict(study_counts, planned_real_blocks=study_counts['real_blocks'],
                    planned_null_words=0, planned_optimizer_updates=0,
                    planned_forward_rows=study_counts['teacher_rows'],
                    basis='One fixed R -> conditional B study; recipe is not authorization.',
                    gpu_binding=recipe.gpu_binding, cpu_rule=recipe.cpu_rule)
    if isinstance(recipe, (PredictionRecipe, SharedMRecipe)):
        training = recipe.training
        rows = training.train + training.dev + training.test
        tasks = len(recipe.tasks) if isinstance(recipe, SharedMRecipe) else 1
        shared = isinstance(recipe, SharedMRecipe)
        real = 8 * tasks * rows if shared else (
            rows if isinstance(recipe, PredictionRecipe) and recipe.label_mode == "cipher" else 0)
        null = rows if isinstance(recipe, PredictionRecipe) and recipe.label_mode == "null" else 0
        return {
            "basis": "Full training, per-epoch DEV, and selected-model final TRAIN/DEV/TEST.",
            "units": {
                "real_blocks": "one 64-bit plaintext encryption, not round operations",
                "null_words": "one independent null draw per prediction bit label",
                "optimizer_updates": "one optimizer step (shared-M: one per task)",
                "forward_rows": "one model sample row; shared-M four pairs remain one row",
            },
            "planned_real_blocks": real,
            "planned_null_words": null,
            "planned_optimizer_updates": tasks * training.epochs * training.train
            // training.batch_size,
            "planned_forward_rows": tasks * (
                training.epochs * training.train + training.epochs * training.dev + rows),
        }
    groups = recipe.groups.discovery + recipe.groups.screen + recipe.groups.confirm
    cells = [
        {
            "rounds": cell.rounds,
            "active": list(cell.active),
            "points_per_group": 1 << len(cell.active),
            "planned_real_groups": groups,
            "planned_null_groups": groups,
            "planned_real_blocks": groups * (1 << len(cell.active)),
            "planned_null_words": groups * (1 << len(cell.active)),
        }
        for cell in recipe.cells
    ]
    words = sum(groups * (1 << len(cell.active)) for cell in recipe.cells)
    return {
        "basis": (
            "Full D/S/C plan per cell and per real/null side; "
            "actual confirmation may be skipped."
        ),
        "units": {
            "groups": "one complete cube on one side (real or null)",
            "real_blocks": "one 64-bit plaintext encryption, not round operations",
            "null_words": "one independent 64-bit random-function output per point",
            "optimizer_updates": "one optimizer step; integral uses none",
            "forward_rows": "one model sample row; integral uses none",
        },
        "planned_real_groups": len(recipe.cells) * groups,
        "planned_null_groups": len(recipe.cells) * groups,
        "planned_real_blocks": words,
        "planned_null_words": words,
        "planned_optimizer_updates": 0,
        "planned_forward_rows": 0,
        "cells": cells,
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
                            "生产适配器计数；密码/RSS 未实测，未经科学验证"
                            if backend in ("simon", "ciphers") else
                            "计数来源未核验；不认定为模拟或实际密码计数，未经科学验证"),
    }
    if reason is not None:
        value["reason"] = reason
    if report_path is not None:
        value["report_path"] = str(report_path)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return code


def _needs_models(recipe: object) -> bool:
    return isinstance(recipe, (PredictionRecipe, SharedMRecipe, SpeckStudyRecipe))


def _run(recipe: Recipe | PredictionRecipe | SharedMRecipe | SpeckStudyRecipe,
         policy: Policy, root: Path, run_id: str,
         backend: str, *, study: LaunchSettings | None = None) -> int:
    # Claim precedes production imports/randomness/resource probes. Never reopen
    # an occupied owner or infer a root from historical cwd artifacts.
    from cipher_lab.artifacts import ArtifactError, RunStore

    try:
        store = RunStore.claim(root, run_id, output_bytes=policy.output_bytes,
                               needs_models=_needs_models(recipe))
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

    if isinstance(recipe, SpeckStudyRecipe):
        application_failed = False
        try:
            from cipher_lab.speck_study.application import run_application

            if study is None:
                raise ValueError('missing launch settings')
            run_application(recipe, policy, store, root, run_id, study)
        except ArtifactError:
            application_failed = True
        except Exception:
            # Started application owns terminal; never fabricate saved success.
            application_failed = True
        code = _saved_report(store, root, run_id, backend)
        return 7 if application_failed else code

    stopped_type: type[Stopped] | None = None
    meter: Meter | None = None
    try:
        from cipher_lab.execution import Clocks, Meter, Stopped, production_rss_bytes

        stopped_type = Stopped
        if backend == "demo":
            from cipher_lab.demo import demo_rss_bytes

            rss = demo_rss_bytes
        else:
            rss = production_rss_bytes
        meter = Meter(policy, Clocks(), rss)
        if isinstance(recipe, (PredictionRecipe, SharedMRecipe)):
            from cipher_lab.training.application import prepare_training

            invoke = prepare_training(recipe, meter, store, backend)
        else:
            from cipher_lab.data.cubes import CubeCollector
            from cipher_lab.integral.method import BackendIdentity, run_integral

            if backend == "demo":
                from cipher_lab.demo import DemoBytes, fake_encrypt_many

                collector = CubeCollector(
                    meter, fake_encrypt_many, DemoBytes(), recipe.chunk_points)
            else:
                from secrets import token_bytes

                from cipher_lab.ciphers.simon import encrypt_many

                collector = CubeCollector(meter, encrypt_many, token_bytes, recipe.chunk_points)
            invoke = lambda: run_integral(  # noqa: E731
                recipe, collector, meter, store, backend_identity=cast(BackendIdentity, backend))
    except Exception as error:
        # Method has not been called: preserve a public setup terminal if possible.
        is_stop = stopped_type is not None and isinstance(error, stopped_type)
        status = "STOPPED" if is_stop else "FAILED"
        reason = (error.reason if stopped_type is not None and isinstance(error, stopped_type)
                  else "BACKEND_SETUP_FAILED")
        setup_summary: dict[str, object] = {
            "backend_identity": backend, "scientific_validation": False, "reason": reason}
        if isinstance(recipe, (PredictionRecipe, SharedMRecipe)) and meter is not None:
            setup_summary["meter"] = meter.snapshot()
            setup_summary["method"] = recipe.method
            setup_summary["synthetic"] = backend == "demo"
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


def _evaluate(args: argparse.Namespace, policy: Policy) -> int:
    from cipher_lab.artifacts import ArtifactError, RunStore

    backend = "UNVERIFIED"
    try:
        store = RunStore.claim(args.root, args.id, output_bytes=policy.output_bytes,
                               needs_models=False)
    except Exception:
        return _public_exit("UNKNOWN", report_status="NOT_ATTEMPTED", backend=backend,
                            reason="CLAIM_FAILED", exit_code=2)
    try:
        store.write_snapshot({"summary": {"method": "explicit-evaluation-v1",
            "backend_identity": backend, "synthetic": None, "scientific_validation": False}})
        store.write_snapshot({"policy": policy.to_dict(),
                              "recipe": {"method": "explicit-evaluation-v1"}})
    except Exception:
        return _saved_report(store, args.root, args.id, backend, reason="SETUP_SAVE_FAILED")
    stopped_type: type[Stopped] | None = None
    try:
        from cipher_lab.execution import Clocks, Meter, Stopped, production_rss_bytes

        stopped_type = Stopped
        # Real RSS remains fail-closed; engineering tests inject only a simulated probe.
        meter = Meter(policy, Clocks(), production_rss_bytes)
        from cipher_lab.evaluation.application import run_evaluation

        # Force scientific imports inside the same post-claim fixed setup boundary.
        from cipher_lab.evaluation.checkpoints import load_checkpoint  # noqa: F401
    except Exception as error:
        is_stop = stopped_type is not None and isinstance(error, stopped_type)
        reason = (error.reason if stopped_type is not None and isinstance(error, stopped_type)
                  else "BACKEND_SETUP_FAILED")
        try:
            store.finalize({"status": "STOPPED" if is_stop else "FAILED", "summary": {
                "method": "explicit-evaluation-v1", "backend_identity": backend,
                "synthetic": None, "scientific_validation": False, "reason": reason}})
        except Exception:
            pass
        return _saved_report(store, args.root, args.id, backend, reason=reason)
    try:
        run_evaluation(args.checkpoint, args.checkpoint_sha256, args.dataset,
                       args.dataset_sha256, meter, store)
    except ArtifactError:
        pass
    except Exception:
        pass  # no invented terminal after method starts
    return _saved_report(store, args.root, args.id, backend)


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
        if method == "explicit-evaluation-v1":
            from cipher_lab.evaluation.application import render_evaluation_report

            files = render_evaluation_report(saved)
        elif method in ("bit-prediction-v1", "shared-m-four-task-v1"):
            from cipher_lab.training.report import render_training_report

            files = render_training_report(saved)
        elif method == "integral-empirical-v1":
            from cipher_lab.integral.report import render_report

            files = render_report(saved)
        elif method in ("speck-fourier-paper-prediction-v1", "speck-fourier-synthetic-v1"):
            from cipher_lab.speck_study.report import render_report as render_study_report

            files = render_study_report(saved)
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
    parser = argparse.ArgumentParser(prog="cipher-lab", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check", help="validate settings only; no collection", allow_abbrev=False
    )
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--policy", type=Path, required=True)
    run = commands.add_parser("run", help="explicit demo or unvalidated SIMON adapter",
                              allow_abbrev=False)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--policy", type=Path, required=True)
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--id", required=True)
    run.add_argument("--backend", choices=("demo", "simon", "ciphers", "speck-study"),
                     required=True)
    for flag in ("teacher-python", "teacher-weights", "source-manifest"):
        run.add_argument("--" + flag, type=Path)
    for flag in ("weights-revision", "weights-sha256", "gpu-uuid", "gpu-pci"):
        run.add_argument("--" + flag)
    evaluate = commands.add_parser("evaluate", help="evaluate explicit saved files only",
                                   allow_abbrev=False)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--checkpoint-sha256", required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--dataset-sha256", required=True)
    evaluate.add_argument("--policy", type=Path, required=True)
    evaluate.add_argument("--root", type=Path, required=True)
    evaluate.add_argument("--id", required=True)
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        try:
            policy = load_policy(args.policy)
        except (OSError, ValueError):
            print("cipher-lab: invalid or unreadable settings", file=sys.stderr)
            return 2
        return _evaluate(args, policy)
    try:
        recipe = load_experiment(args.config)
        policy = load_policy(args.policy)
    except (OSError, ValueError):
        # Never echo input values or private exception text.
        print("cipher-lab: invalid or unreadable settings", file=sys.stderr)
        return 2
    if args.command == "run":
        allowed: tuple[str, ...]
        runtime = (args.teacher_python, args.teacher_weights, args.weights_revision,
                   args.weights_sha256, args.gpu_uuid, args.gpu_pci, args.source_manifest)
        study = None
        if isinstance(recipe, SpeckStudyRecipe):
            allowed = ("speck-study",)
            try:
                if recipe.synthetic or any(v is None for v in runtime):
                    raise ValueError('formal launch settings required')
                study = LaunchSettings(*runtime)
            except (OSError, ValueError):
                print("cipher-lab: invalid study launch settings", file=sys.stderr)
                return 2
        elif any(v is not None for v in runtime):
            print("cipher-lab: invalid method/runtime combination", file=sys.stderr)
            return 2
        else:
            allowed = (("demo", "ciphers") if isinstance(recipe, SharedMRecipe)
                       else ("demo", "simon"))
        if args.backend not in allowed:
            print("cipher-lab: invalid method/backend combination", file=sys.stderr)
            return 2
        if isinstance(recipe, SpeckStudyRecipe) and study is not None:
            import os
            if os.name == 'nt' and os.environ.get('SPECK_STUDY_NATIVE') != '1':
                from cipher_lab.speck_study.native_parent import launch_native
                args.native_argv = list(sys.argv[1:] if argv is None else argv)
                try:
                    return launch_native(args,recipe,policy,study)
                except Exception:
                    return _public_exit('UNKNOWN',report_status='NOT_ATTEMPTED',
                                        backend='speck-study',reason='ENVIRONMENT_BLOCKED')
        return _run(recipe, policy, args.root, args.id, args.backend, study=study)
    estimate = _estimate(recipe)
    print(json.dumps({
        "device": ("gpu-uuid-and-pci-required" if isinstance(recipe, SpeckStudyRecipe) else "cpu"),
        "recipe": recipe.to_dict(),
        "policy": policy.to_dict(),
        "estimate": estimate,
        "planned_calls_fit_policy": {
            "real_blocks": cast("int", estimate["planned_real_blocks"]) <= policy.real_blocks,
            "null_words": cast("int", estimate["planned_null_words"]) <= policy.null_words,
            "optimizer_updates": cast("int", estimate["planned_optimizer_updates"])
            <= policy.optimizer_updates,
            "forward_rows": cast("int", estimate["planned_forward_rows"]) <= policy.forward_rows,
        },
        "limits": "Static validation only, not authorization or runtime/OS resource enforcement.",
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

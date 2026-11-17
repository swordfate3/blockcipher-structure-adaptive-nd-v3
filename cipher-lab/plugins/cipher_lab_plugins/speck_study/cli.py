"""Explicit SPECK check/run assembly; check is stdlib-only and never claims."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from cipher_lab.config import Policy, load_policy
from cipher_lab_plugins.speck_study.config import SpeckStudyRecipe, estimate_study
from cipher_lab_plugins.speck_study.launch import LaunchSettings

PLUGIN = 'cipher_lab_plugins.speck_study.cli'


def load_experiment(path: Path) -> SpeckStudyRecipe:
    from cipher_lab.config import _load

    return SpeckStudyRecipe.parse(_load(path))


def _estimate(recipe: SpeckStudyRecipe) -> dict[str, object]:
    study_counts = estimate_study(recipe)
    return dict(study_counts, planned_real_blocks=study_counts['real_blocks'],
                planned_null_words=0, planned_optimizer_updates=0,
                planned_forward_rows=study_counts['teacher_rows'],
                basis='One fixed R -> conditional B study; recipe is not authorization.',
                gpu_binding=recipe.gpu_binding, cpu_rule=recipe.cpu_rule)


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



def _run(recipe: SpeckStudyRecipe, policy: Policy, root: Path,
         run_id: str, backend: str, *, study: LaunchSettings) -> int:
    # Claim precedes production imports/randomness/resource probes. Never reopen
    # an occupied owner or infer a root from historical cwd artifacts.
    from cipher_lab.artifacts import ArtifactError, RunStore
    from cipher_lab_plugins.speck_study.sources import source_identity

    try:
        needs_models = True
        store = RunStore.claim(root, run_id, output_bytes=policy.output_bytes,
                               needs_models=needs_models, source_supplier=source_identity)
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

    application_failed = False
    try:
        from cipher_lab_plugins.speck_study.application import run_application

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
        method = summary.get('method', cast(dict[str, object], saved['recipe']).get('method'))
        if method not in ('speck-fourier-paper-prediction-v1', 'speck-fourier-synthetic-v1'):
            return _public_exit(str(status), report_status='FAILED', backend=backend,
                                reason='REPORT_METHOD_UNKNOWN')
        from cipher_lab_plugins.speck_study.report import render_report

        files = render_report(saved)
        store.write_report_files(files)
    except Exception:
        return _public_exit(str(status), report_status="FAILED", backend=backend,
                            reason="REPORT_FAILED")
    return _public_exit(str(status), report_status="SAVED", backend=backend, reason=reason,
                        report_path=root.absolute() / "result" / run_id / "index.html")



def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='cipher-lab --plugin '+PLUGIN, allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('check', 'run'):
        sub = commands.add_parser(command, allow_abbrev=False)
        sub.add_argument('--config', type=Path, required=True)
        sub.add_argument('--policy', type=Path, required=True)
        if command == 'run':
            sub.add_argument('--root', type=Path, required=True)
            sub.add_argument('--id', required=True)
            sub.add_argument('--backend', choices=('speck-study',), required=True)
            for flag in ('teacher-python','teacher-weights','source-manifest'):
                sub.add_argument('--'+flag, type=Path)
            for flag in ('weights-revision','weights-sha256','gpu-uuid','gpu-pci'):
                sub.add_argument('--'+flag)
    args = parser.parse_args(raw)
    try:
        recipe, policy = load_experiment(args.config), load_policy(args.policy)
    except (OSError, ValueError):
        print('cipher-lab: invalid or unreadable settings', file=sys.stderr)
        return 2
    if args.command == 'run':
        try:
            runtime = (args.teacher_python,args.teacher_weights,args.weights_revision,
                       args.weights_sha256,args.gpu_uuid,args.gpu_pci,args.source_manifest)
            if recipe.synthetic or any(v is None for v in runtime):
                raise ValueError('formal launch settings required')
            study = LaunchSettings(*runtime)
        except (OSError, ValueError):
            print('cipher-lab: invalid study launch settings', file=sys.stderr)
            return 2
        import os
        if os.name == 'nt' and os.environ.get('SPECK_STUDY_NATIVE') != '1':
            from cipher_lab_plugins.speck_study.native_parent import launch_native

            args.native_argv = ['run','--plugin',PLUGIN]+raw[1:]
            try:
                return launch_native(args,recipe,policy,study)
            except Exception:
                return _public_exit('UNKNOWN',report_status='NOT_ATTEMPTED',
                                    backend='speck-study',reason='ENVIRONMENT_BLOCKED')
        return _run(recipe,policy,args.root,args.id,args.backend,study=study)
    estimate = _estimate(recipe)
    print(json.dumps({
        'device':'gpu-uuid-and-pci-required', 'recipe':recipe.to_dict(),
        'policy':policy.to_dict(), 'estimate':estimate,
        'planned_calls_fit_policy': {
            name:cast(int,estimate['planned_'+name]) <= getattr(policy,name)
            for name in ('real_blocks','null_words','optimizer_updates','forward_rows')},
        'limits':'Static validation only, not authorization or runtime/OS resource enforcement.',
    },ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

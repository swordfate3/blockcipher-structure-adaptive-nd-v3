"""Explicit CLI lifecycle shared by the three existing prediction experiments.

No registry or discovery: each owner binds its parser at its explicit entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from cipher_lab.config import Policy, _load, load_policy
from cipher_lab_plugins.feature_probe.config import FeatureProbeRecipe
from cipher_lab_plugins.frontier.config import FrontierRecipe
from cipher_lab_plugins.frontier.simon_frontier import PublicFeaturePlan
from cipher_lab_plugins.prediction_common.source import source_identity
from cipher_lab_plugins.residual.config import ResidualPredictionRecipe

_MAX_PUBLIC_PLAN_BYTES = 16 * 1024 * 1024


def _read_public_plan(path: Path, recipe: FrontierRecipe
                     ) -> tuple[str, PublicFeaturePlan]:
    """Bounded stdlib-only plan read and identity check; never builds an adapter.

    Runs before any claim or scientific import so a missing/mismatched/oversized
    plan cannot create artifacts or reach Torch/model/data code.
    """
    from cipher_lab_plugins.frontier.simon_frontier import decode_plan

    with path.open('rb') as stream:
        raw = stream.read(_MAX_PUBLIC_PLAN_BYTES + 1)
    if len(raw) > _MAX_PUBLIC_PLAN_BYTES:
        raise ValueError('public plan: exceeds 16 MiB')
    plan = decode_plan(raw)
    if plan.identity != recipe.identity:
        raise ValueError('public plan: identity mismatch')
    return hashlib.sha256(raw).hexdigest(), plan


def _estimate(recipe: ResidualPredictionRecipe | FeatureProbeRecipe | FrontierRecipe
              ) -> dict[str, object]:
    if isinstance(recipe, FrontierRecipe):
        training = recipe.training
        rows = training.train + training.dev + training.test
        cipher = recipe.label_mode == "cipher"
        return {
            "basis": ("Three fixed arms A/B/C over one private panel and one audited public "
                      "plan: full training, per-epoch DEV, and selected-model final "
                      "TRAIN/DEV/TEST. The public zero-master-key reference is a second "
                      "charged encryption of every panel point; TEST stays unreleased "
                      "until all three selected checkpoints strictly load."),
            "units": {
                "real_blocks": "one 64-bit plaintext encryption, either for a private "
                               "label or for the public zero-master-key reference, "
                               "not round operations",
                "null_words": "one independent null draw per prediction bit label",
                "optimizer_updates": "one optimizer step in any arm",
                "forward_rows": "one model sample row in any arm; references add none",
            },
            "identity": recipe.identity.to_dict(),
            "planned_real_blocks": 2 * rows if cipher else rows,
            "planned_null_words": 0 if cipher else rows,
            "planned_optimizer_updates": 3 * training.epochs * training.train
            // training.batch_size,
            "planned_forward_rows": 3 * (training.epochs * (training.train + training.dev)
                                         + rows),
        }
    if isinstance(recipe, FeatureProbeRecipe):
        training = recipe.training
        rows = training.train + training.dev + training.test
        cipher = recipe.label_mode == "cipher"
        return {
            "basis": ("Three fixed arms A/B/C over one private panel: full training, "
                      "per-epoch DEV, and selected-model final TRAIN/DEV/TEST. The public "
                      "zero-master-key reference is a second charged encryption of every "
                      "panel point; TEST stays unreleased until all three selected "
                      "checkpoints strictly load."),
            "units": {
                "real_blocks": "one 64-bit plaintext encryption, either for a private "
                               "label or for the public zero-master-key reference, "
                               "not round operations",
                "null_words": "one independent null draw per prediction bit label",
                "optimizer_updates": "one optimizer step in any arm",
                "forward_rows": "one model sample row in any arm; references add none",
            },
            "planned_real_blocks": 2 * rows if cipher else rows,
            "planned_null_words": 0 if cipher else rows,
            "planned_optimizer_updates": 3 * training.epochs * training.train
            // training.batch_size,
            "planned_forward_rows": 3 * (training.epochs * (training.train + training.dev)
                                         + rows),
        }
    if isinstance(recipe, ResidualPredictionRecipe):
        training = recipe.training
        rows = training.train + training.dev + training.test
        cipher = recipe.label_mode == "cipher"
        return {
            "basis": ("Two fixed arms (direct/residual) over one private panel: full "
                      "training, per-epoch DEV, and selected-model final TRAIN/DEV/TEST. "
                      "The public zero-master-key reference is a second charged "
                      "encryption of every panel point; TEST stays unreleased until "
                      "both selected checkpoints strictly load."),
            "units": {
                "real_blocks": "one 64-bit plaintext encryption, either for a private "
                               "label or for the public zero-master-key reference, "
                               "not round operations",
                "null_words": "one independent null draw per prediction bit label",
                "optimizer_updates": "one optimizer step in either arm",
                "forward_rows": "one model sample row in either arm; references add none",
            },
            "planned_real_blocks": 2 * rows if cipher else rows,
            "planned_null_words": 0 if cipher else rows,
            "planned_optimizer_updates": 2 * training.epochs * training.train
            // training.batch_size,
            "planned_forward_rows": 2 * (training.epochs * (training.train + training.dev)
                                         + rows),
        }
    raise ValueError('unsupported recipe')


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


def _needs_models(recipe: object) -> bool:
    return True


def _run(recipe: ResidualPredictionRecipe | FeatureProbeRecipe | FrontierRecipe,
         policy: Policy, root: Path,
         run_id: str, backend: str, *, plan: PublicFeaturePlan | None = None,
         plan_sha: str | None = None, config_ref: str | None = None,
         plan_ref: str | None = None) -> int:
    # Claim precedes production imports/randomness/resource probes. Never reopen
    # an occupied owner or infer a root from historical cwd artifacts.
    from cipher_lab.artifacts import ArtifactError, RunStore

    try:
        needs_models = _needs_models(recipe)
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
        if isinstance(recipe, FrontierRecipe):
            # Public-plan bytes SHA plus logical (name-only) config/plan references;
            # the full plan stays out of the compact config and run metadata.
            if plan is None or plan_sha is None:
                raise ValueError('MISSING_PUBLIC_PLAN')
            store.write_payload("plan-identity", {
                "config_ref": config_ref, "plan_ref": plan_ref,
                "plan_sha256": plan_sha,
                "identity": recipe.identity.to_dict()})
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
        if isinstance(recipe, (ResidualPredictionRecipe, FeatureProbeRecipe, FrontierRecipe)):
            if isinstance(recipe, FrontierRecipe):
                # The engineering output bound must reject BEFORE PredictionData
                # acquires/constructs panels; run_frontier repeats it as defence.
                from cipher_lab_plugins.frontier.training import output_bound

                if plan is None:
                    raise ValueError('MISSING_PUBLIC_PLAN')
                output_bound(recipe, plan)
                from cipher_lab_plugins.prediction_common.application import prepare_training

                invoke = prepare_training(recipe, meter, store, backend, plan=plan)
            else:
                from cipher_lab_plugins.prediction_common.application import prepare_training

                invoke = prepare_training(recipe, meter, store, backend)
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
        trained = isinstance(recipe, (ResidualPredictionRecipe, FeatureProbeRecipe, FrontierRecipe))
        if trained and meter is not None:
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


def _evaluate(args: argparse.Namespace, policy: Policy, *, method: str) -> int:
    from cipher_lab.artifacts import ArtifactError, RunStore

    backend = "UNVERIFIED"
    try:
        store = RunStore.claim(args.root, args.id, output_bytes=policy.output_bytes,
                               needs_models=False, source_supplier=source_identity)
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
        # Force scientific imports inside the same post-claim fixed setup boundary.
        from cipher_lab_plugins.prediction_common.checkpoints import load_checkpoint  # noqa: F401
        from cipher_lab_plugins.prediction_common.evaluation import run_evaluation
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
                       args.dataset_sha256, meter, store, expected_method=method)
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
            from cipher_lab_plugins.prediction_common.evaluation import render_evaluation_report

            files = render_evaluation_report(saved)
        elif method == 'bit-residual-paired-v1':
            from cipher_lab_plugins.residual.report import render_residual_report
            files = render_residual_report(saved)
        elif method == 'bit-public-feature-probe-v1':
            from cipher_lab_plugins.feature_probe.report import render_feature_probe_report
            files = render_feature_probe_report(saved)
        elif method == 'bit-public-feature-frontier-v1':
            from cipher_lab_plugins.frontier.report import render_frontier_report
            files = render_frontier_report(saved)
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


def main(argv: Sequence[str], *, method: str, parse: Callable[[object], object]) -> int:
    parser = argparse.ArgumentParser(prog='cipher-lab --plugin', allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('check', 'run'):
        sub = commands.add_parser(name, allow_abbrev=False)
        sub.add_argument('--config', type=Path, required=True)
        sub.add_argument('--policy', type=Path, required=True)
        sub.add_argument('--public-plan', type=Path)
        if name == 'run':
            sub.add_argument('--root', type=Path, required=True)
            sub.add_argument('--id', required=True)
            sub.add_argument('--backend', choices=('demo', 'simon', 'ciphers', 'speck-study'),
                             required=True)
    sub = commands.add_parser('evaluate', allow_abbrev=False)
    for flag in ('checkpoint', 'dataset', 'policy', 'root'):
        sub.add_argument('--'+flag, type=Path, required=True)
    for flag in ('checkpoint-sha256', 'dataset-sha256', 'id'):
        sub.add_argument('--'+flag, required=True)
    args = parser.parse_args(argv)
    try:
        policy = load_policy(args.policy)
        if args.command == 'evaluate':
            return _evaluate(args, policy, method=method)
        recipe = parse(_load(args.config))
        if not isinstance(recipe, (ResidualPredictionRecipe, FeatureProbeRecipe, FrontierRecipe)):
            raise ValueError('invalid recipe')
        plan = None
        plan_sha = None
        if isinstance(recipe, FrontierRecipe):
            if args.public_plan is None:
                print('cipher-lab: frontier requires --public-plan', file=sys.stderr)
                return 2
            plan_sha, plan = _read_public_plan(args.public_plan, recipe)
        elif args.public_plan is not None:
            raise ValueError('unexpected public plan')
    except (OSError, ValueError):
        print('cipher-lab: invalid or unreadable settings', file=sys.stderr)
        return 2
    if args.command == 'run':
        if args.backend not in ('demo', 'simon'):
            print('cipher-lab: invalid method/backend combination', file=sys.stderr)
            return 2
        return _run(recipe, policy, args.root, args.id, args.backend, plan=plan,
                    plan_sha=plan_sha, config_ref=args.config.name,
                    plan_ref=args.public_plan.name if args.public_plan else None)
    estimate = _estimate(recipe)
    print(json.dumps({'device': 'cpu', 'recipe': recipe.to_dict(), 'policy': policy.to_dict(),
        'estimate': estimate,
        'planned_calls_fit_policy': {kind: cast(int, estimate['planned_'+kind])
            <= getattr(policy, kind)
            for kind in ('real_blocks', 'null_words', 'optimizer_updates', 'forward_rows')},
        'limits': 'Static validation only, not authorization or runtime/OS resource enforcement.'},
        ensure_ascii=False, indent=2, allow_nan=False))
    return 0

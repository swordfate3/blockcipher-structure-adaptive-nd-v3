"""Explicit stdlib help/check; lazy run/evaluate. No discovery or auto-run."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import Recipe, estimate, load_config


def _device_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--backend', required=True, choices=('demo', 'ciphers'))
    parser.add_argument('--device', required=True, choices=('cpu', 'cuda:0'))
    parser.add_argument('--gpu-uuid', default='')
    parser.add_argument('--gpu-pci', default='')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--id', required=True)


def _finish(store: Any, root: Path, run_id: str, *, resources: Any = None) -> int:
    """Close command, never rewrite the already sealed scientific record.

    Report is saved-only CPU I/O. Same Meter wall/CPU/RSS remains active, not the
    science GPU gate (no new CUDA/telemetry calls). Pre-render evidence is saved;
    post-write counts cannot be written atomically into their own report and are
    explicitly returned in command output instead of pretending otherwise.
    """
    from cipher_lab.artifacts import read_saved
    from cipher_lab.execution import Stopped

    scientific_status, backend = 'UNKNOWN', 'UNVERIFIED'
    command_status, report_status = 'UNKNOWN', 'NOT_ATTEMPTED'
    reason: str | None = None
    written = False
    code = 7

    def check() -> None:
        if resources is not None:
            resources.meter.check()

    def snapshot() -> Any:
        return None if resources is None else resources.meter.snapshot()

    # Even if the after-finalize check stops, read the sealed record to preserve
    # authoritative science identity. No render/write is admitted after a stop.
    try:
        check()
    except Stopped as error:
        reason, command_status, report_status, code = error.reason, 'STOPPED', 'STOPPED', 3
    try:
        saved: Any = read_saved(root, run_id)
    except Exception:
        reason, report_status = 'SAVED_READ_FAILED', 'FAILED'
    else:
        status = str(saved['status'])
        scientific_status = status if status != 'RUNNING' else 'UNKNOWN'
        backend = saved['summary'].get('backend_identity', 'UNVERIFIED')
        if reason is None:
            try:
                check()  # read_saved (including SHA verification) consumed this run budget
                code = {'SUCCEEDED': 0, 'STOPPED': 3, 'FAILED': 4}.get(status, 7)
                command_status = scientific_status
                if scientific_status != 'UNKNOWN':
                    from .report import render
                    check()
                    saved = {**saved, 'report_resources': {
                        'pre_render_resources': snapshot(),
                        'scope': ('saved science counts stop at science finalize; '
                                  'command output carries post-report wall/CPU/RSS status; '
                                  'report checks are meter-only, no GPU probe')}}
                    files = render(saved)
                    check()  # after render / before report persistence
                    store.write_report_files(files)
                    written = True
                    check()
                    report_status = 'SAVED'
            except Stopped as error:
                reason, command_status, report_status, code = (
                    error.reason, 'STOPPED', 'STOPPED', 3)
            except Exception:
                reason, command_status, report_status, code = (
                    'REPORT_FAILED', 'FAILED', 'FAILED', 6)
    # Failed reads/render/writes also consume the same command clock/RSS budget.
    # Retain a first public failure separately if this final check hits a limit.
    phase_reason = reason
    try:
        check()
    except Stopped as error:
        reason, command_status, report_status, code = error.reason, 'STOPPED', 'STOPPED', 3
    print(json.dumps(dict(status=scientific_status, scientific_status=scientific_status,
        command_status=command_status, backend=backend, scientific_validation=False,
        report_status=report_status, report_files_written=written, reason=reason,
        phase_reason=phase_reason,
        command_resources=snapshot(), exit_code=code,
        counts_scope=('science counts stop at science finalize; command_resources are '
                      'report-inclusive sampled wall/CPU/RSS; '
                      'post-write evidence is stdout only'))))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='cipher-lab --plugin cipher_lab_plugins.innovation1.cli',
                                     allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check', allow_abbrev=False)
    check.add_argument('--config', type=Path, required=True)
    run = commands.add_parser('run', allow_abbrev=False)
    run.add_argument('--config', type=Path, required=True)
    _device_args(run)
    evaluation = commands.add_parser('evaluate', allow_abbrev=False)
    _device_args(evaluation)
    evaluation.add_argument('--max-forward-rows', type=int, required=True)
    evaluation.add_argument('--batch-groups', type=int, required=True)
    evaluation.add_argument('--run-root', type=Path, required=True)
    evaluation.add_argument('--run-id', required=True)
    evaluation.add_argument('--run-sha256', required=True)
    for kind in ('checkpoint', 'dataset'):
        evaluation.add_argument(f'--{kind}-ref', required=True)
        evaluation.add_argument(f'--{kind}-sha256', required=True)
    args = parser.parse_args(argv)
    recipe: Recipe | None = None
    try:
        if args.command in ('run', 'check'):
            recipe = load_config(args.config)
        if args.command != 'check':
            if ((args.backend == 'demo' and args.device != 'cpu') or
                (args.backend == 'ciphers' and (args.device != 'cuda:0'
                                               or not args.gpu_uuid or not args.gpu_pci))):
                raise ValueError
    except (ValueError, OSError):
        print('innovation1: invalid settings', file=sys.stderr)
        return 2
    if args.command == 'check':
        assert recipe is not None
        print(json.dumps(dict(recipe=recipe.to_dict(), estimate=estimate(recipe),
            limits='static only; not device proof, authorization or scientific validation'),
            ensure_ascii=False, allow_nan=False))
        return 0
    from cipher_lab.artifacts import RunStore

    from .contracts import SLOTS
    from .source import source_identity
    try:
        store = RunStore.claim(args.root, args.id, output_bytes=(
            recipe.resources.artifact_bytes if recipe else Recipe().resources.artifact_bytes),
            needs_models=args.command == 'run', model_slots=SLOTS,
            source_supplier=source_identity)
    except Exception:
        print('innovation1: CLAIM_FAILED', file=sys.stderr)
        return 2
    from .application import execute, execute_evaluation
    if args.command == 'run':
        assert recipe is not None
        execution = execute(recipe, store, backend=args.backend, device=args.device,
                gpu_uuid=args.gpu_uuid, gpu_pci=args.gpu_pci)
    else:
        execution = execute_evaluation(args, store)
    return _finish(store, args.root, args.id, resources=execution.resources)


if __name__ == '__main__':
    raise SystemExit(main())

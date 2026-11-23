"""Fixed recipe CLI; the only production run path includes the one-shot gate."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Never

from cipher_lab.artifacts import ArtifactError, RunStore, read_saved
from cipher_lab.config import Policy, _load
from cipher_lab.execution import Clocks, Meter, Stopped

from ..source import source_identity
from . import config

_ARTIFACT_REASONS = frozenset({
    'OCCUPIED', 'OUTPUT_LIMIT', 'CLAIM_FAILED', 'SOURCE_UNAVAILABLE', 'SOURCE_CHANGED',
    'WRITE_FAILED', 'READ_FAILED', 'OWNERSHIP_LOST', 'HASH_MISMATCH', 'FINALIZED',
})


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise ValueError('INVALID_ARGUMENTS')


def _fixed(path: Path) -> dict[str, object]:
    actual = _load(path)  # bounded strict JSON; rejects duplicate/non-finite fields
    expected = config.recipe()
    # Python equality alone accepts False==0 and 7.0==7, including in lists.
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('INVALID_CONFIG')
    return expected


def _policy(deadline: str, clocks: Clocks) -> Policy:
    policy = Policy(2, config.WALL_SECONDS, config.CPU_SECONDS, deadline,
                    rss_stop_bytes=config.RSS_BYTES, output_bytes=config.OUTPUT_BYTES,
                    **config.total_caps())
    end = datetime.fromisoformat(deadline)
    if end > clocks.utc() + timedelta(seconds=config.WALL_SECONDS):
        raise ValueError('INVALID_DEADLINE')
    return policy


def _fixed_artifact_reason(error: ArtifactError, default: str) -> str:
    reason = str(error)
    return reason if reason in _ARTIFACT_REASONS else default


def _emit(status: str, reason: str, exit_code: int, *, meter: Meter | None = None) -> int:
    value: dict[str, object] = {'status': status, 'reason': reason}
    # Storage failure may leave only stale durable accounting. The native log
    # can retain the latest in-process accounting without displaying a score.
    if exit_code and meter is not None:
        value['counts'] = meter.snapshot()
    print(json.dumps(value), file=sys.stderr if exit_code else sys.stdout)
    return exit_code


def _read_report(store: RunStore, root: Path, run_id: str, meter: Meter) -> None:
    meter.check()
    saved = read_saved(root, run_id)
    meter.check()
    from .report import report_files

    files = report_files(saved)
    meter.check()
    store.write_report_files(files)
    meter.check()


def _run(root: Path, run_id: str, recipe: dict[str, object], policy: Policy, clocks: Clocks) -> int:
    try:
        # Still no native probe, DLL binding, Torch or NumPy at claim time.
        from . import preflight
        from .resources import working_set_bytes

        meter = Meter(policy, clocks, working_set_bytes)
        store = RunStore.claim(root, run_id, output_bytes=policy.output_bytes,
                               needs_models=True, model_slots=config.SLOTS,
                               source_supplier=source_identity)
    except Stopped as error:
        return _emit('STOPPED', error.reason, 3)
    except ArtifactError as error:
        return _emit('FAILED', _fixed_artifact_reason(error, 'CLAIM_FAILED'), 4)
    except Exception:
        return _emit('FAILED', 'CLAIM_FAILED', 4)

    summary: dict[str, object] = {'phase': 'ADMISSION', 'retry_count': 0,
                                 'preflight': {'status': 'NOT_STARTED'}}

    def phase(stage: str) -> None:
        summary['phase'] = stage

    def record(status: str, values: dict[str, object]) -> dict[str, object]:
        return {'status': status, 'recipe': recipe, 'policy': policy.to_dict(),
                'counts': meter.snapshot(), 'summary': values}

    status, reason, code = 'FAILED', 'RUN_FAILED', 4
    preflight_peaks_attempted = science_started = final_peaks_attempted = False
    try:
        initial = record('RUNNING', summary)
        del initial['status']
        store.write_snapshot(initial)
        meter.check()
        if store.current_source_identity() != store.source_snapshot():
            raise ArtifactError('SOURCE_CHANGED')
        phase('DEVICE')
        summary['device'] = preflight.gpu_gate()
        meter.check()
        phase('RUNTIME')
        from . import engine, training
        from .data import production_panels

        summary['runtime'] = engine.configure_runtime()
        meter.check()
        phase('PREFLIGHT')
        summary['preflight'] = {'status': 'RUNNING'}
        preflight.run(meter, store, summary, make_engine=engine.make_engine)
        meter.check()
        preflight_peaks_attempted = True
        summary['cuda_peaks_after_preflight'] = preflight.gpu_peaks()
        meter.check()
        phase('SCIENCE')
        science_started = True
        score = training.run(meter, store, summary, make_engine=engine.make_engine,
                             make_panels=production_panels,
                             plan=training.Plan(config.FAMILIES, config.EPOCHS, config.BATCH),
                             phase=phase)
        phase('FINALIZE')
        if store.current_source_identity() != store.source_snapshot():
            raise ArtifactError('SOURCE_CHANGED')
        final_peaks_attempted = True
        summary['cuda_peaks'] = preflight.gpu_peaks()
        meter.check()
        counts = meter.snapshot()
        for kind, n in config.total_caps().items():
            if counts[f'attempted_{kind}'] != n or counts[f'completed_{kind}'] != n:
                raise ValueError('count mismatch')
        # Retain a no-score checkpoint of actual accounting. If the terminal
        # write fails, RunStore seals without replaying the proposed TEST.
        store.write_snapshot({'counts': counts, 'summary': summary})
        meter.check()
        store.finalize(record('SUCCEEDED', {**summary, 'phase': 'COMPLETE', 'test': score}))
        status, reason, code = 'SUCCEEDED', 'COMPLETE', 0
    except preflight.DeviceError as error:
        reason = error.reason
    except Stopped as error:
        status, reason, code = 'STOPPED', error.reason, 3
    except ArtifactError as error:
        reason = _fixed_artifact_reason(error, 'ARTIFACT_FAILED')
    except KeyboardInterrupt:
        status, reason, code = 'STOPPED', 'INTERRUPTED', 3
    except Exception:
        reason = f'{summary["phase"]}_FAILED'

    report_allowed = status != 'STOPPED' and reason != 'OUTPUT_LIMIT'
    if code:
        peak_key = None
        if science_started and not final_peaks_attempted:
            peak_key = 'cuda_peaks'
            summary[peak_key] = None
        elif 'cuda_peaks_after_preflight' in summary and not preflight_peaks_attempted:
            peak_key = 'cuda_peaks_after_preflight'
        if report_allowed:
            try:
                # Preserve the original outcome even if a later check hits a limit.
                meter.check()
                if peak_key is not None:
                    try:
                        summary[peak_key] = preflight.gpu_peaks()
                    except Stopped:
                        raise
                    except Exception:
                        pass  # best effort: leave null, never retry a failed probe
                    meter.check()
            except Stopped:
                report_allowed = False
        if summary.get('preflight') == {'status': 'RUNNING'}:
            summary['preflight'] = {'status': status}
        try:
            store.finalize(record(status, {**summary, 'reason': reason}))
        except ArtifactError as error:
            # A storage fault may have sealed the writer already. Never reopen or
            # claim a second time; last durable RUNNING means UNKNOWN.
            if str(error) == 'OUTPUT_LIMIT':
                report_allowed = False
        except Exception:
            if status != 'STOPPED':
                return _emit('FAILED', 'TERMINAL_WRITE_FAILED', 4, meter=meter)
            report_allowed = False
    if report_allowed:
        try:
            _read_report(store, root, run_id, meter)
        except Stopped as error:
            if code == 0:
                return _emit('STOPPED', error.reason, 3, meter=meter)
        except Exception:
            if code == 0:
                return _emit('FAILED', 'REPORT_FAILED', 4, meter=meter)
    return _emit(status, reason, code, meter=meter)


def main(argv: list[str]) -> int:
    parser = _Parser(prog='cube-sharedm', allow_abbrev=False)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check', allow_abbrev=False)
    run = sub.add_parser('run', allow_abbrev=False)
    for command in (check, run):
        command.add_argument('--config', type=Path, required=True)
    run.add_argument('--root', type=Path, required=True)
    run.add_argument('--run-id', required=True)
    run.add_argument('--deadline-utc', required=True)
    try:
        args = parser.parse_args(argv)
        recipe = _fixed(args.config)
        if args.command == 'check':
            print(json.dumps({'status': 'VALID', 'recipe': recipe}, sort_keys=True))
            return 0
        clocks = Clocks()
        policy = _policy(args.deadline_utc, clocks)
    except (OSError, ValueError, RecursionError):
        return _emit('FAILED', 'INVALID_ARGUMENTS_OR_CONFIG', 2)
    return _run(args.root, args.run_id, recipe, policy, clocks)

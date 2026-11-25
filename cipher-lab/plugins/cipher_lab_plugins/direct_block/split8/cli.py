"""Explicit fixed split8 plugin: stdlib check, one retained owner on run."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Never

from cipher_lab.artifacts import ArtifactError, RunStore, _token, read_saved
from cipher_lab.config import Policy, _load
from cipher_lab.execution import Clocks, Meter, Stopped

from ..source import source_identity
from . import config

_PUBLIC_ARTIFACT_ERRORS = frozenset({
    'OCCUPIED', 'OUTPUT_LIMIT', 'CLAIM_FAILED', 'SOURCE_UNAVAILABLE', 'SOURCE_CHANGED',
    'WRITE_FAILED', 'READ_FAILED', 'OWNERSHIP_LOST', 'HASH_MISMATCH', 'FINALIZED',
})


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise ValueError('INVALID_ARGUMENTS')


def _emit(status: str, reason: str, code: int, meter: Meter | None = None) -> int:
    value: dict[str, object] = {'status': status, 'reason': reason}
    if code and meter is not None:
        value['counts'] = meter.snapshot()
    print(json.dumps(value), file=sys.stderr if code else sys.stdout)
    return code


def _run(root: Path, run_id: str, recipe: dict[str, object], policy: Policy, clocks: Clocks) -> int:
    from ..cube_compare import preflight, resources

    try:
        meter = Meter(policy, clocks, resources.working_set_bytes)
        store = RunStore.claim(root, run_id, needs_models=True, model_slots=config.SLOTS,
                               output_bytes=policy.output_bytes, source_supplier=source_identity)
    except Stopped as error:
        return _emit('STOPPED', error.reason, 3)
    except ArtifactError as error:
        reason = str(error)
        return _emit('FAILED', reason if reason in _PUBLIC_ARTIFACT_ERRORS else 'CLAIM_FAILED', 4)
    except Exception:
        return _emit('FAILED', 'CLAIM_FAILED', 4)

    summary: dict[str, object] = {'phase': 'ADMISSION', 'retry_count': 0,
                                 'public_kat': 'NOT_STARTED'}

    def phase(stage: str) -> None:
        summary['phase'] = stage

    def record(values: dict[str, object]) -> dict[str, object]:
        return {'recipe': recipe, 'policy': policy.to_dict(), 'counts': meter.snapshot(),
                'summary': values}

    status, reason, code = 'FAILED', 'RUN_FAILED', 4
    commit_entered = False
    try:
        store.write_snapshot(record(summary))
        meter.check()
        if store.current_source_identity() != store.source_snapshot():
            raise ArtifactError('SOURCE_CHANGED')
        phase('DEVICE')
        summary['device'] = preflight.gpu_gate()
        meter.check()
        phase('KAT')
        from . import kat

        summary['public_kat'] = 'RUNNING'
        kat.production(meter)
        gate_counts = meter.snapshot()
        if any(gate_counts[f'{prefix}_real_blocks'] != config.KAT_BLOCKS
               for prefix in ('attempted', 'completed')):
            raise ValueError('public KAT count mismatch')
        summary['public_kat'] = 'PASSED'
        store.write_snapshot(record(summary))
        meter.check()
        phase('RUNTIME')
        from ..cube_compare import engine
        from . import data, model, training

        summary['runtime'] = engine.configure_runtime()
        meter.check()
        scores = training.run(meter, store, summary, make_panels=data.production_panels,
                              make_engine=model.make_engine, phase=phase)
        phase('FINALIZE')
        meter.check()
        summary['cuda_peaks'] = preflight.gpu_peaks()
        if store.current_source_identity() != store.source_snapshot():
            raise ArtifactError('SOURCE_CHANGED')
        store.write_snapshot(record(summary))
        meter.check()
        commit_entered = True
        store.finalize({'status': 'SUCCEEDED',
                        **record({**summary, 'phase': 'COMPLETE', 'test': scores})})
        status, reason, code = 'SUCCEEDED', 'COMPLETE', 0
    except preflight.DeviceError as error:
        reason = error.reason
    except Stopped as error:
        status, reason, code = 'STOPPED', error.reason, 3
    except ArtifactError as error:
        reason = str(error) if str(error) in _PUBLIC_ARTIFACT_ERRORS else 'ARTIFACT_FAILED'
    except KeyboardInterrupt:
        status, reason, code = 'STOPPED', 'INTERRUPTED', 3
    except Exception:
        reason = f'{summary["phase"]}_FAILED'

    if code:
        if commit_entered:
            return _emit(status, reason, code, meter)
        if summary['public_kat'] == 'RUNNING':
            summary['public_kat'] = status
        try:
            store.finalize({'status': status, **record({**summary, 'reason': reason})})
        except ArtifactError as error:
            # Already sealed by a core artifact fault, or durable state unknown.
            # Do not reopen, claim again, or replay a proposed complete score.
            if str(error) != 'FINALIZED':
                return _emit(status, reason, code, meter)
        except Exception:
            return _emit(status, reason, code, meter)
    if status != 'STOPPED' and reason != 'OUTPUT_LIMIT':
        try:
            meter.check()
            saved = read_saved(root, run_id)
            meter.check()
            from .report import report_files

            files = report_files(saved)
            meter.check()
            store.write_report_files(files)
            meter.check()
        except Stopped as error:
            if not code:
                return _emit('STOPPED', error.reason, 3, meter)
        except KeyboardInterrupt:
            if not code:
                return _emit('STOPPED', 'INTERRUPTED', 3, meter)
        except Exception:
            if not code:
                return _emit('FAILED', 'REPORT_FAILED', 4, meter)
    return _emit(status, reason, code, meter)


def main(argv: list[str]) -> int:
    parser = _Parser(prog='cipher-lab split8', allow_abbrev=False)
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
        recipe = config.recipe()
        if json.dumps(_load(args.config), sort_keys=True) != json.dumps(recipe, sort_keys=True):
            raise ValueError('INVALID_CONFIG')
        if args.command == 'check':
            print(json.dumps({'status': 'VALID', 'recipe': recipe}, sort_keys=True))
            return 0
        _token(args.run_id)
        clocks = Clocks()
        policy = Policy(2, config.WALL_SECONDS, config.CPU_SECONDS, args.deadline_utc,
                        rss_stop_bytes=config.RSS_BYTES, output_bytes=config.OUTPUT_BYTES,
                        **config.CAPS)
        if datetime.fromisoformat(policy.deadline_utc) > clocks.utc() + timedelta(
            seconds=config.WALL_SECONDS
        ):
            raise ValueError('INVALID_DEADLINE')
    except (OSError, ValueError, RecursionError):
        return _emit('FAILED', 'INVALID_ARGUMENTS_OR_CONFIG', 2)
    return _run(args.root, args.run_id, recipe, policy, clocks)

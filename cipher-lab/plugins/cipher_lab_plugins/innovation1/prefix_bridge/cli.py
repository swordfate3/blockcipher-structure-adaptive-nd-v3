"""Explicit core plugin, with a zero-side-effect stdlib check path."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Never

from cipher_lab.artifacts import ArtifactError, RunStore, _token, read_saved
from cipher_lab.config import Policy, _load
from cipher_lab.execution import Clocks, Meter, Stopped

from . import config

_ARTIFACT_REASONS = frozenset({
    'OCCUPIED', 'OUTPUT_LIMIT', 'CLAIM_FAILED', 'SOURCE_UNAVAILABLE', 'SOURCE_CHANGED',
    'WRITE_FAILED', 'READ_FAILED', 'OWNERSHIP_LOST', 'HASH_MISMATCH', 'FINALIZED',
})


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise ValueError('INVALID_ARGUMENTS')


def _emit(status: str, reason: str, code: int, meter: Meter | None = None) -> int:
    record: dict[str, object] = {'status': status, 'reason': reason}
    if code and meter is not None:
        record['counts'] = meter.snapshot()
    print(json.dumps(record), file=sys.stderr if code else sys.stdout)
    return code


def _artifact_reason(error: ArtifactError, fallback: str) -> str:
    value = str(error)
    return value if value in _ARTIFACT_REASONS else fallback


def _device() -> dict[str, object]:
    import os

    if os.environ.get('CUDA_VISIBLE_DEVICES') != config.GPU_UUID:
        raise ValueError('GPU_IDENTITY')
    import torch

    from cipher_lab_plugins.direct_block.cube_compare.resources import device_identity

    count = torch.cuda.device_count()
    if type(count) is not int or count != 1:
        raise ValueError('GPU_VISIBLE_COUNT')
    return {**device_identity(), 'visible_count': count}


def _run_owned(root: Path, run_id: str, recipe: dict[str, object], policy: Policy, clocks: Clocks,
               *, slots: tuple[str, ...], source_identity: Callable[[], dict[str, str]],
               device: Callable[[], dict[str, object]],
               execute: Callable[[Meter, RunStore, dict[str, object]], None],
               report: Callable[[dict[str, object]], dict[str, bytes]]) -> int:
    """Concrete shared admission and terminal/report boundary for the two PRESENT owners."""
    from cipher_lab_plugins.direct_block.cube_compare.resources import working_set_bytes

    try:
        meter = Meter(policy, clocks, working_set_bytes)
        store = RunStore.claim(root, run_id, output_bytes=policy.output_bytes, needs_models=True,
                               model_slots=slots, source_supplier=source_identity)
    except Stopped as error:
        return _emit('STOPPED', error.reason, 3)
    except ArtifactError as error:
        return _emit('FAILED', _artifact_reason(error, 'CLAIM_FAILED'), 4)
    except Exception:
        return _emit('FAILED', 'CLAIM_FAILED', 4)

    summary: dict[str, object] = {'phase': 'ADMISSION', 'retry_count': 0}
    status, reason, code = 'FAILED', 'RUN_FAILED', 4
    try:
        store.write_snapshot({'recipe': recipe, 'policy': policy.to_dict(),
                              'counts': meter.snapshot(), 'summary': summary})
        meter.check()
        if store.current_source_identity() != store.source_snapshot():
            raise ArtifactError('SOURCE_CHANGED')
        summary['phase'] = 'DEVICE'
        summary['device'] = device()
        meter.check()
        summary['phase'] = 'RUNTIME'
        execute(meter, store, summary)
    except Stopped as error:
        status, reason, code = 'STOPPED', error.reason, 3
    except ArtifactError as error:
        reason = _artifact_reason(error, 'ARTIFACT_FAILED')
    except KeyboardInterrupt:
        status, reason, code = 'STOPPED', 'INTERRUPTED', 3
    except Exception:
        reason = f'{summary["phase"]}_FAILED'
    else:
        # The scientific terminal was committed by training. Report errors may
        # change native exit, never that saved terminal or its complete scores.
        try:
            meter.check()
            files = report(read_saved(root, run_id))
            meter.check()
            store.write_report_files(files)
            meter.check()
        except Stopped as error:
            return _emit('STOPPED', error.reason, 3, meter)
        except KeyboardInterrupt:
            return _emit('STOPPED', 'INTERRUPTED', 3, meter)
        except Exception:
            return _emit('FAILED', 'REPORT_FAILED', 4, meter)
        return _emit('SUCCEEDED', 'COMPLETE', 0)

    try:
        store.finalize({'status': status, 'counts': meter.snapshot(),
                        'summary': {**summary, 'reason': reason}})
    except (ArtifactError, OSError):
        # An I/O error may already have sealed the writer. No retry or reopening;
        # retain the nonzero result and most recent in-process counts on stderr.
        pass
    return _emit(status, reason, code, meter)


def _run(root: Path, run_id: str, recipe: dict[str, object], policy: Policy, clocks: Clocks) -> int:
    from .source import source_identity

    def execute(meter: Meter, store: RunStore, summary: dict[str, object]) -> None:
        from cipher_lab_plugins.direct_block.cube_compare.preflight import gpu_peaks

        from . import kat, training
        from .data import production_panels
        from .runtime import FitRuntime, configure

        summary['runtime'] = configure()
        source = store.source_snapshot()

        def make(slot: str, arm: dict[str, object], train: bool) -> FitRuntime:
            return FitRuntime(slot, arm, train, source=source, run_id=run_id)

        training.run(meter, store, summary, gate=kat.production, make=make,
                     make_panels=production_panels, peaks=gpu_peaks)

    def report(saved: dict[str, object]) -> dict[str, bytes]:
        from .report import report_files

        return report_files(saved)

    return _run_owned(root, run_id, recipe, policy, clocks, slots=config.SLOTS,
                      source_identity=source_identity, device=_device,
                      execute=execute, report=report)


def _main(argv: list[str], *, prog: str, recipe: dict[str, object], wall: int, cpu: int,
          rss: int, output: int, caps: config.Counts, clocks_factory: Callable[[], Clocks],
          runner: Callable[[Path, str, dict[str, object], Policy, Clocks], int]) -> int:
    parser = _Parser(prog=prog, allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check', allow_abbrev=False)
    run = commands.add_parser('run', allow_abbrev=False)
    for command in (check, run):
        command.add_argument('--config', type=Path, required=True)
    run.add_argument('--root', type=Path, required=True)
    run.add_argument('--run-id', required=True)
    run.add_argument('--deadline-utc', required=True)
    try:
        args = parser.parse_args(argv)
        if json.dumps(_load(args.config), sort_keys=True) != json.dumps(recipe, sort_keys=True):
            raise ValueError('INVALID_CONFIG')
        if args.command == 'check':
            print(json.dumps({'status': 'VALID', 'recipe': recipe}, sort_keys=True))
            return 0
        _token(args.run_id)
        clocks = clocks_factory()
        policy = Policy(2, wall, cpu, args.deadline_utc, rss_stop_bytes=rss, output_bytes=output,
                        **caps)
        if datetime.fromisoformat(policy.deadline_utc) > (
            clocks.utc() + timedelta(seconds=wall)
        ):
            raise ValueError('INVALID_DEADLINE')
    except (OSError, ValueError, RecursionError):
        return _emit('FAILED', 'INVALID_ARGUMENTS_OR_CONFIG', 2)
    return runner(args.root, args.run_id, recipe, policy, clocks)


def main(argv: list[str]) -> int:
    return _main(argv, prog='present-prefix2-bridge', recipe=config.recipe(),
                 wall=config.WALL_SECONDS, cpu=config.CPU_SECONDS, rss=config.RSS_BYTES,
                 output=config.OUTPUT_BYTES, caps=config.CAPS, clocks_factory=Clocks, runner=_run)

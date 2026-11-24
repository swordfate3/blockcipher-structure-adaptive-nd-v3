"""Fixed recipe CLI with common device admission and explicit panel modes."""
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

_ARTIFACT_REASONS = frozenset({
    'OCCUPIED', 'OUTPUT_LIMIT', 'CLAIM_FAILED', 'SOURCE_UNAVAILABLE', 'SOURCE_CHANGED',
    'WRITE_FAILED', 'READ_FAILED', 'OWNERSHIP_LOST', 'HASH_MISMATCH', 'FINALIZED',
})
_REUSE_ARTIFACT_REASONS = frozenset({'INVALID_SOURCE_RUN', 'SOURCE_SPLIT_OVERLAP', 'UNSAFE_PATH'})


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise ValueError('INVALID_ARGUMENTS')


def _fixed(path: Path) -> dict[str, object]:
    actual = _load(path)  # bounded strict JSON; rejects duplicate/non-finite fields
    if not isinstance(actual, dict):
        raise ValueError('INVALID_CONFIG')
    match actual.get('method'):
        case config.METHOD:
            expected = config.recipe()
        case config.REUSE_METHOD:
            expected = config.reuse_recipe()
        case config.LSTM_METHOD:
            expected = config.lstm_recipe()
        case config.LOW_METHOD:
            expected = config.low_recipe()
        case _:
            raise ValueError('INVALID_CONFIG')
    # Python equality alone accepts False==0 and 7.0==7, including in lists.
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('INVALID_CONFIG')
    return expected


def _policy(deadline: str, clocks: Clocks, *, reuse: bool = False,
            method: str | None = None) -> Policy:
    match method if method is not None else (config.REUSE_METHOD if reuse else config.METHOD):
        case config.METHOD:
            wall, cpu, caps = config.WALL_SECONDS, config.CPU_SECONDS, config.total_caps()
        case config.REUSE_METHOD:
            wall, cpu = config.REUSE_WALL_SECONDS, config.REUSE_CPU_SECONDS
            caps = config.reuse_counts(config.REUSE_EPOCHS, config.REUSE_EPOCHS)
        case config.LSTM_METHOD:
            wall, cpu = config.REUSE_WALL_SECONDS, config.REUSE_CPU_SECONDS
            caps = config.lstm_counts(config.REUSE_EPOCHS)
        case config.LOW_METHOD:
            wall, cpu = config.REUSE_WALL_SECONDS, config.REUSE_CPU_SECONDS
            caps = config.low_counts((config.REUSE_EPOCHS,) * 3)
        case _:
            raise ValueError('INVALID_CONFIG')
    policy = Policy(2, wall, cpu, deadline, rss_stop_bytes=config.RSS_BYTES,
                    output_bytes=config.OUTPUT_BYTES, **caps)
    end = datetime.fromisoformat(deadline)
    if end > clocks.utc() + timedelta(seconds=wall):
        raise ValueError('INVALID_DEADLINE')
    return policy


def _source_output_overlap(root: Path, run_id: str, source_root: Path) -> bool:
    # Compare the output root as well as every directory that claim would own.
    # A shared base is fine; its distinct run-id siblings are not descendants.
    _token(run_id)
    kinds = ('data', 'models', 'result')
    source_dirs = [source_root / kind / config.SOURCE_RUN_ID for kind in kinds]
    outputs = [root, *(root / kind / run_id for kind in kinds)]
    # Check requested paths and filesystem aliases, without opening any artifacts.
    for normalize in (Path.absolute, Path.resolve):
        protected = [normalize(path) for path in source_dirs]
        if any(path.is_relative_to(old) for path in map(normalize, outputs) for old in protected):
            return True
    return False


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


def _run(root: Path, run_id: str, recipe: dict[str, object], policy: Policy, clocks: Clocks,
         source_root: Path | None = None) -> int:
    single = recipe['method'] == config.LSTM_METHOD
    low = recipe['method'] == config.LOW_METHOD
    try:
        # Still no native probe, DLL binding, Torch or NumPy at claim time.
        from . import preflight
        from .resources import working_set_bytes

        meter = Meter(policy, clocks, working_set_bytes)
        store = RunStore.claim(root, run_id, output_bytes=policy.output_bytes,
                               needs_models=True,
                               model_slots=(config.LOW_SLOTS if low else
                                            (config.LSTM_SLOT,) if single else config.SLOTS),
                               source_supplier=source_identity)
    except Stopped as error:
        return _emit('STOPPED', error.reason, 3)
    except ArtifactError as error:
        return _emit('FAILED', _fixed_artifact_reason(error, 'CLAIM_FAILED'), 4)
    except Exception:
        return _emit('FAILED', 'CLAIM_FAILED', 4)

    summary: dict[str, object] = {'phase': 'ADMISSION', 'retry_count': 0,
                                 'preflight': {'status': 'NOT_STARTED'}}
    if low:
        summary.update(mode='plaintext-only-source', topology='lstm-r1r3',
                       preflight={'status': 'SKIPPED_FIXED_MATRIX'})
    elif source_root is not None:
        summary.update(mode='saved-panels', preflight={'status': 'SKIPPED_REUSE'})
    if single:
        summary['topology'] = 'lstm-single'

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
        from collections.abc import Callable

        from . import engine, training

        summary['runtime'] = engine.configure_runtime()
        meter.check()
        make_panels: Callable[[Meter], training.Panels] | None = None
        make_low_panels: Callable[[Meter], training.LowPanels] | None = None
        if source_root is None:
            from .data import production_panels

            make_panels = production_panels
            plan = training.Plan(config.FAMILIES, config.EPOCHS, config.BATCH)
            phase('PREFLIGHT')
            summary['preflight'] = {'status': 'RUNNING'}
            preflight.run(meter, store, summary, make_engine=engine.make_engine)
            meter.check()
            preflight_peaks_attempted = True
            summary['cuda_peaks_after_preflight'] = preflight.gpu_peaks()
            meter.check()
        else:
            from cipher_lab.training.early_stopping import Settings

            from .reuse import SavedPanels

            def saved_panels(panel_meter: Meter) -> SavedPanels:
                panels = SavedPanels(source_root, panel_meter)
                summary['source'] = panels.provenance
                return panels

            if low:
                from .low_rounds import LowRounds, production_low_rounds

                def low_panels(panel_meter: Meter) -> LowRounds:
                    panels = production_low_rounds(source_root, panel_meter)
                    summary['source'] = config.source_recipe()
                    return panels

                make_low_panels = low_panels
            else:
                make_panels = saved_panels
            plan = training.Plan(
                config.FAMILIES, config.REUSE_EPOCHS, config.BATCH,
                Settings(config.WARMUP_EPOCHS, config.PATIENCE, config.MIN_DELTA),
                topology=('lstm-r1r3' if low else 'lstm-single' if single else 'dual'),
            )
        phase('SCIENCE')
        science_started = True
        if low:
            score = training.run(meter, store, summary, make_low_panels=make_low_panels,
                                 make_low_engine=engine.make_low_engine, plan=plan, phase=phase)
        else:
            score = training.run(meter, store, summary,
                                 make_engine=None if single else engine.make_engine,
                                 make_lstm_engine=engine.make_lstm_engine if single else None,
                                 make_panels=make_panels, plan=plan, phase=phase)
        phase('FINALIZE')
        if store.current_source_identity() != store.source_snapshot():
            raise ArtifactError('SOURCE_CHANGED')
        final_peaks_attempted = True
        summary['cuda_peaks'] = preflight.gpu_peaks()
        meter.check()
        counts = meter.snapshot()
        if low:
            expected = config.low_counts(training.completed_low_epochs(summary, plan))
        elif single:
            expected = config.lstm_counts(training.completed_lstm_epochs(summary, plan))
        elif source_root is not None:
            expected = config.reuse_counts(*training.completed_epochs(summary, plan))
        else:
            expected = config.total_caps()
        for kind, n in expected.items():
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
        reason = (str(error) if source_root is not None and str(error) in _REUSE_ARTIFACT_REASONS
                  else _fixed_artifact_reason(error, 'ARTIFACT_FAILED'))
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
    run.add_argument('--source-root', type=Path)
    run.add_argument('--run-id', required=True)
    run.add_argument('--deadline-utc', required=True)
    try:
        args = parser.parse_args(argv)
        recipe = _fixed(args.config)
        if args.command == 'check':
            print(json.dumps({'status': 'VALID', 'recipe': recipe}, sort_keys=True))
            return 0
        method = recipe['method']
        assert isinstance(method, str)  # _fixed accepts only the named recipes
        reuse = method in (config.REUSE_METHOD, config.LSTM_METHOD, config.LOW_METHOD)
        if reuse != (args.source_root is not None):
            raise ValueError('INVALID_SOURCE_ROOT')
        clocks = Clocks()
        policy = _policy(args.deadline_utc, clocks, method=method)
    except (OSError, ValueError, RecursionError):
        return _emit('FAILED', 'INVALID_ARGUMENTS_OR_CONFIG', 2)
    if args.source_root is not None:
        try:
            if _source_output_overlap(args.root, args.run_id, args.source_root):
                return _emit('FAILED', 'SOURCE_OUTPUT_OVERLAP', 2)
        except (OSError, ValueError, RuntimeError):
            return _emit('FAILED', 'INVALID_ARGUMENTS_OR_CONFIG', 2)
    return _run(args.root, args.run_id, recipe, policy, clocks, args.source_root)

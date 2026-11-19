"""Stdlib check/help; one fixed owned run with separate saved/command status."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import FIXED, RUN_ID, SLOTS, estimate, load_config, make_policy

# A fixed run's canonical metadata carries summaries/refs, not panels or logits.
_MAX_RUN_RECORD = 2**20


def _owner_record(store: Any, root: Path) -> tuple[dict[str, Any], str]:
    from cipher_lab.artifacts import ArtifactError, _decode, _validate

    from .source import no_reparse

    path = no_reparse(root / 'data' / RUN_ID / 'run.json')
    # RunStore has no public accessor for the frozen digest. Use only the live
    # original owner's seal, never a SHA supplied by disk or an external caller.
    frozen = store._frozen_sha
    if not store._sealed or frozen is None or path != store._data / 'run.json':
        raise ArtifactError('OWNER_RECORD_UNVERIFIED')
    with path.open('rb') as stream:
        raw = stream.read(_MAX_RUN_RECORD + 1)
    if not 0 < len(raw) <= _MAX_RUN_RECORD or sha256(raw).hexdigest() != frozen:
        raise ArtifactError('OWNER_RECORD_UNVERIFIED')
    saved = _validate(_decode(raw), RUN_ID)
    if saved['status'] not in ('SUCCEEDED', 'STOPPED', 'FAILED'):
        raise ArtifactError('OWNER_RECORD_UNVERIFIED')
    return saved, frozen


def _verified_artifacts(root: Path, record: dict[str, Any]) -> None:
    from cipher_lab.artifacts import _decode, _location, _stream

    from .source import no_reparse

    # Verify the refs from the bounded, owner-confirmed record. Do not reopen
    # run.json through read_saved or decode datasets/checkpoints for a report.
    for ref, meta in record['artifacts'].items():
        path = no_reparse(_location(root, RUN_ID, ref))
        payload = _stream(path, meta, collect=meta['kind'] == 'payload')
        if meta['kind'] == 'payload':
            _decode(payload)


def _finish(store: Any, root: Path, meter: Any) -> tuple[int, str]:
    from cipher_lab.execution import Stopped

    from .report import render

    scientific, command, report, code = 'UNKNOWN', 'UNKNOWN', 'NOT_ATTEMPTED', 7
    reason, scientific_reason, written = None, None, False
    integrity = 'NOT_CHECKED'
    try:
        meter.check()
    except Stopped as error:
        command, code, reason = 'STOPPED', 3, error.reason
    try:
        record, frozen = _owner_record(store, root)
    except Exception:
        if reason is None:
            reason = 'OWNER_RECORD_UNVERIFIED'
    else:
        scientific = record['status']
        scientific_reason = record['summary'].get('reason')
        try:
            _verified_artifacts(root, record)
        except Exception:
            # A damaged best can be exactly why the original owner sealed FAILED.
            # Keep that terminal distinct from an untrusted/replaced run.json.
            try:
                _owner_record(store, root)
            except Exception:
                scientific, scientific_reason = 'UNKNOWN', None
                if reason is None:
                    reason = 'OWNER_RECORD_UNVERIFIED'
            else:
                integrity = 'FAILED'
                if reason is None:
                    command, code, reason = 'FAILED', 6, 'SAVED_ARTIFACT_INTEGRITY_FAILED'
        else:
            integrity = 'VERIFIED'
            if reason is None:
                command = scientific
                code = {'SUCCEEDED': 0, 'STOPPED': 3, 'FAILED': 4}[scientific]
                try:
                    meter.check()
                    files = render({**record, 'run_ref': f'data/{RUN_ID}/run.json',
                                    'run_sha256': frozen})
                    meter.check()
                    store.write_report_files(files)
                    written = True
                    meter.check()
                    report = 'SAVED'
                except Stopped as error:
                    command, report, code, reason = 'STOPPED', 'STOPPED', 3, error.reason
                except Exception:
                    command, report, code, reason = 'FAILED', 'FAILED', 6, 'REPORT_FAILED'
    result = dict(run_id=RUN_ID, scientific_status=scientific, command_status=command,
                  scientific_reason=scientific_reason, artifact_integrity=integrity,
                  report_status=report, report_files_written=written, reason=reason,
                  exit_code=code, exit_code_scope='REQUESTED_NOT_NATIVE',
                  command_resources=meter.snapshot(),
                  command_resources_scope='PRE_STDOUT; excludes stdout/flush/native exit')
    # Charge report/read/initial serialization without refreshing CUDA proof after science.
    json.dumps(result, allow_nan=False)
    try:
        meter.check()
    except Stopped as error:
        code = 3
        result.update(command_status='STOPPED', exit_code=code, closeout_reason=error.reason)
    result['command_resources'] = meter.snapshot()
    return code, json.dumps(result, allow_nan=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='cipher-lab --plugin cipher_lab_plugins.innovation1.single_train.cli',
        description='Fixed SKINNY single-pair M/D fresh training; no resume or fallback.',
        allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check', allow_abbrev=False)
    run = commands.add_parser('run', allow_abbrev=False)
    for command in (check, run):
        command.add_argument('--config', required=True, type=Path)
    run.add_argument('--root', required=True, type=Path, help=r'absolute output root under G:\lxy')
    args = parser.parse_args(argv)
    try:
        recipe = load_config(args.config)
    except (OSError, ValueError):
        print('single_train: CONFIG_EXACT_REQUIRED', file=sys.stderr)
        return 2
    if args.command == 'check':
        print(json.dumps(dict(recipe=recipe.to_dict(), estimate=estimate(recipe),
                              scope='STATIC_ONLY_NO_DEVICE_ADMISSION'), allow_nan=False))
        return 0
    if (not sys.dont_write_bytecode or sys.platform != 'win32'
            or not args.root.is_absolute() or not args.root.is_relative_to(Path(r'G:\lxy'))
            or any(part.endswith(('.', ' ')) or ':' in part for part in args.root.parts[1:])):
        print(r'single_train: Windows, python -B, and absolute --root under G:\lxy required',
              file=sys.stderr)
        return 2
    from cipher_lab.artifacts import RunStore
    from cipher_lab.execution import Clocks, Meter, Stopped

    from ..resources import Resources, production_rss_bytes
    from .application import execute
    from .source import no_reparse, source_identity

    clocks = Clocks()
    try:
        resources = Resources(Meter(make_policy(FIXED, clocks), clocks, production_rss_bytes))
        resources.check()
        root = no_reparse(args.root)
        for kind in ('data', 'models', 'result'):
            no_reparse(root / kind / RUN_ID)
        store = RunStore.claim(root, RUN_ID, output_bytes=recipe.resources.artifact_bytes,
                               needs_models=True, model_slots=SLOTS,
                               source_supplier=source_identity)
    except Stopped as error:
        print(json.dumps(dict(command_status='STOPPED', scientific_status='UNKNOWN',
                              reason=error.reason, exit_code=3,
                              exit_code_scope='REQUESTED_NOT_NATIVE')), flush=True)
        return 3
    except Exception:
        print('single_train: CLAIM_FAILED', file=sys.stderr)
        return 2
    execute(store, resources)
    code, message = _finish(store, root, resources.meter)
    print(message, flush=True)
    return code

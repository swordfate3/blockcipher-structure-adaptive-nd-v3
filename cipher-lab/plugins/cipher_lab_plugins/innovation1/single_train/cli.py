"""Stdlib check/help; one fixed owned run with separate saved/command status."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import RUN_ID, SLOTS, SingleRecipe, estimate, load_config, make_policy

if TYPE_CHECKING:
    from cipher_lab.artifacts import RunStore

    from ..resources import Resources

Render = Callable[[dict[str, Any]], dict[str, bytes]]
Source = Callable[[], dict[str, str]]
Execute = Callable[["RunStore", "Resources"], None]

# A fixed run's canonical metadata carries summaries/refs, not panels or logits.
_MAX_RUN_RECORD = 2**20


def _owner_record(store: Any, root: Path, run_id: str = RUN_ID) -> tuple[dict[str, Any], str]:
    from cipher_lab.artifacts import ArtifactError, _decode, _validate

    from .source import no_reparse

    path = no_reparse(root / 'data' / run_id / 'run.json')
    # RunStore has no public accessor for the frozen digest. Use only the live
    # original owner's seal, never a SHA supplied by disk or an external caller.
    frozen = store._frozen_sha
    if not store._sealed or frozen is None or path != store._data / 'run.json':
        raise ArtifactError('OWNER_RECORD_UNVERIFIED')
    with path.open('rb') as stream:
        raw = stream.read(_MAX_RUN_RECORD + 1)
    if not 0 < len(raw) <= _MAX_RUN_RECORD or sha256(raw).hexdigest() != frozen:
        raise ArtifactError('OWNER_RECORD_UNVERIFIED')
    saved = _validate(_decode(raw), run_id)
    if saved['status'] not in ('SUCCEEDED', 'STOPPED', 'FAILED'):
        raise ArtifactError('OWNER_RECORD_UNVERIFIED')
    return saved, frozen


def _verified_artifacts(root: Path, record: dict[str, Any], run_id: str = RUN_ID) -> None:
    from cipher_lab.artifacts import _decode, _location, _stream

    from .source import no_reparse

    # Verify the refs from the bounded, owner-confirmed record. Do not reopen
    # run.json through read_saved or decode datasets/checkpoints for a report.
    for ref, meta in record['artifacts'].items():
        path = no_reparse(_location(root, run_id, ref))
        payload = _stream(path, meta, collect=meta['kind'] == 'payload')
        if meta['kind'] == 'payload':
            _decode(payload)


def _finish(store: Any, root: Path, meter: Any) -> tuple[int, str]:
    from .report import render

    return finish_run(store, root, meter, RUN_ID, render)


def finish_run(store: Any, root: Path, meter: Any, run_id: str, render: Render) -> tuple[int, str]:
    from cipher_lab.execution import Stopped

    scientific, command, report, code = 'UNKNOWN', 'UNKNOWN', 'NOT_ATTEMPTED', 7
    reason, scientific_reason, written = None, None, False
    integrity = 'NOT_CHECKED'
    try:
        meter.check()
    except Stopped as error:
        command, code, reason = 'STOPPED', 3, error.reason
    try:
        record, frozen = _owner_record(store, root, run_id)
    except Exception:
        if reason is None:
            reason = 'OWNER_RECORD_UNVERIFIED'
    else:
        scientific = record['status']
        scientific_reason = record['summary'].get('reason')
        try:
            _verified_artifacts(root, record, run_id)
        except Exception:
            # A damaged best can be exactly why the original owner sealed FAILED.
            # Keep that terminal distinct from an untrusted/replaced run.json.
            try:
                _owner_record(store, root, run_id)
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
                    files = render({**record, 'run_ref': f'data/{run_id}/run.json',
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
    result = dict(run_id=run_id, scientific_status=scientific, command_status=command,
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
    def wiring(_recipe: SingleRecipe) -> tuple[Execute, Source, Render]:
        from .application import execute
        from .report import render
        from .source import source_identity

        return execute, source_identity, render

    return fixed_cli(argv, name='single_train',
                     description='Fixed SKINNY single-pair M/D fresh training; '
                                 'no resume or fallback.',
                     load=load_config, run_id=lambda _: RUN_ID, slots=SLOTS, wiring=wiring)


def fixed_cli[RecipeT: SingleRecipe](
        argv: Sequence[str] | None, *, name: str, description: str,
        load: Callable[[Path], RecipeT], run_id: Callable[[RecipeT], str],
        slots: tuple[str, str], wiring: Callable[[RecipeT], tuple[Execute, Source, Render]]) -> int:
    parser = argparse.ArgumentParser(
        prog=f'cipher-lab --plugin cipher_lab_plugins.innovation1.{name}.cli',
        description=description,
        allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check', allow_abbrev=False)
    run = commands.add_parser('run', allow_abbrev=False)
    for command in (check, run):
        command.add_argument('--config', required=True, type=Path)
    run.add_argument('--root', required=True, type=Path, help=r'absolute output root under G:\lxy')
    args = parser.parse_args(argv)
    try:
        recipe = load(args.config)
    except (OSError, ValueError):
        print(f'{name}: CONFIG_EXACT_REQUIRED', file=sys.stderr)
        return 2
    if args.command == 'check':
        print(json.dumps(dict(recipe=recipe.to_dict(), estimate=estimate(recipe),
                              scope='STATIC_ONLY_NO_DEVICE_ADMISSION'), allow_nan=False))
        return 0
    if (not sys.dont_write_bytecode or sys.platform != 'win32'
            or not args.root.is_absolute() or not args.root.is_relative_to(Path(r'G:\lxy'))
            or any(part.endswith(('.', ' ')) or ':' in part for part in args.root.parts[1:])):
        print(rf'{name}: Windows, python -B, and absolute --root under G:\lxy required',
              file=sys.stderr)
        return 2
    from cipher_lab.artifacts import RunStore
    from cipher_lab.execution import Clocks, Meter, Stopped

    from ..resources import Resources, production_rss_bytes
    from .source import no_reparse

    clocks = Clocks()
    try:
        selected_id = run_id(recipe)
        execute, source_identity, render = wiring(recipe)
        resources = Resources(Meter(make_policy(recipe, clocks), clocks, production_rss_bytes))
        resources.check()
        root = no_reparse(args.root)
        for kind in ('data', 'models', 'result'):
            no_reparse(root / kind / selected_id)
        store = RunStore.claim(root, selected_id, output_bytes=recipe.resources.artifact_bytes,
                               needs_models=True, model_slots=slots,
                               source_supplier=source_identity)
    except Stopped as error:
        print(json.dumps(dict(command_status='STOPPED', scientific_status='UNKNOWN',
                              reason=error.reason, exit_code=3,
                              exit_code_scope='REQUESTED_NOT_NATIVE')), flush=True)
        return 3
    except Exception:
        print(f'{name}: CLAIM_FAILED', file=sys.stderr)
        return 2
    execute(store, resources)
    code, message = finish_run(store, root, resources.meter, selected_id, render)
    print(message, flush=True)
    return code

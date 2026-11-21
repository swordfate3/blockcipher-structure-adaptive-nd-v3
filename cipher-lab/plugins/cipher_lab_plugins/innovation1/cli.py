"""Single explicit stdlib check/run interface; no scientific imports on check."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, cast

from .config import GPU_UUID, Recipe, estimate, load_config, validate_run_id
from .layout_probe.config import LAYOUT_NAMESPACES, LayoutRecipe, official
from .single_train.config import estimate as estimate_pair

if TYPE_CHECKING:
    from cipher_lab.artifacts import RunStore

    from .resources import Resources

_MAX_RUN_RECORD = 2**20


def _windows_root(root: PureWindowsPath) -> bool:
    return (root.is_absolute() and root.is_relative_to(PureWindowsPath(r'G:\lxy'))
            and '..' not in root.parts
            and all(not part.endswith(('.', ' ')) and not PureWindowsPath(part).is_reserved()
                    and not any(ord(ch) < 32 or ch in ':<>\"|?*' for ch in part)
                    for part in root.parts[1:]))


def _owner_record(store: RunStore, root: Path, run_id: str) -> tuple[dict[str, Any], str]:
    from cipher_lab.artifacts import ArtifactError, _decode, _validate

    from .source import no_reparse

    # RunStore exposes no public frozen digest. Confirm this owner's existing seal,
    # not just a terminal-looking JSON read from disk (also works on main core).
    path = no_reparse(root / 'data' / run_id / 'run.json')
    frozen = store._frozen_sha
    if not store._sealed or frozen is None or path != store._data / 'run.json':
        raise ArtifactError('OWNER_RECORD_UNVERIFIED')
    with path.open('rb') as stream:
        raw = stream.read(_MAX_RUN_RECORD + 1)
    if not raw or len(raw) > _MAX_RUN_RECORD or sha256(raw).hexdigest() != frozen:
        raise ArtifactError('OWNER_RECORD_UNVERIFIED')
    saved = cast(dict[str, Any], _validate(_decode(raw), run_id))
    if saved['status'] not in ('SUCCEEDED', 'STOPPED', 'FAILED'):
        raise ArtifactError('OWNER_RECORD_UNVERIFIED')
    return saved, frozen


def _verified_artifacts(root: Path, saved: dict[str, Any], run_id: str) -> None:
    from cipher_lab.artifacts import _location, _stream

    from .source import no_reparse

    for ref, metadata in saved['artifacts'].items():
        _stream(no_reparse(_location(root, run_id, ref)), metadata)


def _finish(store: RunStore, root: Path, run_id: str, *, resources: Resources,
            requested: Mapping[str, str] | None = None) -> int:
    from cipher_lab.execution import Stopped

    from .source import no_reparse

    scientific, command, report, code = 'UNKNOWN', 'UNKNOWN', 'NOT_ATTEMPTED', 7
    reason, scientific_reason, written = None, None, False
    integrity = 'NOT_CHECKED'
    request_identity: dict[str, object] | None = None
    try:
        resources.check()
    except Stopped as error:
        command, report, code, reason = 'STOPPED', 'STOPPED', 3, error.reason
    try:
        saved, frozen = _owner_record(store, root, run_id)
    except Exception:
        if reason is None:
            reason = 'OWNER_RECORD_UNVERIFIED'
    else:
        scientific = saved['status']
        scientific_reason = saved['summary'].get('reason')
        request = saved['policy'].get('request')
        if isinstance(request, Mapping) and isinstance(request.get('gpu_uuid'), str):
            request_identity = {**request, 'scope': 'PERSISTED_POLICY'}
        elif requested is not None:
            request_identity = {**requested, 'scope': 'REQUESTED_ONLY_NOT_PERSISTED'}
        if reason is None:
            try:
                _verified_artifacts(root, saved, run_id)
                integrity = 'VERIFIED'
                resources.check()
                if not saved['recipe']:
                    # The owner sealed before its first business snapshot: there is no
                    # scientific setup or recipe to render. This is a storage failure,
                    # not a report failure; the seal reason remains scientific_reason.
                    detail = (scientific_reason if isinstance(scientific_reason, str)
                              else 'SEALED_WITHOUT_RECIPE')
                    command, report, code = 'FAILED', 'NOT_ATTEMPTED', 4
                    reason = f'ARTIFACT_{detail}'
                else:
                    method = saved['recipe'].get('method')
                    if method in LAYOUT_NAMESPACES:
                        from .layout_probe.report import render
                    elif method == Recipe().method:
                        from .report import render
                    else:
                        raise ValueError('SAVED_METHOD_UNVERIFIED')
                    command = scientific
                    code = {'SUCCEEDED': 0, 'STOPPED': 3, 'FAILED': 4}[scientific]
                    files = render({**saved, 'run_ref': f'data/{run_id}/run.json',
                                    'run_sha256': frozen})
                    for name in ('report.md', 'index.html', 'source.json'):
                        no_reparse(root / 'result' / run_id / name)
                    resources.check()
                    store.write_report_files(files)
                    written = True
                    resources.check()
                    report = 'SAVED'
            except Stopped as error:
                command, report, code, reason = 'STOPPED', 'STOPPED', 3, error.reason
            except Exception:
                command, report, code = 'FAILED', 'FAILED', 6
                if integrity != 'VERIFIED':
                    integrity, reason = 'FAILED', 'SAVED_ARTIFACT_INTEGRITY_FAILED'
                else:
                    reason = 'REPORT_FAILED'
    if request_identity is None and requested is not None:
        # The sealed record could not be read, so never claim the value was saved.
        request_identity = {**requested, 'scope': 'REQUESTED_UNVERIFIED'}
    phase_reason = reason
    try:
        resources.check()
    except Stopped as error:
        command, report, code, reason = 'STOPPED', 'STOPPED', 3, error.reason
    print(json.dumps(dict(run_id=run_id, scientific_status=scientific, command_status=command,
                          scientific_reason=scientific_reason, artifact_integrity=integrity,
                          report_status=report, report_files_written=written, reason=reason,
                          phase_reason=phase_reason, request_identity=request_identity,
                          exit_code=code, exit_code_scope='REQUESTED_NOT_NATIVE',
                          command_resources=resources.snapshot(),
                          counts_scope='science finalize; command_resources pre-stdout'),
                     allow_nan=False), flush=True)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='cipher-lab --plugin cipher_lab_plugins.innovation1.cli',
                                     allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check', allow_abbrev=False)
    run = commands.add_parser('run', allow_abbrev=False)
    for command in (check, run):
        command.add_argument('--config', type=Path, required=True)
    run.add_argument('--root', type=Path, required=True)
    run.add_argument('--id', required=True)
    run.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args(argv)
    try:
        recipe = load_config(args.config)
        if args.command == 'check':
            print(json.dumps(dict(recipe=recipe.to_dict(), estimate=(
                estimate_pair(recipe) if isinstance(recipe, LayoutRecipe) else estimate(recipe)),
                scope='STATIC_ONLY_NO_DEVICE_ADMISSION'), allow_nan=False))
            return 0
        validate_run_id(args.id)
        if not GPU_UUID.fullmatch(args.gpu_uuid):
            raise ValueError
        if sys.platform == 'win32' and (not sys.dont_write_bytecode
                                        or not _windows_root(PureWindowsPath(args.root))):
            raise ValueError
        if isinstance(recipe, LayoutRecipe):
            recipe = official(replace(recipe, run_id=args.id), args.id)
            if recipe.pool_source is not None:
                # Keep write ownership disjoint from the pinned read-only archive.
                requested_root = PureWindowsPath(args.root)
                source_root = PureWindowsPath(recipe.pool_source.root)
                if (requested_root.is_relative_to(source_root)
                        or source_root.is_relative_to(requested_root)):
                    raise ValueError('POOL_ROOT_OVERLAP')
    except (ValueError, OSError):
        print('innovation1: invalid settings', file=sys.stderr)
        return 2

    from cipher_lab.artifacts import RunStore
    from cipher_lab.execution import Stopped

    from .application import execute, make_resources
    from .contracts import SLOTS
    from .layout_probe.config import SLOTS as LAYOUT_SLOTS
    from .source import no_reparse, source_identity

    try:
        root = no_reparse(args.root)
        for kind in ('data', 'models', 'result'):
            no_reparse(root / kind / args.id)
        resources = make_resources(recipe)
        resources.check()
        store = RunStore.claim(root, args.id, output_bytes=recipe.resources.artifact_bytes,
                               needs_models=True, source_supplier=source_identity,
                               model_slots=LAYOUT_SLOTS if isinstance(recipe, LayoutRecipe)
                               else SLOTS)
    except Stopped as error:
        print(json.dumps(dict(scientific_status='UNKNOWN', command_status='STOPPED',
                              report_status='NOT_ATTEMPTED', reason=error.reason, exit_code=3,
                              exit_code_scope='REQUESTED_NOT_NATIVE')), flush=True)
        return 3
    except Exception:
        print('innovation1: CLAIM_FAILED', file=sys.stderr)
        return 2
    execute(recipe, store, resources, gpu_uuid=args.gpu_uuid, run_id=args.id)
    return _finish(store, root, args.id, resources=resources,
                   requested=dict(gpu_uuid=args.gpu_uuid))


if __name__ == '__main__':
    raise SystemExit(main())

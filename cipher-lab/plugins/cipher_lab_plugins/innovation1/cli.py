"""Single explicit stdlib check/run interface; no scientific imports on check."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from math import isfinite
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, cast

from .config import GPU_UUID, Recipe, canonical, estimate, load_config, validate_run_id
from .four_single.config import METHOD as FOUR_SINGLE_METHOD
from .four_single.config import REPLAY_METHOD, FourSingleRecipe, ReplayRecipe, validate_replay_root
from .four_single.config import SLOT as FOUR_SINGLE_SLOT
from .four_single.config import estimate as estimate_four_single
from .four_single.config import official as official_four_single
from .four_single.interleave_config import INTERLEAVE_METHODS, InterleaveRecipe
from .four_single.interleave_config import validate_launch as validate_interleave_launch
from .layout_probe.config import LAYOUT_NAMESPACES, LayoutRecipe, official
from .present_neighborhood.config import CARD_METHOD as PRESENT_CARD_METHOD
from .present_neighborhood.config import GPU_UUID as NEIGHBORHOOD_GPU_UUID
from .present_neighborhood.config import METHOD as NEIGHBORHOOD_METHOD
from .present_neighborhood.config import NeighborhoodRecipe, PresentCardRecipe
from .present_neighborhood.config import estimate as estimate_neighborhood
from .present_neighborhood.config import official as official_neighborhood
from .present_neighborhood.config import validate_root as validate_neighborhood_root
from .serial_bc.config import (
    DEV_OBSERVATION_BYTES,
    INDEPENDENT_METHOD,
    INDEPENDENT_ROOT,
    VOLUME_GPU_UUID,
    VOLUME_METHOD,
    VOLUME_ROOT,
    IndependentRecipe,
    SerialRecipe,
    VolumeRecipe,
)
from .serial_bc.config import METHOD as SERIAL_METHOD
from .serial_bc.config import estimate as estimate_serial
from .serial_bc.config import official as official_serial
from .serial_bc.config import validate_root as validate_serial_root
from .single_train.config import estimate as estimate_pair
from .skinny_card.config import GPU_UUID as CARD_GPU_UUID
from .skinny_card.config import METHOD as CARD_METHOD
from .skinny_card.config import SLOTS as CARD_SLOTS
from .skinny_card.config import CardRecipe
from .skinny_card.config import estimate as estimate_card
from .skinny_card.config import official as official_card
from .skinny_card.config import validate_root as validate_card_root

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


def _volume_points(recipe: dict[str, Any]) -> dict[str, tuple[int, int]]:
    tr = recipe['training']
    per_epoch, epochs = tr['batches_per_epoch'], tr['epochs']
    updates = recipe['selection']['dev_updates']
    if (any(type(n) is not int or n <= 0 for n in
            (per_epoch, epochs, tr['batch_groups'], tr['dev_groups']))
            or not isinstance(updates, list) or len(updates) != 3
            or any(type(n) is not int or n <= 0 or n % per_epoch for n in updates)
            or updates != sorted(set(updates)) or updates[-1] != per_epoch * epochs):
        raise ValueError('VOLUME_DEV_EVIDENCE')
    return {f'payload/dev-{n}.json': (n, n // per_epoch) for n in updates}


def _dev_observation(raw: bytes, ref: str, point: tuple[int, int],
                     dev_groups: int) -> dict[str, Any]:
    from cipher_lab.artifacts import _decode, _mapping

    def scalar(value: object, upper: float = float('inf')) -> bool:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        return isfinite(value) and 0 <= value <= upper

    row = _mapping(_decode(raw))
    dev, routes = _mapping(row.get('dev')), _mapping(row.get('routes'))
    weights = _mapping(routes.get('SKINNY'))
    if (set(row) != {'updates', 'epoch', 'task', 'dev', 'routes', 'mean_online_train_bce'}
            or row['task'] != 'SKINNY'
            or any(type(row[k]) is not int or row[k] != n
                   for k, n in zip(('updates', 'epoch'), point, strict=True))
            or set(dev) != {'groups', 'bce', 'accuracy'}
            or type(dev['groups']) is not int or dev['groups'] != dev_groups
            or not scalar(dev['bce']) or not scalar(dev['accuracy'], 1.)
            or not scalar(row['mean_online_train_bce'])
            or set(routes) != {'SKINNY'} or not weights
            or not all(stem.isascii() and stem.isidentifier() and scalar(w, 1.)
                       for stem, w in weights.items())):
        raise ValueError('VOLUME_DEV_EVIDENCE')
    return dict(row, evidence_ref=ref)


def _check_volume_counts(saved: dict[str, Any], observations: list[dict[str, Any]]) -> None:
    # Counts can include later partial TRAIN/DEV work. Require enough completed
    # work for every saved observation, not equality to the last saved point.
    tr, counts = saved['recipe']['training'], saved['counts']
    total = tr['batches_per_epoch'] * tr['epochs']
    batch, dev = tr['batch_groups'], tr['dev_groups']
    completed = counts.get('completed_optimizer_updates')
    attempted = counts.get('attempted_optimizer_updates')
    last_saved = observations[-1]['updates'] if observations else 0
    success = saved['status'] == 'SUCCEEDED'
    if (type(completed) is not int or type(attempted) is not int
            or not last_saved <= completed <= attempted <= total
            or success and completed != total):
        raise ValueError('VOLUME_DEV_EVIDENCE')
    points = saved['recipe']['selection']['dev_updates']
    reached = sum(point <= completed for point in points)
    lower_bounds = dict(forward_rows=completed * batch + len(observations) * dev,
                        real_blocks=0, null_words=0)
    upper_bounds = dict(forward_rows=attempted * batch + reached * dev,
                        real_blocks=0, null_words=0)
    for kind, low in lower_bounds.items():
        done, tried = counts.get(f'completed_{kind}'), counts.get(f'attempted_{kind}')
        high = upper_bounds[kind]
        if (type(done) is not int or type(tried) is not int or not low <= done <= tried <= high
                or success and done != high):
            raise ValueError('VOLUME_DEV_EVIDENCE')


def _verified_artifacts(root: Path, saved: dict[str, Any], run_id: str) -> dict[str, Any]:
    from cipher_lab.artifacts import _location, _stream

    from .source import no_reparse

    artifacts, recipe = saved['artifacts'], saved['recipe']
    volume = recipe.get('method') == VOLUME_METHOD
    points: dict[str, tuple[int, int]] = {}
    registered: list[str] = []
    if volume:
        if any(not ref.startswith(('payload/batches-', 'payload/dev-')) for ref in artifacts):
            raise ValueError('VOLUME_ARTIFACT_KIND')
        points = _volume_points(recipe)
        registered = [ref for ref in points if ref in artifacts]
        if (set(registered) != {ref for ref in artifacts if ref.startswith('payload/dev-')}
                or registered != list(points)[:len(registered)]
                or saved['status'] == 'SUCCEEDED' and registered != list(points)):
            raise ValueError('VOLUME_DEV_EVIDENCE')
    verified: dict[str, dict[str, Any]] = {}
    for ref, metadata in artifacts.items():
        collect = ref in points
        if collect and not 0 < metadata['bytes'] <= DEV_OBSERVATION_BYTES:
            raise ValueError('VOLUME_DEV_EVIDENCE')
        raw = _stream(no_reparse(_location(root, run_id, ref)), metadata, collect=collect)
        if collect:
            verified[ref] = _dev_observation(raw, ref, points[ref],
                                             recipe['training']['dev_groups'])
    if not volume:
        return saved
    observations = [verified[ref] for ref in registered]
    _check_volume_counts(saved, observations)
    if (saved['status'] == 'SUCCEEDED'
            and canonical(saved['summary'].get('dev_observations')) != canonical(observations)):
        raise ValueError('VOLUME_DEV_EVIDENCE')
    # Keep the sealed summary and digest intact. The renderer uses only these
    # separately verified rows, even if the bounded FAILED seal dropped its list.
    return {**saved, 'verified_dev_observations': observations}


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
                render_record = _verified_artifacts(root, saved, run_id)
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
                    elif method in (FOUR_SINGLE_METHOD, REPLAY_METHOD, SERIAL_METHOD,
                                    INDEPENDENT_METHOD, VOLUME_METHOD, *INTERLEAVE_METHODS):
                        from .four_single.report import render
                    elif method in (NEIGHBORHOOD_METHOD, PRESENT_CARD_METHOD):
                        from .present_neighborhood.report import render
                    elif method == CARD_METHOD:
                        from .skinny_card.report import render
                    elif method == Recipe().method:
                        from .report import render
                    else:
                        raise ValueError('SAVED_METHOD_UNVERIFIED')
                    command = scientific
                    code = {'SUCCEEDED': 0, 'STOPPED': 3, 'FAILED': 4}[scientific]
                    files = render({**render_record, 'run_ref': f'data/{run_id}/run.json',
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
                estimate_serial(recipe) if isinstance(recipe, SerialRecipe)
                else estimate_neighborhood(recipe) if isinstance(recipe, NeighborhoodRecipe)
                else estimate_card(recipe) if isinstance(recipe, CardRecipe)
                else estimate_four_single(recipe) if isinstance(recipe, FourSingleRecipe)
                else estimate_pair(recipe) if isinstance(recipe, LayoutRecipe)
                else estimate(recipe)),
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
            pool_source = recipe.pool_source
        elif isinstance(recipe, SerialRecipe):
            official_serial(recipe, args.id)
            if isinstance(recipe, VolumeRecipe):
                validate_serial_root(args.root, expected=VOLUME_ROOT)
                if args.gpu_uuid.casefold() != VOLUME_GPU_UUID.casefold():
                    raise ValueError('VOLUME_GPU_UUID')
            elif isinstance(recipe, IndependentRecipe):
                validate_serial_root(args.root, expected=INDEPENDENT_ROOT)
            else:
                validate_serial_root(args.root)
            pool_source = recipe.skinny_source
        elif isinstance(recipe, NeighborhoodRecipe):
            official_neighborhood(recipe, args.id)
            if isinstance(recipe, PresentCardRecipe):
                validate_neighborhood_root(args.root, recipe=recipe)
            else:
                validate_neighborhood_root(args.root)
            if args.gpu_uuid.casefold() != NEIGHBORHOOD_GPU_UUID.casefold():
                raise ValueError('NEIGHBORHOOD_GPU_UUID')
            pool_source = recipe.skinny_source
        elif isinstance(recipe, CardRecipe):
            official_card(recipe, args.id)
            validate_card_root(args.root)
            if args.gpu_uuid.casefold() != CARD_GPU_UUID.casefold():
                raise ValueError('CARD_GPU_UUID')
            pool_source = recipe.skinny_source
        elif isinstance(recipe, FourSingleRecipe):
            recipe = official_four_single(replace(recipe, run_id=args.id), args.id)
            if isinstance(recipe, InterleaveRecipe):
                validate_interleave_launch(args.root, args.gpu_uuid, recipe=recipe)
            elif isinstance(recipe, ReplayRecipe):
                validate_replay_root(args.root)
            pool_source = recipe.skinny_source
        else:
            pool_source = None
        if pool_source is not None:
            # Keep every write owner disjoint from the explicitly bound archive.
            requested_root = PureWindowsPath(args.root)
            source_root = PureWindowsPath(pool_source.root)
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
        # Start the inherited wall budget before touching the output/source owner.
        inherited = (SerialRecipe, CardRecipe, NeighborhoodRecipe, InterleaveRecipe)
        inherited_resources = make_resources(recipe) if isinstance(recipe, inherited) else None
        root = no_reparse(args.root)
        needs_models = not isinstance(recipe, VolumeRecipe)
        for kind in (('data', 'models', 'result') if needs_models else ('data', 'result')):
            no_reparse(root / kind / args.id)
        resources = (inherited_resources if inherited_resources is not None
                     else make_resources(recipe))
        resources.check()
        claim_in_progress = True

        def deadline_claim_source() -> dict[str, str]:
            identity = source_identity()
            # Claim scans before any owner mkdir. This is cooperative, not atomic
            # with subsequent filesystem calls. Later live scans share the supplier.
            if claim_in_progress:
                resources.check()
            return identity

        claim_checked = (VolumeRecipe, CardRecipe, NeighborhoodRecipe, InterleaveRecipe)
        source_supplier = (deadline_claim_source if isinstance(recipe, claim_checked)
                           else source_identity)
        store = RunStore.claim(root, args.id, output_bytes=recipe.resources.artifact_bytes,
                               needs_models=needs_models, source_supplier=source_supplier,
                               model_slots=LAYOUT_SLOTS if isinstance(recipe, LayoutRecipe)
                               else recipe.stage_slots if isinstance(recipe, SerialRecipe)
                               else recipe.slots if isinstance(recipe, NeighborhoodRecipe)
                               else CARD_SLOTS if isinstance(recipe, CardRecipe)
                               else (FOUR_SINGLE_SLOT,) if isinstance(recipe, FourSingleRecipe)
                               else SLOTS)
        claim_in_progress = False
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

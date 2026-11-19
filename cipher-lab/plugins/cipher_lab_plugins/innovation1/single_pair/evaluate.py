"""Exactly one sealed SKINNY TEST panel and two frozen weights; no training path."""
from __future__ import annotations

import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from cipher_lab.artifacts import (
    ArtifactError,
    RunStore,
    _decode,
    _token,
    _validate,
    read_binary,
)
from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter, Stopped

from ..config import Recipe, Task, validate
from ..config import _decode as decode_recipe
from ..contracts import MAX_CHECKPOINT
from ..contracts import SLOTS as ORIGINAL_SLOTS
from ..resources import Resources, ResourceStop, production_rss_bytes
from ..source import source_identity as original_source_identity

METHOD = 'innovation1-skinny-single-pair-v1'
INPUT_ID = 'innovation1-fresh-20261117-01'
INPUT_RELEASE = 'cb3491afd0b6865e3ff9899984a0526150b30b66'
INPUT_SHA = '5d70f6ce4e5e0fc73cdc9f9107496f9aa600e3e953ec66164fc5cf639bfa64c7'
OUTPUT_ID = 'innovation1-skinny-single-pair-20261119-02'
GPU_UUID = 'GPU-171dfbbe-c1a7-d93b-549d-8824cb8a18c7'
GPU_PCI = '00000000:3B:00.0'
SLOTS = ('M-best', 'D-SKINNY-best')
TEST_REF = 'dataset/test-SKINNY.npz'
# Same binary limit as the original codec, without importing torch before admission.
PANEL_LIMIT = 128 * 2**20
RECORD_LIMIT = 2**20
OUTPUT_LIMIT = 16 * 2**20
GROUPS = 262_144
CHUNK = 512
MAX_ROWS = 524_288
PAIR_INDEX = 0
WALL = 1200
CPU = 4800
RSS = 16 * 2**30
GPU = 20 * 2**30


class Refused(ValueError):
    """Own fixed public reason, never raw exceptions from a dependency."""


def no_reparse(path: Path) -> Path:
    if '..' in path.parts or (os.name == 'nt' and any(
            part.endswith(('.', ' ')) or ':' in part for part in path.parts[1:])):
        raise Refused('UNSAFE_PATH')
    absolute = path.absolute()
    for p in (*reversed(absolute.parents), absolute):
        try:
            info = p.lstat()
        except FileNotFoundError:
            continue
        if getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise Refused('UNSAFE_PATH')
        if not (stat.S_ISDIR(info.st_mode) or (p == absolute and stat.S_ISREG(info.st_mode))):
            raise Refused('UNSAFE_PATH')
    return absolute


def source_identity() -> dict[str, str]:
    """Preserve the original top-level scan, then hash every nested source byte."""
    result = original_source_identity()
    nested = Path(__file__).absolute().parent
    prefix = 'cipher_lab_plugins/innovation1/single_pair/'
    required = {prefix + name for name in ('__init__.py', 'cli.py', 'evaluate.py',
                                           'models.py', 'report.py')}

    def fail(error: OSError) -> None:
        raise error

    try:
        for directory, dirs, files in os.walk(no_reparse(nested), onerror=fail):
            for name in dirs:
                no_reparse(Path(directory) / name)
            for name in files:
                if name.endswith('.py'):
                    path = no_reparse(Path(directory) / name)
                    key = path.relative_to(nested.parents[2]).as_posix()
                    if key in result:
                        raise ValueError
                    result[key] = sha256(path.read_bytes()).hexdigest()
        if not required <= result.keys():
            raise ValueError
    except (OSError, ValueError):
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
    return dict(sorted(result.items()))


def make_policy(clocks: Clocks) -> Policy:
    return Policy(2, WALL, CPU, (clocks.utc() + timedelta(seconds=WALL)).isoformat(),
                  0, 0, RSS, OUTPUT_LIMIT, 0, MAX_ROWS)


def read_record(root: Path, run_id: str, cap: int, expected_sha: str | None = None
                ) -> dict[str, Any]:
    """Bound the first read and validate the original RunStore schema, not its payloads."""
    path = no_reparse(root / 'data' / _token(run_id) / 'run.json')
    if not path.is_file():
        raise Refused('RUN_JSON_TYPE')
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
    if not 0 < len(raw) <= cap:
        raise Refused('RUN_JSON_SIZE')
    digest = sha256(raw).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise Refused('INPUT_RUN_SHA')
    saved = _validate(_decode(raw), run_id)
    return {**saved, 'run_ref': f'data/{run_id}/run.json', 'run_sha256': digest}


@dataclass(frozen=True)
class Inputs:
    recipe: Recipe
    task: Task
    source: dict[str, str]
    checkpoints: dict[str, bytes]
    metadata: dict[str, dict[str, Any]]
    test: bytes
    old_accuracy: dict[str, float]
    refs: dict[str, str]


def load_inputs(root: Path, source: dict[str, str], check: Callable[[], None]) -> Inputs:
    saved = read_record(root, INPUT_ID, RECORD_LIMIT, INPUT_SHA)
    summary = saved['summary']
    if (saved['status'] != 'SUCCEEDED' or summary.get('reason') != 'COMPLETE'
            or summary.get('backend_identity') != 'ciphers' or summary.get('synthetic') is not False
            or summary.get('persisted_reloads') != list(ORIGINAL_SLOTS)):
        raise Refused('INPUT_RUN_IDENTITY')
    if saved['source'] != source:
        raise Refused('SOURCE_IDENTITY')
    recipe = Recipe()
    try:
        parsed = decode_recipe(saved['recipe'], recipe, 'recipe')
        validate(parsed)
        if parsed != recipe or summary.get('method') != recipe.method:
            raise ValueError
    except ValueError:
        raise Refused('INPUT_RECIPE') from None
    task = next(t for t in recipe.tasks if t.cipher == 'SKINNY')
    refs: dict[str, str] = {}
    artifacts = {f'model/{s}.pt': (root / 'models' / INPUT_ID / f'{s}.pt', MAX_CHECKPOINT)
                 for s in SLOTS}
    artifacts[TEST_REF] = (root / 'data' / INPUT_ID / 'dataset' / 'test-SKINNY.npz', PANEL_LIMIT)
    try:
        for ref, (path, limit) in artifacts.items():
            entry = saved['artifacts'][ref]
            if not 0 < entry['bytes'] <= limit:
                raise ValueError
            refs[ref] = entry['sha256']
            no_reparse(path)
        best = {slot: summary['best'][slot] for slot in SLOTS}
        metadata = {slot: best[slot]['metadata'] for slot in SLOTS}
        for slot, entry in best.items():
            ref = f'model/{slot}.pt'
            if entry['ref'] != ref or entry['sha256'] != refs[ref]:
                raise ValueError
        compare = summary['compare']['SKINNY']
        old = {slot: compare[key]['accuracy'] for slot, key in
               zip(SLOTS, ('shared', 'dedicated'), strict=True)}
        if not all(type(value) in (int, float) and 0 <= value <= 1 for value in old.values()):
            raise ValueError
    except Refused:
        raise
    except (TypeError, KeyError, ValueError):
        raise Refused('INPUT_BINDING') from None
    raw = {}
    # Frozen read_binary reopens and validates run.json per reference. The trusted,
    # non-concurrently-mutated root contract makes these rereads of the already
    # bounded pinned record; recheck pin and path before/after those calls as well.
    for ref, (path, limit) in artifacts.items():
        check()
        read_record(root, INPUT_ID, RECORD_LIMIT, INPUT_SHA)
        no_reparse(path)
        raw[ref] = read_binary(root, INPUT_ID, ref, refs[ref], limit)
        check()
    read_record(root, INPUT_ID, RECORD_LIMIT, INPUT_SHA)
    return Inputs(recipe, task, source, {s: raw[f'model/{s}.pt'] for s in SLOTS},
                  metadata, raw[TEST_REF], old, refs)


class Forwards:
    def __init__(self, resources: Resources) -> None:
        self.resources = resources
        self.by_slot = {slot: {'attempted': 0, 'completed': 0} for slot in SLOTS}

    def reserve(self, slot: str) -> None:
        if self.by_slot[slot]['attempted'] + CHUNK > GROUPS:
            raise Stopped('FORWARD_ROWS_LIMIT')
        self.resources.reserve('forward_rows', CHUNK)
        self.by_slot[slot]['attempted'] += CHUNK

    def complete(self, slot: str) -> None:
        self.resources.complete('forward_rows', CHUNK)
        self.by_slot[slot]['completed'] += CHUNK


def check_panel(panel: Any, task: Task) -> None:
    import numpy as np

    x, y = panel.ciphertext, panel.labels
    if (panel.groups != GROUPS or panel.task != task or panel.split != 'test' or panel.index != 0
            or x.shape != (GROUPS, 4, 2, 8) or x.dtype != np.uint8
            or y.shape != (GROUPS, 1) or y.dtype != np.uint8):
        raise Refused('PANEL_SHAPE')


def score(panel: Any, runtime: Any, ledger: Forwards,
          old: dict[str, float]) -> dict[str, dict[str, float]]:
    import numpy as np

    from ..scoring import bce

    check_panel(panel, Recipe().tasks[-1])
    correct = dict.fromkeys(SLOTS, 0)
    bce_sum = dict.fromkeys(SLOTS, 0.)
    for start in range(0, GROUPS, CHUNK):
        ledger.resources.check()
        stop = start + CHUNK
        packed = panel.ciphertext[start:stop, PAIR_INDEX:PAIR_INDEX + 1]
        y = panel.labels[start:stop]
        if packed.shape != (CHUNK, 1, 2, 8) or y.shape != (CHUNK, 1):
            raise Refused('PANEL_SHAPE')
        bits = np.unpackbits(packed, axis=-1, bitorder='big').astype(np.float32)
        prepared = runtime.prepare(bits)
        for slot in SLOTS:
            ledger.reserve(slot)
            z = np.asarray(runtime.predict(slot, prepared), dtype=np.float64)
            if z.shape != (CHUNK, 1) or not np.isfinite(z).all():
                raise Refused('SCORE_INVALID')
            ledger.complete(slot)
            ledger.resources.check()
            correct[slot] += int(np.count_nonzero((z >= 0) == y))
            bce_sum[slot] += CHUNK * bce(y, z)
    ledger.resources.check()
    return {slot: dict(accuracy=correct[slot] / GROUPS, bce=bce_sum[slot] / GROUPS,
                       old_four_pair_accuracy=old[slot],
                       vs_four_pair_pp=100 * (correct[slot] / GROUPS - old[slot]))
            for slot in SLOTS}


def evaluate(inputs: Inputs, ledger: Forwards, evidence: dict[str, Any]) -> dict[str, Any]:
    from ..application import production_resources

    resources = ledger.resources
    if inputs.recipe.resources.gpu_bytes != GPU:
        raise Refused('DEVICE_BUDGET')

    def observe(event: dict[str, Any]) -> None:
        # No probe/proof refresh here. Original production assembly owns its gate.
        evidence['setup_stage'] = event['stage']
        if event.get('attestation') is not None:
            evidence['attestation'] = event['attestation']
        resources.meter.check()

    resources.check()
    wired = production_resources(inputs.recipe, 'cuda:0', GPU_UUID, GPU_PCI, observer=observe)
    if wired.gpu is None:
        raise ResourceStop(detail_code='GPU_SAMPLE_INVALID')
    resources.gpu = wired.gpu
    resources.check()

    # Dataset/checkpoint codecs import torch. Admission must already have run.
    from ..checkpoint import canonical, identity
    from ..dataset import MAX_PANEL, decode_panel
    from .models import restore

    resources.check()
    if MAX_PANEL != PANEL_LIMIT:
        raise Refused('CODEC_BUDGET')
    evidence['phase'] = 'checkpoint-identity'
    try:
        expected = {s: identity(inputs.recipe, inputs.source, 'ciphers', s,
                                inputs.metadata[s]['epoch'], inputs.metadata[s]['score'])
                    for s in SLOTS}
        if canonical(expected) != canonical(inputs.metadata):
            raise ValueError
    except (ValueError, KeyError, TypeError):
        raise Refused('CHECKPOINT_IDENTITY') from None
    resources.check()
    evidence['phase'] = 'test-decode'
    panel = decode_panel(inputs.test, inputs.recipe, inputs.task, 'test', inputs.source, 'ciphers')
    resources.check()
    check_panel(panel, inputs.task)
    evidence['phase'] = 'checkpoint-restore'
    runtime = restore(inputs.recipe, inputs.task, inputs.checkpoints, expected, resources.check)
    resources.check()
    evidence['phase'] = 'forward'
    return score(panel, runtime, ledger, inputs.old_accuracy)


def _finish(root: Path, store: RunStore, meter: Meter, terminal: str, persist_failed: bool
            ) -> dict[str, Any]:
    from .report import render

    scientific = 'UNKNOWN'
    scientific_reason = None
    report_status, report_reason = 'NOT_ATTEMPTED', None
    code = 6
    try:
        # Only new, bounded, no-artifact run is read. Never scan the old record's references.
        saved = read_record(root, OUTPUT_ID, OUTPUT_LIMIT)
        if saved['status'] in ('SUCCEEDED', 'FAILED', 'STOPPED'):
            scientific = saved['status']
            scientific_reason = saved['summary'].get('reason')
        meter.check()
        if persist_failed or scientific != terminal:
            raise Refused('FINALIZE_FAILED')
        if saved.get('artifacts'):
            raise Refused('UNEXPECTED_OUTPUT_ARTIFACTS')
        files = render(saved)
        meter.check()
        for name in files:
            no_reparse(root / 'result' / OUTPUT_ID / name)
        store.write_report_files(files)
        report_status = 'SAVED'
        code = {'SUCCEEDED': 0, 'FAILED': 4, 'STOPPED': 3}[scientific]
    except Stopped as error:
        report_status, report_reason, code = 'STOPPED', error.reason, 3
    except Exception:
        report_status = 'FAILED'
        report_reason = 'FINALIZE_FAILED' if persist_failed else 'REPORT_FAILED'
    phase_reason = report_reason
    try:
        meter.check()
    except Stopped as error:
        report_reason, code = error.reason, 3
    return dict(scientific_status=scientific, scientific_reason=scientific_reason,
                command_status='SUCCEEDED' if code == 0 else 'STOPPED' if code == 3 else 'FAILED',
                report_status=report_status, closeout_reason=report_reason,
                report_reason=phase_reason,
                command_resources=meter.snapshot(), exit_code=code)


def run(input_root: Path, output_root: Path) -> tuple[dict[str, Any], Meter | None]:
    """Return command evidence and its original Meter for the CLI pre-stdout check."""
    try:
        clocks = Clocks()
        policy = make_policy(clocks)
        meter = Meter(policy, clocks, production_rss_bytes)
    except Stopped as error:
        return dict(scientific_status='UNKNOWN', command_status='STOPPED',
                    closeout_reason=error.reason, exit_code=3), None
    try:
        meter.check()
        for kind in ('data', 'result', 'models'):
            no_reparse(output_root / kind / OUTPUT_ID)
        if (output_root / 'models' / OUTPUT_ID).exists():
            raise Refused('OUTPUT_OCCUPIED')
        store = RunStore.claim(output_root, OUTPUT_ID, output_bytes=OUTPUT_LIMIT,
                               source_supplier=source_identity)
    except Stopped as error:
        return dict(scientific_status='UNKNOWN', command_status='STOPPED',
                    closeout_reason=error.reason, command_resources=meter.snapshot(),
                    exit_code=3), meter
    except Exception:
        try:
            meter.check()
        except Stopped as error:
            return dict(scientific_status='UNKNOWN', command_status='STOPPED',
                        closeout_reason=error.reason, command_resources=meter.snapshot(),
                        exit_code=3), meter
        return dict(scientific_status='UNKNOWN', command_status='FAILED',
                    closeout_reason='CLAIM_FAILED', command_resources=meter.snapshot(),
                    exit_code=2), meter

    ledger = Forwards(Resources(meter))
    evidence: dict[str, Any] = dict(method=METHOD, phase='input', scientific_validation=False,
        input_run_id=INPUT_ID, input_release=INPUT_RELEASE, expected_input_run_sha256=INPUT_SHA)
    terminal, reason = 'FAILED', 'INPUT_INVALID'
    try:
        meter.check()
        old_source = original_source_identity()
        snapshot = store.source_snapshot()
        if (snapshot != store.current_source_identity()
                or any(snapshot.get(k) != v for k, v in old_source.items())):
            raise ResourceStop('SOURCE_CHANGED')
        inputs = load_inputs(input_root, old_source, meter.check)
        evidence.update(input_sha256=INPUT_SHA, input_source=inputs.source,
                        input_artifacts=inputs.refs,
                        old_accuracy_source='run.json:summary.compare.SKINNY.{shared,dedicated}.accuracy',
                        old_four_pair_accuracy=inputs.old_accuracy,
                        checkpoint_selection={s: {'epoch': m.get('epoch'), 'score': m.get('score')}
                                              for s, m in inputs.metadata.items()}, phase='device')
        if store.current_source_identity() != snapshot:
            raise ResourceStop('SOURCE_CHANGED')
        scores = evaluate(inputs, ledger, evidence)
        if any(v != {'attempted': GROUPS, 'completed': GROUPS} for v in ledger.by_slot.values()):
            raise Refused('FORWARD_COUNT')
        ledger.resources.check()
        if store.current_source_identity() != snapshot:
            raise ResourceStop('SOURCE_CHANGED')
        ledger.resources.check()
        evidence.update(scores=scores, m_minus_d_pp=100 * (
            scores[SLOTS[0]]['accuracy'] - scores[SLOTS[1]]['accuracy']))
        terminal, reason = 'SUCCEEDED', 'COMPLETE'
    except (Stopped, ResourceStop) as error:
        terminal, reason = 'STOPPED', error.reason
        if isinstance(error, ResourceStop):
            evidence['detail_code'] = error.detail_code
    except Refused as error:
        reason = str(error)
    except ArtifactError:
        reason = 'INPUT_ARTIFACT'
    except Exception:
        reason = {'checkpoint-restore': 'CHECKPOINT_INVALID', 'test-decode': 'DATASET_INVALID',
                  'forward': 'FORWARD_FAILED'}.get(evidence['phase'], 'EXECUTION_FAILED')

    # Bounded failure evidence remains writable after a stop. Check again after
    # persistence/reporting, but never try a second science finalize.
    evidence.update(reason=reason, backend_identity='ciphers' if 'input_source' in evidence else
                    'UNVERIFIED', synthetic=False if 'input_source' in evidence else None,
                    forward_rows_by_slot=ledger.by_slot)
    persist_failed = False
    try:
        no_reparse(output_root / 'data' / OUTPUT_ID / 'run.json')
        store.finalize(dict(status=terminal, summary=evidence, counts=ledger.resources.snapshot(),
            recipe=dict(method=METHOD, cipher='SKINNY', rounds=8, difference=0x2000, split='test',
                        groups=GROUPS, pair_index=PAIR_INDEX, chunk_groups=CHUNK, retries=0,
                        new_labels=0, threshold=0, slots=list(SLOTS)),
            policy={**policy.to_dict(), 'gpu_bytes': GPU,
                    'gpu_uuid': GPU_UUID, 'gpu_pci': GPU_PCI}))
    except (OSError, ValueError):
        persist_failed = True
    result = _finish(output_root, store, meter, terminal, persist_failed)
    return dict(result, attempted_terminal_reason=reason,
                forward_rows_by_slot=ledger.by_slot,
                detail_code=evidence.get('detail_code')), meter

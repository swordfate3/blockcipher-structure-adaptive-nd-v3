"""Read only the fixed TRAIN/DEV and, after the runner's reload gate, TEST panels."""
from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast
from zipfile import BadZipFile, ZipFile

from cipher_lab.artifacts import ArtifactError, _decode, _encode, _validate

from ..config import Task, canonical, validate_run_id
from ..contracts import MAX_CHECKPOINT
from ..layout_probe.config import PUBLIC_STREAMS, FilePin
from ..layout_probe.reuse import MAX_RECORD, PoolSourceError, _array, _pinned, load_reused_pool
from ..single_train.config import BACKEND
from ..single_train.data import FixedPairReplay, PairBatch
from .config import FOUR_SINGLE, PANEL_OVERHEAD, PARAMETERS, SLOT, ReplayRecipe, estimate
from .data import skinny_pool_recipe

_SOURCE_ERRORS = (ArtifactError, OSError, ValueError, BadZipFile, EOFError, KeyError, TypeError,
                  SyntaxError, AttributeError)


class TestSourceError(PoolSourceError):
    """No substitute or partial scoring when a reused TEST cannot be verified."""


@dataclass(frozen=True)
class _PanelPin:
    ref: str
    file: FilePin
    digest: str
    groups: int
    namespace: str


def _panel_pin(record: dict[str, Any], info: dict[str, Any], task: Task, split: str,
               groups: int, namespace: str) -> _PanelPin:
    ref = f'dataset/{split}-{task.cipher}.npz'
    entry = record['artifacts'][ref]  # _validate already enforces kind/ref/SHA/type.
    size = entry['bytes']
    payload = groups * (25 if split == 'test' else 17)
    if not payload < size <= payload + PANEL_OVERHEAD:
        raise PoolSourceError
    file = FilePin(size, entry['sha256'])
    digest = info['digest']
    if (not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)
            or canonical(info) != canonical(dict(ref=ref, **asdict(file), digest=digest,
                                                groups=groups, namespace=namespace,
                                                origin='fresh'))):
        raise PoolSourceError
    return _PanelPin(ref, file, digest, groups, namespace)


class ReplayPools:
    """No generation provider, oracle, entropy callback, old checkpoint or logits decoder."""

    def __init__(self, recipe: ReplayRecipe, check: Callable[[], None]) -> None:
        self.recipe, self._check = recipe, check
        self._skinny = skinny_pool_recipe(recipe)
        self._fixed: dict[str, FixedPairReplay] = {}
        self._pool_tasks = recipe.tasks
        self._ready = self._released = False
        self._test_pins: dict[str, _PanelPin] = {}
        self._record: dict[str, Any] = {}
        self._base = Path(recipe.archive.root) / 'data' / recipe.archive.run_id
        self.source_record = dict(**asdict(recipe.archive),
                                  record_ref=f'data/{recipe.archive.run_id}/run.json')

    def _manifest(self) -> dict[str, Any]:
        pin, tr = self.recipe.archive, self.recipe.training
        validate_run_id(pin.run_id)
        if not Path(pin.root).is_absolute():
            raise PoolSourceError
        # The official recipe is exact. Only the direct artificial seam can
        # scale panel dimensions; the old method, model, epochs and RNG stay fixed.
        old = replace(FOUR_SINGLE, run_id=pin.run_id, skinny_source=self.recipe.skinny_source,
                      training=replace(tr, epochs=FOUR_SINGLE.training.epochs))
        with _pinned(self._base / 'run.json', pin.record, MAX_RECORD, self._check) as raw:
            content = raw.read()
        record = cast(dict[str, Any], _validate(_decode(content), pin.run_id))
        summary = record['summary']
        if (content != _encode(record) or record['status'] != 'SUCCEEDED'
                or canonical(record['recipe']) != canonical(old.to_dict())
                or summary['method'] != old.method or summary['backend_identity'] != BACKEND
                or summary['synthetic'] is not False or summary['pool_status'] != 'READY'
                or summary['reason'] != 'COMPLETE' or set(summary['models']) != {SLOT}
                or type(summary['models'][SLOT]['parameters']) is not int
                or summary['models'][SLOT]['parameters'] != PARAMETERS):
            raise PoolSourceError
        # Compare identities in the pinned record only; do not read its old weights.
        best, reloaded = summary['best'], summary['persisted_reload']
        model_ref = f'model/{SLOT}.pt'
        file = record['artifacts'][model_ref]
        if (best['ref'] != model_ref or best['sha256'] != file['sha256']
                or not 0 < file['bytes'] <= MAX_CHECKPOINT
                or type(best['epoch']) is not int or not 1 <= best['epoch'] <= old.training.epochs
                or type(best['macro_dev_bce']) not in (int, float) or best['macro_dev_bce'] < 0
                or canonical(best) != canonical(reloaded)):
            raise PoolSourceError
        return record

    def _read(self, task: Task, split: str, pin: _PanelPin) -> PairBatch:
        groups = pin.groups
        scored = split == 'test'
        members = ['ciphertext.npy', 'labels.npy', 'metadata.npy']
        record = self._record
        expected = dict(schema=1, method=record['recipe']['method'], run_id=record['run_id'],
                        recipe=record['recipe'], backend=BACKEND, source=record['source'],
                        cipher=task.cipher, split=split, namespace=pin.namespace, digest=pin.digest,
                        rows=groups, encoding='packed-msb0-single-pair-v1')
        if scored:
            members.insert(2, 'logits.npy')  # Only the ZIP directory entry is inspected.
            expected['best'] = record['summary']['persisted_reload']
        with (_pinned(self._base / pin.ref, pin.file, groups * (25 if scored else 17)
                      + PANEL_OVERHEAD, self._check) as raw, ZipFile(raw) as archive):
            if sorted(archive.namelist()) != members:
                raise PoolSourceError
            meta = _decode(_array(archive, 'metadata', None, PANEL_OVERHEAD).tobytes())
            if canonical(meta) != canonical(expected):
                raise PoolSourceError
            self._check()
            x = _array(archive, 'ciphertext', (groups, 1, 2, 8), groups * 16)
            self._check()
            y = _array(archive, 'labels', (groups, 1), groups)
        batch = PairBatch(x, y, split, 0, pin.namespace)
        if (batch.digest != pin.digest
                or (split == 'dev' and (y[::2].any() or not (y[1::2] == 1).all()))):
            raise PoolSourceError
        self._check()
        return batch

    def _reference(self, pin: _PanelPin) -> dict[str, Any]:
        return dict(origin='reused', ref=pin.ref, **asdict(pin.file), digest=pin.digest,
                    groups=pin.groups, namespace=pin.namespace, source_record=self.source_record)

    def _view(self, only: Task | None) -> tuple[Task, ...]:
        if only is None:
            return self.recipe.tasks
        if only not in self.recipe.tasks:
            raise ValueError('POOL_TASK')
        return (only,)

    def _read_pools(self, visit: Callable[[str], None], *, only: Task | None = None
                    ) -> Iterator[tuple[Task, PairBatch, PairBatch, dict[str, Any]]]:
        if self._record:
            raise ValueError('POOL_ORDER')
        try:
            self._pool_tasks = self._view(only)
            self._record = self._manifest()
            summary = self._record['summary']
            tr, tasks = self.recipe.training, self.recipe.tasks
            old_pools = summary['pools']
            if set(old_pools) != {task.cipher for task in tasks} or set(summary['test']) != set(
                    old_pools):
                raise PoolSourceError
            # Bind TEST refs only for the view, without touching their paths.
            self._test_pins = {task.cipher: _panel_pin(
                self._record, summary['test'][task.cipher]['panel'], task, 'test',
                tr.test_groups, self.recipe.namespace) for task in self._pool_tasks}
            if only is None or only == tasks[-1]:
                visit('SKINNY')
                train, dev, source = load_reused_pool(self._skinny, self._check)
                if canonical(source) != canonical(old_pools['SKINNY']):
                    raise PoolSourceError
                yield tasks[-1], train, dev, source
                del train, dev
            if only == tasks[-1]:
                return
            train_groups = estimate(self.recipe)['fixed_train_rows_each']
            for task in (tasks[:-1] if only is None else (only,)):
                cipher = task.cipher
                visit(cipher)
                train_pin = _panel_pin(self._record, old_pools[cipher]['train'], task, 'train',
                                       train_groups, self.recipe.namespace)
                dev_pin = _panel_pin(self._record, old_pools[cipher]['dev'], task, 'dev',
                                     tr.dev_groups, self.recipe.namespace)
                yield (task, self._read(task, 'train', train_pin),
                       self._read(task, 'dev', dev_pin), dict(train=self._reference(train_pin),
                                                           dev=self._reference(dev_pin)))
        except _SOURCE_ERRORS:
            raise PoolSourceError from None

    def bind(self, visit: Callable[[str], None]) -> tuple[dict[str, PairBatch], dict[str, Any]]:
        dev: dict[str, PairBatch] = {}
        evidence: dict[str, Any] = {}
        for task, train, valid, source in self._read_pools(visit):
            dev[task.cipher], evidence[task.cipher] = valid, source
            streams, namespace = self.recipe.streams, self.recipe.namespace
            if task.cipher == 'SKINNY':
                streams = replace(PUBLIC_STREAMS, namespace=self._skinny.namespace)
                namespace = self._skinny.shuffle_namespace
            self._fixed[task.cipher] = FixedPairReplay(train, self.recipe.training, streams,
                                                     task, namespace, self._check)
            del train
        return dev, evidence

    def activate(self) -> None:
        if self._ready or set(self._fixed) != {task.cipher for task in self._pool_tasks}:
            raise ValueError('POOL_NOT_READY')
        self._ready = True

    def train(self, task: Task, cycle: int) -> PairBatch:
        if not self._ready or self._released:
            raise ValueError('POOL_NOT_READY')
        if len(self._pool_tasks) == 1 and task != self._pool_tasks[0]:
            raise ValueError('POOL_TASK')
        return self._fixed[task.cipher].train(cycle)

    def release_train(self) -> None:
        self._fixed.clear()
        self._released = True

    def load_tests(self, visit: Callable[[str], None], *, only: Task | None = None
                   ) -> dict[str, tuple[PairBatch, dict[str, Any]]]:
        if (not self._released or not self._test_pins or self._view(only) != self._pool_tasks
                or (only is not None and not self._ready)):
            raise ValueError('TEST_ORDER')
        result = {}
        try:
            for task in self._pool_tasks:
                visit(task.cipher)
                pin = self._test_pins.pop(task.cipher)
                old_best = self._record['summary']['persisted_reload']
                provenance = self._reference(pin)
                provenance['source_best'] = {k: old_best[k] for k in ('ref', 'sha256', 'epoch')}
                result[task.cipher] = (self._read(task, 'test', pin), provenance)
        except _SOURCE_ERRORS:
            raise TestSourceError from None
        return result

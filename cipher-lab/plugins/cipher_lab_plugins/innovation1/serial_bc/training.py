"""Shared serial updates, with separate legacy TEST and volume DEV-only fits."""
from __future__ import annotations

import math
from collections.abc import Callable
from hashlib import sha256
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Stopped

from ..config import Task, canonical
from ..contracts import MAX_CHECKPOINT
from ..four_single.config import BATCH_LEDGER_ROWS, ROW_BYTES_LIMIT, SLOT
from ..four_single.replay import TestSourceError
from ..four_single.training import score
from ..layout_probe.reuse import PoolSourceError
from ..resources import Resources, ResourceStop
from ..single_train.config import BACKEND, learning_rate
from ..single_train.data import PairBatch
from ..single_train.weights import bound_identity
from .config import DEV_OBSERVATION_BYTES, SerialRecipe, VolumeRecipe, estimate, official
from .data import SerialPools, VolumePools, encode_test
from .models import SerialRuntime


class _FitLedger:
    def __init__(self, store: RunStore, res: Resources, summary: dict[str, Any]) -> None:
        self.res, self.summary = res, summary
        # Resolve the live source probe at each check, not only at assembly.
        self._current_source: Callable[[], dict[str, str]] = (
            lambda: store.current_source_identity())
        self._snapshot = store.write_snapshot
        self.write_payload = store.write_payload
        self.source = store.source_snapshot()
        self.pending: list[dict[str, Any]] = summary['pending']
        self.refs: list[str] = summary['batch_refs']
        self.updates: dict[str, dict[str, int]] = summary['updates']

    def check_source(self) -> None:
        self.res.check()
        try:
            current = self._current_source()
        except ArtifactError:
            raise ResourceStop('SOURCE_CHANGED') from None
        if current != self.source:
            raise ResourceStop('SOURCE_CHANGED')
        self.res.check()

    def snapshot(self) -> None:
        self._snapshot(dict(summary=self.summary, counts=self.res.snapshot()))
        self.res.check()

    def flush(self) -> None:
        if self.pending:
            self.check_source()
            self.snapshot()
            self.refs.append(self.write_payload(f'batches-{len(self.refs):04d}', self.pending))
            self.pending.clear()
            self.snapshot()

    def audit(self, update_count: int, rows: int) -> None:
        counts = self.res.snapshot()
        expected = dict(real_blocks=0, null_words=0, optimizer_updates=update_count,
                        forward_rows=rows)
        if any(counts[f'{kind}_{unit}'] != n for unit, n in expected.items()
               for kind in ('attempted', 'completed')):
            raise ValueError('COUNT_MISMATCH')


def _train_stage(recipe: SerialRecipe, task: Task, stage: int, ledger: _FitLedger,
                 train: Callable[[Task, int], PairBatch], runtime: SerialRuntime, *,
                 after_epoch: Callable[[int, int, float], None] | None = None) -> float:
    tr, res, cipher = recipe.training, ledger.res, task.cipher
    losses = 0.
    for epoch in range(1, tr.epochs + 1):
        ledger.check_source()
        for offset in range(tr.batches_per_epoch):
            cycle = (epoch - 1) * tr.batches_per_epoch + offset
            loc = dict(phase='train', stage=stage, task=cipher, epoch=epoch, cycle=cycle)
            ledger.summary['location'] = loc
            res.check()
            batch = train(task, cycle)
            if (batch.split, batch.index, batch.groups, batch.namespace) != (
                    'train', cycle, tr.batch_groups, recipe.namespace):
                raise ValueError('TRAIN_BATCH_IDENTITY')
            lr = learning_rate(recipe, cycle)  # restart schedule, not M or Adam
            row = dict(loc, groups=batch.groups, digest=batch.digest, lr=lr,
                       attempted=False, completed=False)
            x, y = runtime.prepare(batch)
            res.reserve('optimizer_updates', 1)
            row['attempted'] = True
            ledger.pending.append(row)
            ledger.updates[cipher]['attempted'] += 1
            loss = runtime.update(SLOT, task, x, y, lr, res)
            res.complete('optimizer_updates', 1)
            ledger.updates[cipher]['completed'] += 1
            row['completed'] = True
            if not math.isfinite(loss):
                raise ValueError('UPDATE_INVALID')
            row['bce'] = loss
            if len(canonical(row)) + 1 > ROW_BYTES_LIMIT:
                raise ValueError('BATCH_LEDGER_SIZE')
            losses += loss
            del batch, x, y
            res.check()
            if len(ledger.pending) >= BATCH_LEDGER_ROWS:
                ledger.flush()
        ledger.flush()
        if after_epoch is not None:
            after_epoch(epoch, epoch * tr.batches_per_epoch, losses)
    return losses


def _fit(recipe: SerialRecipe, store: RunStore, res: Resources, pools: SerialPools,
         runtime: SerialRuntime, summary: dict[str, Any]) -> None:
    tr, tasks, slots = recipe.training, recipe.tasks, recipe.stage_slots
    ledger = _FitLedger(store, res, summary)
    totals, source = estimate(recipe), ledger.source
    assert recipe.run_id is not None
    run_id = recipe.run_id
    stages: list[dict[str, Any]] = summary['stages']
    check_source, snapshot, audit = ledger.check_source, ledger.snapshot, ledger.audit

    def at_row(start: int) -> None:
        summary['location']['start_row'] = start

    def visit_pool(cipher: str) -> None:
        summary['location'] = dict(phase='pool-source-verify', task=cipher)

    def visit_test(cipher: str) -> None:
        summary['location'] = dict(phase='test-source-verify', task=cipher)

    def identity(stage: int, value: float) -> dict[str, Any]:
        return dict(bound_identity(recipe, source, BACKEND, slots[stage-1], tr.epochs,
                                    value, slots=slots, fixed=recipe, schema=1,
                                    method=recipe.method, run_id=run_id), stage=stage,
                    stage_task=tasks[stage-1].cipher, routing_mode=recipe.routing_mode,
                    pools=summary['pools'], geometry=summary['models'][SLOT]['geometry'])

    check_source()
    summary['models'] = runtime.metadata()
    summary['pool_status'] = 'PREPARING'
    summary['test_protocol'] = dict(source=pools.source_record, prior_test_reuse=True,
                                    fresh_confirmation=False, old_logits_used=False)
    dev, summary['pools'] = pools.bind(visit_pool)
    audit(0, 0)
    check_source()
    summary['pool_status'] = 'READY'
    summary['location'] = dict(phase='pool-ready')
    snapshot()

    for stage, (task, checkpoint_slot) in enumerate(zip(tasks, slots, strict=True), 1):
        cipher = task.cipher
        losses = _train_stage(recipe, task, stage, ledger, pools.train, runtime)

        scores: dict[str, Any] = {}
        for target in tasks:
            summary['location'] = dict(phase='dev-score', stage=stage, task=target.cipher)
            scores[target.cipher], _ = score(target, dev[target.cipher], runtime,
                                              tr.eval_batch_groups, res, at_row)
        routes = runtime.routes()
        macro_bce = math.fsum(s['bce'] for s in scores.values()) / len(tasks)
        consumed = stage * totals['updates_each']
        audit(consumed, consumed * tr.batch_groups + stage * len(tasks) * tr.dev_groups)
        check_source()
        summary['location'] = dict(phase='stage-save', stage=stage, task=cipher)
        snapshot()
        raw = runtime.checkpoint(SLOT, identity(stage, macro_bce))
        if not 0 < len(raw) <= MAX_CHECKPOINT:
            raise ValueError('CHECKPOINT_SIZE')
        res.check()
        ref = store.write_model(checkpoint_slot, raw)
        observation = dict(stage=stage, task=cipher, epochs=tr.epochs, ref=ref,
                           sha256=sha256(raw).hexdigest(), bytes=len(raw), dev=scores,
                           macro_dev_bce=macro_bce, routes=routes,
                           mean_online_train_bce=losses / totals['updates_each'])
        del raw
        # Keep DEV/routes as registered durable evidence even if a later storage
        # failure reduces the run summary to RunStore's bounded emergency seal.
        evidence_ref = store.write_payload(f'stage-{stage}', observation)
        stages.append(dict(observation, evidence_ref=evidence_ref))
        snapshot()

    final = stages[-1]
    if (final['stage'] != len(slots) or final['task'] != tasks[-1].cipher
            or final['epochs'] != tr.epochs or final['ref'] != f'model/{SLOT}.pt'):
        raise ValueError('FINAL_IDENTITY')
    summary['location'] = dict(phase='persisted-final-reload')
    check_source()
    snapshot()
    del dev
    pools.release_train()
    runtime.release()
    res.check()
    raw = store.read_model(final['ref'], final['sha256'], MAX_CHECKPOINT)
    res.check()
    runtime.restore(SLOT, raw, identity(final['stage'], final['macro_dev_bce']))
    del raw
    check_source()
    reload = {k: final[k] for k in ('stage', 'task', 'ref', 'sha256')}
    reload.update(routes=runtime.routes(), routing_mode=recipe.routing_mode)
    if canonical(reload['routes']) != canonical(final['routes']):
        raise ValueError('RELOAD_ROUTE_MISMATCH')
    summary['persisted_reload'] = reload
    snapshot()

    tests = pools.load_tests(visit_test)  # verify the whole requested view before scoring
    outputs: list[tuple[Task, PairBatch, dict[str, Any], NDArray[np.float64], dict[str, Any]]] = []
    for task in tasks:
        panel, provenance = tests.pop(task.cipher)
        summary['location'] = dict(phase='test-score', task=task.cipher)
        scores_test, logits = score(task, panel, runtime, tr.eval_batch_groups, res, at_row)
        outputs.append((task, panel, scores_test, logits, provenance))
    audit(totals['optimizer_updates'], totals['forward_rows'])
    expected_updates = dict.fromkeys(('attempted', 'completed'), totals['updates_each'])
    if any(ledger.updates[t.cipher] != expected_updates for t in tasks):
        raise ValueError('UPDATE_COUNT_MISMATCH')
    entries = {}
    for task, panel, scores_test, logits, provenance in outputs:
        summary['location'] = dict(phase='test-save', task=task.cipher)
        check_source()
        snapshot()
        raw = encode_test(panel, task, recipe, source, logits=logits, final=reload,
                          reused_input=provenance)
        res.check()
        ref = store.write_dataset(f'test-{task.cipher}', raw)
        entries[task.cipher] = dict(scores=scores_test, panel=dict(origin='rescored', ref=ref,
            sha256=sha256(raw).hexdigest(), bytes=len(raw), digest=panel.digest,
            groups=panel.groups, namespace=panel.namespace, reused_input=provenance))
        del raw
    check_source()
    summary['test'] = entries
    summary['location'] = dict(phase='complete')

def _fit_volume(recipe: VolumeRecipe, ledger: _FitLedger, pools: VolumePools) -> None:
    tr, res, summary = recipe.training, ledger.res, ledger.summary
    task, = recipe.tasks
    totals = estimate(recipe)

    def visit(cipher: str) -> None:
        summary['location'] = dict(phase='pool-source-verify', task=cipher)

    ledger.check_source()
    summary['pool_status'] = 'PREPARING'
    dev, summary['pools'] = pools.bind(visit)
    panel = dev[task.cipher]
    if (panel.split, panel.index, panel.groups, panel.namespace, panel.digest) != (
            'dev', 0, tr.dev_groups, recipe.namespace, recipe.common_dev.digest):
        raise PoolSourceError
    ledger.audit(0, 0)
    ledger.check_source()
    summary['pool_status'] = 'READY'
    summary['location'] = dict(phase='pool-ready')
    ledger.snapshot()
    runtime = SerialRuntime(recipe, res.check)
    summary['models'] = runtime.metadata()  # geometry only; no checkpoint/weights
    observations: list[dict[str, Any]] = summary['dev_observations']

    def at_row(start: int) -> None:
        summary['location']['start_row'] = start

    def observe(epoch: int, updates: int, losses: float) -> None:
        if updates not in recipe.dev_updates:
            return
        previous = len(observations)
        if updates != recipe.dev_updates[previous]:
            raise ValueError('VOLUME_OBSERVATION_ORDER')
        train_rows = updates * tr.batch_groups
        ledger.audit(updates, train_rows + previous * tr.dev_groups)
        summary['location'] = dict(phase='dev-score', task=task.cipher,
                                   epoch=epoch, updates=updates)
        scores, _ = score(task, panel, runtime, tr.eval_batch_groups, res, at_row)
        observation = dict(updates=updates, epoch=epoch, task=task.cipher, dev=scores,
                           routes=runtime.routes(), mean_online_train_bce=losses / updates)
        if len(canonical(observation)) + 1 > DEV_OBSERVATION_BYTES:
            raise ValueError('DEV_OBSERVATION_SIZE')
        ledger.audit(updates, train_rows + (previous + 1) * tr.dev_groups)
        ledger.check_source()
        summary['location'] = dict(phase='dev-save', task=task.cipher, epoch=epoch, updates=updates)
        ledger.snapshot()
        ref = ledger.write_payload(f'dev-{updates}', observation)
        observations.append(dict(observation, evidence_ref=ref))
        ledger.snapshot()

    _train_stage(recipe, task, 1, ledger, pools.train, runtime, after_epoch=observe)
    pools.release_train()
    ledger.audit(totals['optimizer_updates'], totals['forward_rows'])
    if (tuple(o['updates'] for o in observations) != recipe.dev_updates
            or ledger.updates[task.cipher] != dict.fromkeys(('attempted', 'completed'),
                                                          totals['optimizer_updates'])):
        raise ValueError('VOLUME_COUNT_MISMATCH')
    ledger.check_source()
    summary['location'] = dict(phase='complete')


def execute(recipe: SerialRecipe, store: RunStore, res: Resources, *,
            launch: dict[str, str]) -> None:
    official(recipe, store._run_id)
    _execute(recipe, store, res, launch=launch)


def _execute(recipe: SerialRecipe, store: RunStore, res: Resources, *, launch: dict[str, str],
             pools: SerialPools | None = None, runtime: SerialRuntime | None = None) -> None:
    summary: dict[str, Any] = dict(method=recipe.method, backend_identity=BACKEND, synthetic=False,
        scientific_validation=False, launch=launch, location=dict(phase='assembly'),
        pools={}, pool_status='NOT_READY', updates={t.cipher: dict(attempted=0, completed=0)
        for t in recipe.tasks}, pending=[], batch_refs=[], retries=0)
    if isinstance(recipe, VolumeRecipe):
        summary['dev_observations'] = []
    else:
        summary.update(stages=[], test={})
    status = 'FAILED'
    try:
        if estimate(recipe)['artifact_peak_bound_bytes'] > recipe.resources.artifact_bytes:
            raise ValueError('ARTIFACT_BOUND')
        store._failure_evidence_key = 'pending'
        store.write_snapshot(dict(summary=summary, counts=res.snapshot()))
        res.check()
        if isinstance(recipe, VolumeRecipe):
            if pools is not None or runtime is not None:
                raise ValueError('VOLUME_ASSEMBLY')
            _fit_volume(recipe, _FitLedger(store, res, summary), VolumePools(recipe, res.check))
        else:
            if pools is None:
                pools = SerialPools(recipe, res.check)
            if runtime is None:
                runtime = SerialRuntime(recipe, res.check)
            _fit(recipe, store, res, pools, runtime, summary)
        summary['reason'] = 'COMPLETE'
        status = 'SUCCEEDED'
    except ArtifactError:
        return  # RunStore owns storage-failure sealing.
    except TestSourceError:
        summary['reason'] = 'TEST_SOURCE_INVALID'
    except PoolSourceError:
        summary['reason'] = 'POOL_SOURCE_INVALID'
    except (ResourceStop, Stopped, KeyboardInterrupt) as error:
        status = 'STOPPED'
        summary['reason'] = 'INTERRUPTED' if isinstance(error, KeyboardInterrupt) else error.reason
    except Exception:
        summary['reason'] = 'EXECUTION_FAILED'
    try:
        store.finalize(dict(status=status, summary=summary, counts=res.snapshot()))
    except ArtifactError:
        return

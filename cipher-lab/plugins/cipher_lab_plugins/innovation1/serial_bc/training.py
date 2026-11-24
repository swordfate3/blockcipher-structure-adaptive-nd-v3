"""One continuous Adam fit, immutable stage observations, disk-only final TEST."""
from __future__ import annotations

import math
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
from .config import STAGE_SLOTS, SerialRecipe, estimate, official
from .data import SerialPools, encode_test
from .models import SerialRuntime


def _fit(recipe: SerialRecipe, store: RunStore, res: Resources, pools: SerialPools,
         runtime: SerialRuntime, summary: dict[str, Any]) -> None:
    tr, tasks = recipe.training, recipe.tasks
    totals, source = estimate(recipe), store.source_snapshot()
    assert recipe.run_id is not None
    run_id = recipe.run_id
    pending: list[dict[str, Any]] = summary['pending']
    stages: list[dict[str, Any]] = summary['stages']
    refs: list[str] = summary['batch_refs']
    updates: dict[str, dict[str, int]] = summary['updates']

    def check_source() -> None:
        res.check()
        try:
            current = store.current_source_identity()
        except ArtifactError:
            raise ResourceStop('SOURCE_CHANGED') from None
        if current != source:
            raise ResourceStop('SOURCE_CHANGED')
        res.check()

    def snapshot() -> None:
        store.write_snapshot(dict(summary=summary, counts=res.snapshot()))
        res.check()

    def flush() -> None:
        if pending:
            check_source()
            snapshot()
            refs.append(store.write_payload(f'batches-{len(refs):04d}', pending))
            pending.clear()
            snapshot()

    def audit(update_count: int, rows: int) -> None:
        counts = res.snapshot()
        expected = dict(real_blocks=0, null_words=0, optimizer_updates=update_count,
                        forward_rows=rows)
        if any(counts[f'{kind}_{unit}'] != n for unit, n in expected.items()
               for kind in ('attempted', 'completed')):
            raise ValueError('COUNT_MISMATCH')

    def at_row(start: int) -> None:
        summary['location']['start_row'] = start

    def visit_pool(cipher: str) -> None:
        summary['location'] = dict(phase='pool-source-verify', task=cipher)

    def visit_test(cipher: str) -> None:
        summary['location'] = dict(phase='test-source-verify', task=cipher)

    def identity(stage: int, value: float) -> dict[str, Any]:
        return dict(bound_identity(recipe, source, BACKEND, STAGE_SLOTS[stage-1], tr.epochs,
                                    value, slots=STAGE_SLOTS, fixed=recipe, schema=1,
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

    for stage, (task, checkpoint_slot) in enumerate(zip(tasks, STAGE_SLOTS, strict=True), 1):
        cipher = task.cipher
        losses = 0.
        for epoch in range(1, tr.epochs + 1):
            check_source()
            for offset in range(tr.batches_per_epoch):
                cycle = (epoch - 1) * tr.batches_per_epoch + offset
                loc = dict(phase='train', stage=stage, task=cipher, epoch=epoch, cycle=cycle)
                summary['location'] = loc
                res.check()
                batch = pools.train(task, cycle)
                if (batch.split, batch.index, batch.groups, batch.namespace) != (
                        'train', cycle, tr.batch_groups, recipe.namespace):
                    raise ValueError('TRAIN_BATCH_IDENTITY')
                lr = learning_rate(recipe, cycle)  # restart schedule, not M or Adam
                row = dict(loc, groups=batch.groups, digest=batch.digest, lr=lr,
                           attempted=False, completed=False)
                x, y = runtime.prepare(batch)
                res.reserve('optimizer_updates', 1)
                row['attempted'] = True
                pending.append(row)
                updates[cipher]['attempted'] += 1
                loss = runtime.update(SLOT, task, x, y, lr, res)
                res.complete('optimizer_updates', 1)
                updates[cipher]['completed'] += 1
                row['completed'] = True
                if not math.isfinite(loss):
                    raise ValueError('UPDATE_INVALID')
                row['bce'] = loss
                if len(canonical(row)) + 1 > ROW_BYTES_LIMIT:
                    raise ValueError('BATCH_LEDGER_SIZE')
                losses += loss
                del batch, x, y
                res.check()
                if len(pending) >= BATCH_LEDGER_ROWS:
                    flush()
            flush()

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
    if final['stage'] != len(STAGE_SLOTS) or final['ref'] != f'model/{SLOT}.pt':
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

    tests = pools.load_tests(visit_test)  # verify all four before any TEST scoring
    outputs: list[tuple[Task, PairBatch, dict[str, Any], NDArray[np.float64], dict[str, Any]]] = []
    for task in tasks:
        panel, provenance = tests.pop(task.cipher)
        summary['location'] = dict(phase='test-score', task=task.cipher)
        scores_test, logits = score(task, panel, runtime, tr.eval_batch_groups, res, at_row)
        outputs.append((task, panel, scores_test, logits, provenance))
    audit(totals['optimizer_updates'], totals['forward_rows'])
    if any(updates[t.cipher] != dict.fromkeys(('attempted', 'completed'), totals['updates_each'])
           for t in tasks):
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


def execute(recipe: SerialRecipe, store: RunStore, res: Resources, *,
            launch: dict[str, str]) -> None:
    official(recipe, store._run_id)
    _execute(recipe, store, res, launch=launch)


def _execute(recipe: SerialRecipe, store: RunStore, res: Resources, *, launch: dict[str, str],
             pools: SerialPools | None = None, runtime: SerialRuntime | None = None) -> None:
    summary: dict[str, Any] = dict(method=recipe.method, backend_identity=BACKEND, synthetic=False,
        scientific_validation=False, launch=launch, location=dict(phase='assembly'),
        pools={}, pool_status='NOT_READY', updates={t.cipher: dict(attempted=0, completed=0)
        for t in recipe.tasks}, pending=[], batch_refs=[], stages=[], test={}, retries=0)
    status = 'FAILED'
    try:
        if estimate(recipe)['artifact_peak_bound_bytes'] > recipe.resources.artifact_bytes:
            raise ValueError('ARTIFACT_BOUND')
        store._failure_evidence_key = 'pending'
        store.write_snapshot(dict(summary=summary, counts=res.snapshot()))
        res.check()
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

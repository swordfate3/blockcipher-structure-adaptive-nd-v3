"""One M, fixed four-task replay, one disk-best barrier before TEST."""
from __future__ import annotations

import math
import traceback
from collections.abc import Callable
from hashlib import sha256
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Stopped
from cipher_lab.training.early_stopping import EarlyStopping

from ..config import Task, canonical
from ..contracts import MAX_CHECKPOINT
from ..layout_probe.reuse import PoolSourceError
from ..resources import Resources, ResourceStop
from ..scoring import bce, predict
from ..single_train.config import BACKEND, learning_rate
from ..single_train.data import PairBatch
from ..single_train.weights import bound_identity
from .config import (
    BATCH_LEDGER_ROWS,
    PROGRESS_CHUNKS,
    ROW_BYTES_LIMIT,
    SLOT,
    FourSingleRecipe,
    ReplayRecipe,
    estimate,
    official,
)
from .data import Pools, encode_panel
from .interleave_config import FeistelWordRecipe, FeistelWordRetryRecipe, InterleaveRecipe
from .replay import ReplayPools, TestSourceError
from .runtime import Runtime


def assemble(recipe: FourSingleRecipe, resources: Resources) -> tuple[Pools, Runtime]:
    from secrets import token_bytes

    from cipher_lab.ciphers import present, simeck, simon, skinny

    return (Pools(recipe, dict(SIMECK=simeck.encrypt_many, PRESENT=present.encrypt_many,
                                  SIMON=simon.encrypt_many, SKINNY=skinny.encrypt_many),
                      token_bytes, resources), Runtime(recipe, resources.check))


def score(task: Task, panel: PairBatch, runtime: Runtime, chunk: int, res: Resources,
          at_row: Callable[[int], None]) -> tuple[dict[str, Any], NDArray[np.float64]]:
    """Evaluate each row once, including the final short (balanced) batch."""
    chunks = []
    for start in range(0, panel.groups, chunk):
        at_row(start)
        res.check()
        batch = panel.slice(start, min(start + chunk, panel.groups))
        x, y = runtime.prepare(batch)
        chunks.append(predict(runtime, SLOT, task, x, y, res))
    logits: NDArray[np.float64] = np.concatenate(chunks)
    return dict(groups=panel.groups, bce=bce(panel.labels, logits),
                accuracy=float(np.mean((logits >= 0) == panel.labels))), logits


def _fit(recipe: FourSingleRecipe, pools: Pools | ReplayPools, runtime: Runtime, store: RunStore,
         res: Resources, summary: dict[str, Any]) -> None:
    tr, tasks = recipe.training, recipe.tasks
    totals = estimate(recipe)
    source = store.source_snapshot()
    best: dict[str, Any] = summary['best']
    pending: list[dict[str, Any]] = summary['pending']
    batch_refs: list[str] = summary['batch_refs']
    bindings: dict[str, Any] = summary['pools']
    updates: dict[str, dict[str, int]] = summary['updates']
    test_results: dict[str, Any] = summary['test']
    interleave = isinstance(recipe, InterleaveRecipe)

    def route_observation() -> dict[str, Any]:
        routes = runtime.routes()
        observation: dict[str, Any] = {'routes': routes}
        if isinstance(recipe, InterleaveRecipe):
            observation['nibble_lookup'] = recipe.nibble_lookup(routes)
        if isinstance(recipe, FeistelWordRecipe):
            observation['word_lookup'] = recipe.word_lookup(routes)
        return observation
    tracker = (EarlyStopping((SLOT,), recipe.early_stopping, epoch_ceiling=tr.epochs)
               if isinstance(recipe, ReplayRecipe) else None)
    cosine_epochs = recipe.cosine_epochs if isinstance(recipe, ReplayRecipe) else None

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
            snapshot()  # persist paid counts and bounded pending evidence before artifact writes
            batch_refs.append(store.write_payload(f'batches-{len(batch_refs):04d}', pending))
            pending.clear()
            snapshot()

    def progress(chunks: int) -> None:
        summary['location']['chunks_completed'] = chunks
        if chunks % PROGRESS_CHUNKS == 0:
            snapshot()

    def audit(**expected: int) -> None:
        counts = res.snapshot()
        if any(counts[f'{kind}_{unit}'] != n for unit, n in expected.items()
               for kind in ('attempted', 'completed')):
            raise ValueError('COUNT_MISMATCH')

    def save_panel(task: Task, panel: PairBatch, **kwargs: Any) -> dict[str, Any]:
        summary['location'] = dict(phase='panel-save', task=task.cipher, split=panel.split)
        check_source()
        snapshot()
        raw = encode_panel(panel, task, recipe, source, BACKEND, **kwargs)
        res.check()
        ref = store.write_dataset(f'{panel.split}-{task.cipher}', raw)
        entry = dict(origin='fresh', ref=ref, sha256=sha256(raw).hexdigest(), bytes=len(raw),
                     digest=panel.digest, groups=panel.groups, namespace=panel.namespace)
        if 'reused_input' in kwargs:
            entry.update(origin='rescored', reused_input=kwargs['reused_input'])
        return entry

    def at_row(start: int) -> None:
        summary['location']['start_row'] = start

    def identity(epoch: int, value: float) -> dict[str, Any]:
        assert recipe.run_id is not None
        return dict(bound_identity(recipe, source, BACKEND, SLOT, epoch, value,
                                    slots=(SLOT,), fixed=recipe, schema=1,
                                    method=recipe.method, run_id=recipe.run_id),
                    family='four-task-single-pair-shared-M', pools=bindings,
                    geometry=summary['models'][SLOT]['geometry'])

    check_source()
    summary['models'] = runtime.metadata()
    summary['pool_status'] = 'PREPARING'
    summary['location'] = dict(phase='pool-source-verify', task='SKINNY')
    snapshot()
    if isinstance(pools, ReplayPools):
        def visit_pool(cipher: str) -> None:
            summary['location'] = dict(phase='pool-source-verify', task=cipher)

        dev, evidence = pools.bind(visit_pool)
        bindings.update(evidence)
        audit(real_blocks=0, optimizer_updates=0, forward_rows=0, null_words=0)
    else:
        # Verify the mandatory archive before collecting any fresh pool.
        skinny_dev, bindings['SKINNY'] = pools.bind_reused()
        dev = {'SKINNY': skinny_dev}
        del skinny_dev
        audit(real_blocks=0, optimizer_updates=0, forward_rows=0, null_words=0)
        snapshot()
        for i, task in enumerate(tasks[:-1], 1):
            summary['location'] = dict(phase='pool-generate', task=task.cipher, chunks_completed=0)
            snapshot()
            train, dev[task.cipher] = pools.collect(task, progress)
            audit(real_blocks=2 * i * (totals['fixed_train_rows_each'] + tr.dev_groups),
                  optimizer_updates=0, forward_rows=0, null_words=0)
            bindings[task.cipher] = dict(train=save_panel(task, train))
            del train
            bindings[task.cipher]['dev'] = save_panel(task, dev[task.cipher])
            snapshot()
    if set(bindings) != {task.cipher for task in tasks}:
        raise ValueError('POOL_BINDING')
    check_source()
    summary['pool_status'] = 'READY'
    summary['location'] = dict(phase='pool-ready')
    snapshot()
    pools.activate()

    completed_epochs = 0
    for epoch in range(1, tr.epochs + 1):
        check_source()
        losses = {task.cipher: 0. for task in tasks}
        for offset in range(tr.batches_per_epoch):
            cycle = (epoch - 1) * tr.batches_per_epoch + offset
            lr = learning_rate(recipe, cycle, cosine_epochs=cosine_epochs)
            for task in tasks:
                cipher = task.cipher
                loc = dict(phase='train', task=cipher, epoch=epoch, cycle=cycle)
                summary['location'] = loc
                res.check()
                batch = pools.train(task, cycle)
                if (batch.split, batch.index, batch.groups, batch.namespace) != (
                        'train', cycle, tr.batch_groups, dev[cipher].namespace):
                    raise ValueError('TRAIN_BATCH_IDENTITY')
                row = dict(loc, groups=batch.groups, digest=batch.digest, lr=lr,
                           attempted=False, completed=False)
                pending.append(row)
                x, y = runtime.prepare(batch)
                res.reserve('optimizer_updates', 1)
                row['attempted'] = True
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
                losses[cipher] += loss
                del batch, x, y
                res.check()
                if len(pending) >= BATCH_LEDGER_ROWS:
                    flush()
            if cycle == 0:
                snapshot()
        flush()
        scores = {}
        for task in tasks:
            summary['location'] = dict(phase='dev-score', epoch=epoch, task=task.cipher)
            scores[task.cipher], dev_logits = score(task, dev[task.cipher], runtime,
                                                   tr.eval_batch_groups, res, at_row)
            del dev_logits
        observation = route_observation()  # public router only; no data forward rows
        value = math.fsum(v['bce'] for v in scores.values()) / len(tasks)
        summary['curves'].append(dict(epoch=epoch, dev=scores, macro_dev_bce=value,
                     train_bce={k: v / tr.batches_per_epoch for k, v in losses.items()},
                     **observation))
        consumed = epoch * tr.batches_per_epoch * len(tasks)
        audit(real_blocks=totals['fresh_pool_real_blocks'], null_words=0,
              optimizer_updates=consumed,
              forward_rows=consumed * tr.batch_groups + epoch * len(tasks) * tr.dev_groups)
        if interleave:
            summary['progress'] = {'completed_dev_epochs': epoch}
        snapshot()
        if not best or value < best['macro_dev_bce']:
            summary['location'] = dict(phase='best-save', epoch=epoch)
            check_source()
            raw = runtime.checkpoint(SLOT, identity(epoch, value))
            res.check()
            ref = store.write_model(SLOT, raw, replace=bool(best))
            best.update(ref=ref, sha256=sha256(raw).hexdigest(), epoch=epoch,
                        macro_dev_bce=value, **observation)
            del raw
            snapshot()
        completed_epochs = epoch
        if tracker is not None and (reason := tracker.observe(epoch, {SLOT: value})) is not None:
            summary['training_termination'] = dict(reason=reason, completed_epochs=epoch)
            break

    summary['location'] = dict(phase='persisted-best-reload')
    check_source()
    if tracker is not None:
        snapshot()
    if best.get('ref') != f'model/{SLOT}.pt':
        raise ValueError('BEST_REF')
    del dev
    pools.release_train()
    runtime.release()
    res.check()
    raw = store.read_model(best['ref'], best['sha256'], MAX_CHECKPOINT)
    res.check()
    runtime.restore(SLOT, raw, identity(best['epoch'], best['macro_dev_bce']))
    del raw
    check_source()
    reloaded = route_observation()
    if interleave and any(best.get(k) != v for k, v in reloaded.items()):
        raise ValueError('RELOADED_ROUTE_IDENTITY')
    summary['persisted_reload'] = {**best, **reloaded}
    snapshot()
    replay_tests: dict[str, tuple[PairBatch, dict[str, Any]]] = {}
    if isinstance(pools, ReplayPools):
        summary['location'] = dict(phase='test-source-verify')
        snapshot()

        def visit_test(cipher: str) -> None:
            summary['location'] = dict(phase='test-source-verify', task=cipher)

        # Verify all four prior TESTs before any scoring. A missing or corrupt
        # file cannot produce a misleading subset of TEST scores.
        replay_tests = pools.load_tests(visit_test)
    rescored: list[tuple[Task, PairBatch, dict[str, Any], NDArray[np.float64],
                         dict[str, Any]]] = []
    for i, task in enumerate(tasks, 1):
        if isinstance(pools, ReplayPools):
            panel, prior = replay_tests.pop(task.cipher)
            provenance = dict(reused_input=prior)
        else:
            summary['location'] = dict(phase='test-generate', task=task.cipher, chunks_completed=0)
            snapshot()
            panel = pools.test(task, progress)
            provenance = {}
        summary['location'] = dict(phase='test-score', task=task.cipher)
        audit(real_blocks=totals['fresh_pool_real_blocks']
              + i * (totals['test_real_blocks'] // len(tasks)))
        scores_test, logits = score(task, panel, runtime, tr.eval_batch_groups, res, at_row)
        if isinstance(pools, ReplayPools):
            # Keep only the bounded four TEST panels and new logits in memory;
            # never publish a subset of scores if one task fails to score.
            rescored.append((task, panel, scores_test, logits, provenance))
        else:
            test_results[task.cipher] = dict(scores=scores_test, panel=save_panel(
                task, panel, logits=logits, best=summary['persisted_reload']))
            snapshot()
        del panel, logits
    check_source()
    cycles = completed_epochs * tr.batches_per_epoch
    audit(real_blocks=totals['real_blocks'], null_words=0, optimizer_updates=cycles * len(tasks),
          forward_rows=len(tasks) * (cycles * tr.batch_groups + completed_epochs * tr.dev_groups
                                     + tr.test_groups))
    if any(updates[t.cipher] != dict.fromkeys(('attempted', 'completed'), cycles) for t in tasks):
        raise ValueError('UPDATE_COUNT_MISMATCH')
    if rescored:
        entries = {}
        for task, panel, scores_test, logits, provenance in rescored:
            entries[task.cipher] = dict(scores=scores_test, panel=save_panel(
                task, panel, logits=logits, best=summary['persisted_reload'], **provenance))
        check_source()
        test_results.update(entries)


def execute(recipe: FourSingleRecipe, store: RunStore, res: Resources, *,
            launch: dict[str, str]) -> None:
    official(recipe, store._run_id)
    summary: dict[str, Any] = dict(method=recipe.method, backend_identity=BACKEND, synthetic=False,
        scientific_validation=False, launch=launch, location=dict(phase='assembly'),
        pools={}, pool_status='NOT_READY', updates={t.cipher: dict(attempted=0, completed=0)
        for t in recipe.tasks}, pending=[], batch_refs=[], curves=[], best={}, test={})
    interleave = isinstance(recipe, InterleaveRecipe)
    if interleave:
        summary['progress'] = {'completed_dev_epochs': 0}
    status = 'FAILED'
    try:
        if estimate(recipe)['artifact_peak_bound_bytes'] > recipe.resources.artifact_bytes:
            raise ValueError('ARTIFACT_BOUND')
        store._failure_evidence_key = 'progress' if interleave else 'pending'
        store.write_snapshot(dict(summary=summary, counts=res.snapshot()))
        res.check()
        pools: Pools | ReplayPools
        runtime: Runtime
        if isinstance(recipe, ReplayRecipe):
            if isinstance(recipe, InterleaveRecipe):
                from .interleave_runtime import InterleaveRuntime, assemble_pools

                pools = assemble_pools(recipe, res.check)
                runtime = InterleaveRuntime(recipe, res.check)
            else:
                pools = ReplayPools(recipe, res.check)
                runtime = Runtime(recipe, res.check)
            summary['test_protocol'] = dict(source=pools.source_record, prior_test_reuse=True,
                                            paired_comparison=False, fresh_confirmation=False)
        else:
            pools, runtime = assemble(recipe, res)
        _fit(recipe, pools, runtime, store, res, summary)
        if interleave:
            res.check()
        summary['reason'] = 'COMPLETE'
        status = 'SUCCEEDED'
    except ArtifactError:
        return
    except TestSourceError:
        summary['reason'] = 'TEST_SOURCE_INVALID'
    except PoolSourceError:
        summary['reason'] = 'POOL_SOURCE_INVALID'
    except (ResourceStop, Stopped, KeyboardInterrupt) as error:
        status = 'STOPPED'
        summary['reason'] = 'INTERRUPTED' if isinstance(error, KeyboardInterrupt) else error.reason
    except Exception:
        if isinstance(recipe, FeistelWordRetryRecipe):
            # Diagnostics must not prevent the owner from sealing the failed run.
            try:
                traceback.print_exc()
            except Exception:
                pass
        summary['reason'] = 'EXECUTION_FAILED'
    if interleave:
        summary['counts_scope'] = 'PRE_OWNER_SEAL'
    try:
        store.finalize(dict(status=status, summary=summary, counts=res.snapshot()))
    except ArtifactError:
        return

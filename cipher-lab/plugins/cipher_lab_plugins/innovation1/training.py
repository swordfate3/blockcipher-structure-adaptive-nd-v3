"""Synchronous same-batch loop; dependency injection ends at runtime/provider.

No private state is logged, no retries, no TRAIN replay or extra TRAIN scoring.
ArtifactError is storage-owned: never finalize again after a storage seal.
"""
from __future__ import annotations

import math
from hashlib import sha256
from typing import Any

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Stopped

from .checkpoint import MAX_CHECKPOINT, SLOTS, identity
from .config import Recipe, learning_rate, require_official
from .data import derive_seed
from .dataset import encode_panel
from .metrics import compare
from .resources import ResourceStop
from .scoring import bce, score_panel, tensor_digest


def run(recipe: Recipe, provider: Any, models: Any, store: RunStore, resources: Any,
        *, backend: str = 'demo', launch: dict[str, str] | None = None) -> dict[str, Any]:
    if backend not in ('demo', 'ciphers') or models.backend != backend:
        raise ValueError('BACKEND_MISMATCH')
    if backend == 'ciphers':
        # Artificial components cannot enter the production identity via DI.
        from .data import Provider
        from .models import TorchRuntime
        if type(provider) is not Provider or type(models) is not TorchRuntime:
            raise ValueError('PRODUCTION_DEPENDENCY_MISMATCH')
        require_official(recipe)
    summary: dict[str, Any] = dict(method=recipe.method, backend_identity=backend,
        synthetic=backend == 'demo', scientific_validation=False, best={}, curves=[],
        batch_refs=[], pending=[], paired_completed=0, compare={},
        count_semantics='SIMULATED manual work' if backend == 'demo' else 'production attempts')
    if launch is not None:
        summary['launch'] = launch
    best: dict[str, dict[str, Any]] = summary['best']
    pending: list[dict[str, Any]] = summary['pending']
    source = store.source_snapshot()
    location: dict[str, Any] = {}
    def check_identity() -> None:
        resources.check()
        try:
            current = store.current_source_identity()
        except ArtifactError:
            raise ResourceStop('SOURCE_CHANGED') from None
        if current != source:
            raise ResourceStop('SOURCE_CHANGED')
    def save() -> None:
        store.write_snapshot({'summary': summary, 'counts': resources.snapshot()})
    def flush() -> None:
        if pending:
            save()  # commit paid counts before any sealing payload write
            ref = store.write_payload(f'batches-{len(summary["batch_refs"]):05d}', list(pending))
            summary['batch_refs'].append(ref)
            pending.clear()
    try:
        summary['models'] = models.metadata()
        policy = dict(vars(recipe.resources))
        if launch is not None:
            policy['request'] = dict(launch)  # never overwrite the saved request identity
        store.write_snapshot({'recipe': recipe.to_dict(), 'policy': policy})
        if hasattr(store, '_failure_evidence_key'):
            setattr(store, '_failure_evidence_key', 'pending')  # optional release-core capability
        save()
        check_identity()
        dev = provider.prepare_dev()
        for task in recipe.tasks:
            raw = encode_panel(dev[task.cipher], recipe, source, backend)
            if task == recipe.tasks[0]:
                save()  # all DEV collection has already been paid
            store.write_dataset(f'dev-{task.cipher}', raw)
        for epoch in range(1, recipe.training.epochs + 1):
            train_sum = {t.cipher: [0., 0.] for t in recipe.tasks}
            for offset in range(recipe.training.batches_per_epoch):
                cycle = (epoch - 1) * recipe.training.batches_per_epoch + offset
                check_identity()
                lr = learning_rate(recipe, cycle)
                for task in recipe.tasks:
                    location = dict(epoch=epoch, cycle=cycle, task=task.cipher, phase='train')
                    resources.check()
                    batch = provider.train(task, cycle)  # exactly once
                    x, y = models.prepare(batch)  # exactly once, same tensors for both arms
                    digest = tensor_digest(x, y)
                    entry = dict(location, batch_digest=batch.digest, input_digest=digest,
                                 shared_completed=False, dedicated_completed=False)
                    pending.append(entry)
                    for arm, slot in enumerate(('M-best', f'D-{task.cipher}-best')):
                        location['slot'] = slot
                        before = tensor_digest(x, y)
                        if before != digest:
                            raise ValueError('BATCH_MUTATED')
                        entry['shared_digest' if arm == 0 else 'dedicated_digest'] = before
                        resources.reserve('optimizer_updates', 1)
                        loss = models.update(slot, task, x, y, lr, resources)
                        # update returned only after step; preserve partial completion even
                        # if an input mutation/post-update resource check then stops the run.
                        resources.complete('optimizer_updates', 1)
                        entry['shared_completed' if arm == 0 else 'dedicated_completed'] = True
                        if not math.isfinite(loss) or tensor_digest(x, y) != digest:
                            raise ValueError('UPDATE_INVALID')
                        train_sum[task.cipher][arm] += loss
                        resources.check()
                    summary['paired_completed'] += 1
                    del x, y, batch
                    if len(pending) >= 256:
                        flush()
                        save()
            location = dict(epoch=epoch, phase='dev')
            scores: dict[str, dict[str, float]] = {}
            for task in recipe.tasks:
                slot = f'D-{task.cipher}-best'
                panel = dev[task.cipher]
                logits = score_panel(panel, models, ('M-best', slot),
                                     recipe.training.eval_batch_groups, resources)
                scores[task.cipher] = {s: bce(panel.labels, z) for s, z in logits.items()}
            selection = {'M-best': sum(v['M-best'] for v in scores.values()) / 4,
                         **{f'D-{t.cipher}-best': scores[t.cipher][f'D-{t.cipher}-best']
                            for t in recipe.tasks}}
            # Preserve completed science before a potentially sealing best write.
            if not pending:
                save()
            flush()
            for slot, score in selection.items():
                if slot not in best or score < best[slot]['metadata']['score']:
                    check_identity()
                    meta = identity(recipe, source, backend, slot, epoch, score)
                    raw = models.checkpoint(slot, meta)
                    ref = store.write_model(slot, raw, replace=slot in best)
                    best[slot] = dict(ref=ref, sha256=sha256(raw).hexdigest(), metadata=meta)
            summary['curves'].append(dict(epoch=epoch, dev=scores,
                train_bce={t: [v / recipe.training.batches_per_epoch for v in values]
                           for t, values in train_sum.items()}))
            flush()
            save()
        location = dict(phase='persisted-best-reload')
        check_identity()
        if set(best) != set(SLOTS):
            raise ValueError('BEST_INCOMPLETE')
        models.release()  # discard initial graph/Adam before reconstructing selected models
        for slot in SLOTS:
            saved = best[slot]
            raw = store.read_model(saved['ref'], saved['sha256'], MAX_CHECKPOINT)
            models.restore(slot, raw, saved['metadata'])
        summary['persisted_reloads'] = list(SLOTS)
        check_identity()
        for task in recipe.tasks:
            location = dict(phase='test', task=task.cipher)
            panel = provider.test(task)  # only AFTER all five persisted strict reloads
            save()
            store.write_dataset(f'test-{task.cipher}', encode_panel(panel, recipe, source, backend))
            slot = f'D-{task.cipher}-best'
            logits = score_panel(panel, models, ('M-best', slot),
                                 recipe.training.eval_batch_groups, resources)
            seed = derive_seed(recipe, task, 'test', 0, 'bootstrap')
            comparison = compare(panel.labels, logits['M-best'], logits[slot],
                                  replicates=recipe.statistics.replicates, seed=seed)
            save()  # preserve paid forwards but do not expose TEST before its payload
            store.write_payload(f'predictions-{task.cipher}', dict(labels=panel.labels.tolist(),
                shared=logits['M-best'].tolist(), dedicated=logits[slot].tolist()))
            summary['compare'][task.cipher] = comparison
            save()
        check_identity()
        summary['reason'] = 'COMPLETE'
        result = dict(status='SUCCEEDED', summary=summary, counts=resources.snapshot())
        store.finalize(result)
        return result
    except ArtifactError:
        # The owner has already sealed (possibly no terminal could be persisted).
        return dict(status='FAILED', reason='ARTIFACT_FAILED', summary=summary,
                    counts=resources.snapshot())
    except (Exception, KeyboardInterrupt) as error:
        stopped = isinstance(error, (Stopped, ResourceStop, KeyboardInterrupt))
        summary['reason'] = (error.reason if isinstance(error, (Stopped, ResourceStop))
                             else 'INTERRUPTED' if isinstance(error, KeyboardInterrupt)
                             else 'EXECUTION_FAILED')
        summary['failure_location'] = location
        result = dict(status='STOPPED' if stopped else 'FAILED', summary=summary,
                      counts=resources.snapshot())
        try:
            store.finalize(result)
        except ArtifactError:
            pass  # no retry/second finalize after ownership/media failure
        return result

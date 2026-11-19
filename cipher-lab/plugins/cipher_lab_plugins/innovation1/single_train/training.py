"""Online M-then-D training and a disk-only two-best barrier before TEST."""
from __future__ import annotations

import math
from hashlib import sha256
from typing import Any

import numpy as np

from cipher_lab.artifacts import ArtifactError, RunStore

from ..contracts import MAX_CHECKPOINT
from ..resources import Resources, ResourceStop
from ..scoring import bce, predict, tensor_digest
from .config import SLOTS, TASK, SingleRecipe, estimate, learning_rate
from .data import PairBatch, encode_panel
from .weights import identity


def train(recipe: SingleRecipe, provider: Any, runtime: Any, store: RunStore,
          resources: Resources, summary: dict[str, Any], backend: str) -> None:
    """Compute and snapshot only; application owns the single terminal transition."""
    tr = recipe.training
    source = store.source_snapshot()
    best: dict[str, Any] = summary['best']
    pending: list[dict[str, Any]] = summary['pending']
    refs: list[str] = summary['batch_refs']
    check = resources.check

    def check_source() -> None:
        check()
        try:
            current = store.current_source_identity()
        except ArtifactError:
            raise ResourceStop('SOURCE_CHANGED') from None
        if current != source:
            raise ResourceStop('SOURCE_CHANGED')
        check()

    def snapshot() -> None:
        check()
        store.write_snapshot(dict(summary=summary, counts=resources.snapshot()))
        check()

    def flush() -> None:
        if pending:
            snapshot()  # a sealing storage fault must not lose committed update evidence
            refs.append(store.write_payload(f'batches-{len(refs):04d}', pending))
            pending.clear()
            snapshot()

    def score(panel: PairBatch) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
        chunks: dict[str, list[Any]] = {s: [] for s in SLOTS}
        for start in range(0, panel.groups, tr.eval_batch_groups):
            check()
            loc = summary['location']
            loc['start_group'] = start
            batch = panel.slice(start, start + tr.eval_batch_groups)
            x, y = runtime.prepare(batch)
            digest = tensor_digest(x, y)
            for slot in SLOTS:
                loc['slot'] = slot
                chunks[slot].append(predict(runtime, slot, TASK, x, y, resources))
                if digest != tensor_digest(x, y):
                    raise ValueError('BATCH_MUTATED')
        logits = {slot: np.concatenate(chunks[slot]) for slot in SLOTS}
        scores = {slot: dict(accuracy=float(np.mean((z >= 0) == panel.labels)),
                             bce=bce(panel.labels, z)) for slot, z in logits.items()}
        return scores, logits

    check_source()
    summary['models'] = runtime.metadata()
    summary['location'] = dict(phase='dev-generate')
    dev = provider.panel('dev')
    check()
    raw = encode_panel(dev, recipe, source, backend)
    check()
    ref = store.write_dataset('dev-SKINNY', raw)
    check()
    summary['dev_panel'] = dict(ref=ref, sha256=sha256(raw).hexdigest(), digest=dev.digest)
    del raw
    snapshot()
    for epoch in range(1, tr.epochs + 1):
        check_source()
        losses = dict.fromkeys(SLOTS, 0.)
        for offset in range(tr.batches_per_epoch):
            cycle = (epoch - 1) * tr.batches_per_epoch + offset
            summary['location'] = dict(phase='train', epoch=epoch, cycle=cycle)
            check()
            batch = provider.train(cycle)
            x, y = runtime.prepare(batch)
            digest = tensor_digest(x, y)
            lr = learning_rate(recipe, cycle)
            row: dict[str, Any] = dict(epoch=epoch, cycle=cycle, lr=lr, groups=batch.groups,
                                      ciphertext_labels_sha256=batch.digest, input_sha256=digest,
                                      loss={}, updates_completed=[])
            pending.append(row)
            for slot in SLOTS:
                summary['location']['slot'] = slot
                resources.reserve('optimizer_updates', 1)
                summary['updates'][slot]['attempted'] += 1
                loss = runtime.update(slot, TASK, x, y, lr, resources)
                resources.complete('optimizer_updates', 1)
                summary['updates'][slot]['completed'] += 1
                row['updates_completed'].append(slot)
                if len(row['updates_completed']) == len(SLOTS):
                    summary['paired_completed'] += 1
                if math.isfinite(loss):
                    row['loss'][slot] = loss
                if cycle == 0 and slot == SLOTS[0]:
                    snapshot()  # do not wait for the D update to publish the first M update
                check()
                if not math.isfinite(loss) or tensor_digest(x, y) != digest:
                    raise ValueError('UPDATE_INVALID')
                losses[slot] += loss
            del batch, x, y
            if cycle == 0:
                snapshot()  # first full batch is visible, too
            if len(pending) >= 256:
                # 256 completed batches (131072 groups in the exact recipe).
                check_source()
                flush()
        flush()
        summary['location'] = dict(epoch=epoch, phase='dev-score')
        scores, _ = score(dev)
        summary['curves'].append(dict(epoch=epoch, dev=scores, train_bce={
            s: v / tr.batches_per_epoch for s, v in losses.items()}))
        snapshot()  # full DEV and counts survive before any best write
        for slot in SLOTS:
            val = scores[slot]['bce']
            if slot in best and val >= best[slot]['bce']:
                continue
            summary['location'] = dict(epoch=epoch, phase='best-save', slot=slot)
            check_source()
            meta = identity(recipe, source, backend, slot, epoch, val)
            raw = runtime.checkpoint(slot, meta)
            check()
            ref = store.write_model(slot, raw, replace=slot in best)
            best[slot] = dict(ref=ref, sha256=sha256(raw).hexdigest(), epoch=epoch, bce=val,
                              dev_digest=dev.digest)
            del raw
            snapshot()
    summary['location'] = dict(phase='persisted-best-reload')
    check_source()
    if set(best) != set(SLOTS):
        raise ValueError('BEST_INCOMPLETE')
    runtime.release()
    check()
    for slot in SLOTS:
        summary['location']['slot'] = slot
        selected = best[slot]
        if selected['ref'] != f'model/{slot}.pt':
            raise ValueError('BEST_REF')
        # Canonical run.json registration, exact current owner path, bounded size and SHA.
        raw = store.read_model(selected['ref'], selected['sha256'], MAX_CHECKPOINT)
        check()
        runtime.load_best(slot, raw, recipe, source, selected['epoch'], selected['bce'], check)
        del raw
        summary['persisted_reloads'][slot] = dict(selected)
        snapshot()
    summary['location'] = dict(phase='test-generate')
    check_source()
    test = provider.panel('test')
    summary['location'] = dict(phase='test-score')
    scores, logits = score(test)
    summary['location'] = dict(phase='test-save')
    raw = encode_panel(test, recipe, source, backend, logits=logits,
                       weights=summary['persisted_reloads'])
    check()
    ref = store.write_dataset('test-SKINNY', raw)
    check()
    summary['test_panel'] = dict(ref=ref, sha256=sha256(raw).hexdigest(), digest=test.digest)
    summary['test_scores'] = scores
    summary['m_minus_d'] = {metric: scores[SLOTS[0]][metric] - scores[SLOTS[1]][metric]
                            for metric in ('accuracy', 'bce')}
    snapshot()
    check_source()
    counts = resources.snapshot()
    totals = estimate(recipe)
    if any(counts[f'{kind}_{unit}'] != totals[unit]
           for kind in ('attempted', 'completed')
           for unit in ('real_blocks', 'optimizer_updates', 'forward_rows', 'null_words')):
        raise ValueError('COUNT_MISMATCH')
